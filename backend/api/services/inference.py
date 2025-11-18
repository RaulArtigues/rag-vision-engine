from backend.api.utils.base_model_qwen_vl_2b import QwenLocalModelLoader
from backend.api.services.support_index import SupportIndex
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
import os

def crop_patch(image_path: str, box) -> Image.Image:
    """
    Crop a patch from a given image using the specified bounding box.

    This function loads an image, resizes it to 224×224 pixels (matching CLIP
    vision encoder preprocessing), and then extracts the region defined by
    the provided box coordinates.

    Args:
        image_path (str): Path to the source image file.
        box (tuple or list): Coordinates defining the crop region
            in the format (left, upper, right, lower).

    Returns:
        Image.Image: A cropped RGB patch of the original image.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    return img.crop(box)

class RagVisionInference:
    """
    Orchestrates:
    - Visual retrieval over SupportIndex (CLIP).
    - Multimodal reasoning with Qwen2-VL-2B-Instruct.
    """

    def __init__(self, support_index: SupportIndex, qwen_local_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Main orchestration class for the RAG-Vision pipeline.

        Combines:
            - CLIP-based support retrieval via `SupportIndex`.
            - Multimodal reasoning using the Qwen2-VL-2B-Instruct vision-language model.

        Responsibilities:
            - Retrieve the most relevant support patches for a query image.
            - Construct the multimodal input sequence (images + prompts).
            - Execute generation on the Qwen VLM.
            - Return structured results including raw responses and evidence.
        """
        self.support_index = support_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
        qwen_local_dir = os.path.join(BACKEND_DIR, "artifacts", "qwen2_vl_2b_instruct")

        self.qwen_loader = QwenLocalModelLoader(
            local_dir=qwen_local_dir,
            device=self.device,
        )

        self.vlm_model = self.qwen_loader.get_model()
        self.vlm_processor = self.qwen_loader.get_processor()

    def build_rag_images(self, query_img: Image.Image, evidence: List[Dict[str, Any]], max_per_class: int = 3) -> List[Image.Image]:
        """
        Initialize the RAG-Vision inference engine.

        Args:
            support_index (SupportIndex): 
                The CLIP-based index responsible for retrieving relevant support patches.

            qwen_local_dir (Optional[str]): 
                Optional manual override for the Qwen model directory.
                If not provided, defaults to the backend artifacts path.

            device (Optional[str]): 
                Device used for inference ("cpu" or "cuda"). If None, automatically
                uses CUDA when available.
        """
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for e in evidence:
            label = e["ref"]["label"]
            by_label.setdefault(label, []).append(e)

        for label in by_label:
            by_label[label] = by_label[label][:max_per_class]

        rag_images: List[Image.Image] = [query_img]

        for label, items in by_label.items():
            for e in items:
                ref = e["ref"]
                img_patch = crop_patch(ref["image_path"], ref["box"])
                rag_images.append(img_patch)

        return rag_images

    def run(
        self,
        imageId: str,
        query_img: Image.Image,
        system_prompt: str,
        user_prompt: str,
        k_retrieval: int = 4,
        max_patches_per_class: int = 3,
        max_new_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Build the ordered sequence of images used as input for the VLM.

        The order is:
            1. The query image.
            2. A limited number (`max_per_class`) of retrieved patches for each class.

        Args:
            query_img (Image.Image): 
                The main user-provided image.

            evidence (List[Dict[str, Any]]): 
                Retrieved support entries containing patch paths, labels, and bounding boxes.

            max_per_class (int): 
                Maximum number of patches to include per class.

        Returns:
            List[Image.Image]: 
                A list beginning with the query image followed by cropped support patches.
        """
        class_scores, evidence = self.support_index.retrieve(
            query_img=query_img,
            k=k_retrieval,
            imageId=imageId,
        )

        rag_images = self.build_rag_images(
            query_img,
            evidence,
            max_per_class=max_patches_per_class,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in rag_images],
                    {"type": "text", "text": user_prompt.strip()},
                ],
            },
        ]

        prompt = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.vlm_processor(
            text=prompt,
            images=rag_images,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            out = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        resp = self.vlm_processor.batch_decode(out, skip_special_tokens=True)[0]

        return {
            "raw_response": resp,
            "class_scores": class_scores,
            "evidence": evidence,
        }