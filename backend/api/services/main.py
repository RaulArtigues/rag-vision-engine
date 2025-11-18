from backend.api.services.preprocessor import ImagePreprocessor
from backend.api.services.inference import RagVisionInference
from backend.api.services.postprocessor import Postprocessor
from backend.api.routers.events.logging import LoggerManager
from backend.api.services.support_index import SupportIndex
from typing import Dict, Any, Optional
import traceback
import os

class RagVisionService:
    """
    High-level orchestration service for the complete RAG-Vision pipeline.

    This service is a singleton responsible for:
        - Warming up the Vision-Language Model (Qwen2-VL-2B-Instruct).
        - Discovering support class directories.
        - Initializing CLIP-based SupportIndex for retrieval.
        - Preprocessing input images.
        - Running the RAG-Vision inference engine.
        - Postprocessing the model response.
        - Handling errors and structured output formatting.

    It acts as the main entry point for the RAG-Vision API.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    SUPPORT_ROOT = os.path.join(BACKEND_DIR, "data", "support")

    _instance: Optional["RagVisionService"] = None

    def __new__(cls, *args, **kwargs):
        """
        Enforce singleton behavior for RagVisionService.

        Ensures only one instance is created and reused throughout the API lifecycle.
        """
        if cls._instance is None:
            cls._instance = super(RagVisionService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the RAG-Vision service.

        Notes:
            - Initialization is skipped if already performed.
            - Only the VLM (Qwen) is loaded during warm-up.
            - SupportIndex and CLIP are intentionally not loaded at startup
              to minimize memory consumption and startup time.
        """
        if getattr(self, "_initialized", False):
            return

        self.support_index: Optional[SupportIndex] = None
        self.engine: Optional[RagVisionInference] = None
        self.preprocessor: Optional[ImagePreprocessor] = None

        self._warmup_vlm()

        self._initialized = True

    def _warmup_vlm(self):
        """
        Load only the Vision-Language Model (Qwen2-VL-2B-Instruct) into memory.

        This function runs once at server startup to reduce latency
        during the first inference request.

        Notes:
            - CLIP and the SupportIndex are NOT initialized here.
            - A dummy SupportIndex is used since Qwen loader requires it.
        """
        warm_image_id = "INIT-VLM"

        LoggerManager.log_formatter(
            "Warming up: Loading Qwen2-VL model into memory...",
            warm_image_id, 2000, level="INFO")

        try:
            dummy_index = SupportIndex(
                class_dirs={},
                clip_local_dir="",
                res=0,
                patch_size=0
            )

            self.engine = RagVisionInference(dummy_index)

            LoggerManager.log_formatter(
                "Qwen2-VL model loaded successfully and ready.",
                warm_image_id, 2000, level="INFO")

        except Exception as e:
            LoggerManager.log_formatter(
                f"VLM warm-up FAILED: {e}",
                warm_image_id, 4000, level="ERROR")
            
            raise

    def _discover_class_dirs(self, imageId: str):
        """
        Scan the support directory and discover available class folders.

        Args:
            imageId (str): Identifier used for logging and tracking.

        Returns:
            dict: Mapping of class names → absolute directory paths.

        Raises:
            RuntimeError: If the support root or class folders do not exist.
        """
        root = self.SUPPORT_ROOT

        if not os.path.isdir(root):
            LoggerManager.log_formatter(
                f"Support root not found: {root}", imageId, 4000, level="ERROR")
            raise RuntimeError(f"Support root not found: {root}")

        class_dirs = {
            name: os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        }

        if not class_dirs:
            LoggerManager.log_formatter(
                f"No support class folders found under: {root}",
                imageId, 4000, level="ERROR")
            raise RuntimeError(f"No class folders found under {root}")

        LoggerManager.log_formatter(
            f"Support classes discovered: {list(class_dirs.keys())}",
            imageId, 2000, level="INFO")

        return class_dirs

    def analyze(
        self,
        b64_image: str,
        flagKey: str,
        system_prompt: str,
        user_prompt: str,
        *,
        k_retrieval: int,
        max_patches_per_class: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        input_resolution: int,
        support_res: int,
        support_patch_size: int,
        support_clip_local_dir: Optional[str],
        imageId: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete RAG-Vision inference pipeline on a single image request.

        Steps:
            1. Discover available support class directories.
            2. Initialize SupportIndex and run CLIP-based patch retrieval.
            3. Preprocess the base64-encoded input image.
            4. Execute multimodal reasoning via RagVisionInference.
            5. Postprocess VLM output into structured, meaningful fields.
            6. Handle and report errors gracefully.

        Args:
            b64_image (str): Base64-encoded representation of the input image.
            flagKey (str): Identifier for routing/logic within the RAG pipeline.
            system_prompt (str): System-level instruction for the VLM.
            user_prompt (str): User query or request.
            k_retrieval (int): Number of CLIP patches to retrieve.
            max_patches_per_class (int): Maximum number of patches per class.
            temperature (float): Sampling temperature for text generation.
            top_p (float): Nucleus sampling parameter.
            max_new_tokens (int): Maximum tokens the VLM can generate.
            input_resolution (int): Image preprocessing resolution.
            support_res (int): Resolution applied to support images.
            support_patch_size (int): Patch size for CLIP retrieval.
            support_clip_local_dir (Optional[str]): Directory containing CLIP weights.
            imageId (Optional[str]): Optional image identifier for tracing.

        Returns:
            dict: A structured response containing:
                - success (bool)
                - flag (bool or None)
                - explanation (str)
                - class_scores (dict)
                - raw_response (str)
                - original_size (tuple)
                - flagKey (str)
                - imageId (str)
                - error (str, if failure)
        """

        imageId = imageId

        postprocessor = Postprocessor(flag_key=flagKey)
        self.preprocessor = ImagePreprocessor(target_size=input_resolution)

        try:
            LoggerManager.log_formatter(
                f"Starting RAG-Vision pipeline with flagKey='{flagKey}'",
                imageId, 2000, level="INFO")

            class_dirs = self._discover_class_dirs(imageId=imageId)
            self.support_index = SupportIndex(
                class_dirs=class_dirs,
                clip_local_dir=support_clip_local_dir,
                res=support_res,
                patch_size=support_patch_size,
            )
            self.support_index.build(imageId=imageId)

            self.engine.support_index = self.support_index

            query_img, orig_size = self.preprocessor.preprocess(
                b64_image=b64_image
            )

            result = self.engine.run(
                imageId=imageId,
                query_img=query_img,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                k_retrieval=k_retrieval,
                max_patches_per_class=max_patches_per_class,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            raw = result["raw_response"]

            parsed = postprocessor.parse(raw_response=raw)

            LoggerManager.log_formatter(
                "RAG-Vision pipeline completed successfully.",
                imageId, 2000, level="INFO")

            return {
                "success": True,
                "flag": parsed["flag"],
                "explanation": parsed["explanation"],
                "class_scores": result["class_scores"],
                "raw_response": raw,
                "original_size": orig_size,
                "flagKey": flagKey,
                "imageId": imageId,
            }

        except Exception as e:
            LoggerManager.log_formatter(
                f"Pipeline ERROR: {e}",
                imageId, 5000, level="ERROR")
            
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "flagKey": flagKey,
                "imageId": imageId,
            }