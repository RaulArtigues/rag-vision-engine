from backend.api.utils.base_model_clip import CLIPLocalModelLoader
from backend.api.routers.events.logging import LoggerManager
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np
import torch
import math
import os

class SupportIndex:
    """
    Builds, stores, and retrieves CLIP patch embeddings for the support dataset.

    The SupportIndex is responsible for:
        - Scanning support class directories.
        - Extracting local image patches.
        - Computing CLIP visual embeddings for every patch.
        - Storing a large matrix of embeddings + associated metadata.
        - Performing nearest-neighbor retrieval for query patches.

    Notes:
        - The index is constructed *per request*; `imageId` is not part of
          initialization and must be passed to `.build()` and `.retrieve()`.
        - CLIP model and processor are loaded once via CLIPLocalModelLoader.
    """
    def __init__(
        self,
        class_dirs: Dict[str, str],
        clip_local_dir: Optional[str],
        res: int,
        patch_size: int,
        device: Optional[str] = None,
        ):
        """
        Initialize the SupportIndex.

        Args:
            class_dirs (dict):
                Mapping of class_name → directory containing support images.

            clip_local_dir (str or None):
                Directory where CLIP weights are stored.
                If None, the default artifacts path is used.

            res (int):
                Resolution (width=height) to which all support images
                are resized before patch extraction.

            patch_size (int):
                Size (in pixels) of each square patch extracted from the
                resized support images.

            device (str or None):
                Device for inference ("cpu" or "cuda").
                Defaults to CUDA when available.
        """
        self.class_dirs = class_dirs
        self.RES = res
        self.PATCH = patch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
        clip_local_dir = os.path.join(BACKEND_DIR, "artifacts", "clip_vit_base_patch32")

        LoggerManager.log_formatter(
            f"SupportIndex initialized. Using CLIP dir: {clip_local_dir}",
            "", 2000, level="INFO")

        self.loader = CLIPLocalModelLoader(local_dir=clip_local_dir, device=self.device)
        self.clip_model = self.loader.get_model()
        self.clip_processor = self.loader.get_processor()
        self.VISION_DIM = self.loader.get_vision_dim()

        self.vectors: Optional[np.ndarray] = None
        self.meta: List[Dict[str, Any]] = []

        self.VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def _is_image(self, path: str) -> bool:
        """
        Check whether a file path corresponds to a valid image.

        Args:
            path (str): Path to the file.

        Returns:
            bool: True if the file extension is image-like, False otherwise.
        """
        return os.path.splitext(path)[1].lower() in self.VALID_EXTS

    def _patch_coords(self):
        """
        Compute all patch bounding boxes for the configured RES/PATCH grid.

        Returns:
            list of tuple:
                Bounding boxes in the form (left, top, right, bottom)
                for each patch across the full resized image.
        """
        return [
            (x * self.PATCH, y * self.PATCH, x * self.PATCH + self.PATCH, y * self.PATCH + self.PATCH)
            for y in range(self.RES // self.PATCH)
            for x in range(self.RES // self.PATCH)
        ]

    def _get_patch_embeddings(self, img: Image.Image) -> torch.Tensor:
        """
        Compute CLIP patch embeddings for a single processed image.

        Steps:
            - Encode full image via CLIP vision model
            - Drop the CLS token
            - L2-normalize patch tokens
            - Reshape them into an (S × S × dim) grid

        Args:
            img (PIL.Image):
                Already-resized image to (RES × RES).

        Returns:
            torch.Tensor:
                Tensor of shape (grid_height, grid_width, vision_dim).
        """
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
            tokens = out.last_hidden_state[:, 1:, :]

        tokens = tokens.squeeze(0)
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)

        s = int(math.sqrt(tokens.shape[0]))

        return tokens.reshape(s, s, self.VISION_DIM)

    def _process_folder(self, folder: str, label: str, imageId: str):
        """
        Process a support class directory by extracting patches,
        computing embeddings, and storing metadata.

        Args:
            folder (str): Path to the class directory.
            label (str): Class label for all images inside the folder.
            imageId (str): ID used for logging and traceability.
        """
        boxes = self._patch_coords()

        LoggerManager.log_formatter(
            f"Processing class '{label}' in folder: {folder}",
            imageId, 2000, level="INFO")

        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if not self._is_image(path):
                continue

            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                LoggerManager.log_formatter(
                    f"Invalid image skipped: {path} ({e})",
                    imageId, 2001, level="WARNING")
                continue

            img_r = img.resize((self.RES, self.RES), Image.BICUBIC)
            grid = self._get_patch_embeddings(img_r).detach().cpu().numpy()
            vecs = grid.reshape(-1, self.VISION_DIM).astype(np.float32)

            if self.vectors is None:
                self.vectors = vecs
            else:
                self.vectors = np.vstack([self.vectors, vecs])

            for idx, box in enumerate(boxes):
                self.meta.append({
                    "label": label,
                    "image_path": path,
                    "patch_idx": idx,
                    "box": box,
                })

            LoggerManager.log_formatter(
                f"Added support image {fname} ({label}) → {vecs.shape[0]} patches",
                imageId, 2000, level="INFO")

    def build(self, imageId: str):
        """
        Build the full support index by processing all class directories.

        Args:
            imageId (str):
                Request-specific ID for logging.

        Raises:
            RuntimeError: If no valid support vectors could be built.
        """
        if self.vectors is not None:
            LoggerManager.log_formatter(
                "SupportIndex already built — skipping.",
                imageId, 2000, level="INFO")
            return

        LoggerManager.log_formatter(
            "Building SupportIndex from support classes...",
            imageId, 2000, level="INFO")

        for label, folder in self.class_dirs.items():
            if not os.path.isdir(folder):
                LoggerManager.log_formatter(
                    f"Support folder not found: {folder}",
                    imageId, 2001, level="WARNING")
                continue

            self._process_folder(folder, label, imageId)

        if self.vectors is None:
            LoggerManager.log_formatter(
                "ERROR: No support vectors built.",
                imageId, 4000, level="ERROR")
            
            raise RuntimeError("SupportIndex: No vectors built.")

        LoggerManager.log_formatter(
            f"SupportIndex completed. Total vectors: {self.vectors.shape[0]} | Dim: {self.vectors.shape[1]}",
            imageId, 2000, level="INFO")

    def retrieve(self, query_img: Image.Image, k: int, imageId: str):
        """
        Retrieve the top-K most similar support patches for each patch of the query image.

        Steps:
            1. Resize query image to RES × RES.
            2. Extract all query patch embeddings.
            3. Compute cosine similarity between every query patch and every support patch.
            4. Aggregate class-level similarity scores.
            5. Return sorted evidence list.

        Args:
            query_img (PIL.Image):
                Input image from the user.

            k (int):
                Number of top patches to retrieve per query patch.

            imageId (str):
                Identifier used for logging.

        Returns:
            tuple:
                - class_score (dict): Aggregated similarity score per class.
                - evidence (list): Detailed retrieval results including:
                    {
                        "query_patch_idx": int,
                        "sim": float,
                        "ref": {
                            "label": str,
                            "image_path": str,
                            "patch_idx": int,
                            "box": (tuple)
                        }
                    }

        Raises:
            RuntimeError:
                If the index has not been built prior to retrieval.
        """
        if self.vectors is None or not self.meta:
            LoggerManager.log_formatter(
                "ERROR: SupportIndex not built before retrieval.",
                imageId, 4000, level="ERROR")
            
            raise RuntimeError("SupportIndex: build() must be called first.")

        img_r = query_img.resize((self.RES, self.RES), Image.BICUBIC)
        grid = self._get_patch_embeddings(img_r).detach().cpu().numpy()
        vecs = grid.reshape(-1, self.VISION_DIM).astype(np.float32)

        S = self.vectors.astype(np.float32)

        class_score = {}
        evidence = []

        for q_idx in range(vecs.shape[0]):
            q = vecs[q_idx]
            sims = S @ q
            top = sims.argsort()[-k:][::-1]

            for idx in top:
                sim = float(sims[idx])
                ref = self.meta[idx]
                label = ref["label"]

                class_score[label] = class_score.get(label, 0.0) + sim
                evidence.append({
                    "query_patch_idx": q_idx,
                    "sim": sim,
                    "ref": ref,
                })

        evidence.sort(key=lambda e: e["sim"], reverse=True)

        LoggerManager.log_formatter(
            f"Retrieval completed — patches compared: {vecs.shape[0]}, evidence returned: {len(evidence)}",
            imageId, 2000, level="INFO")

        return class_score, evidence