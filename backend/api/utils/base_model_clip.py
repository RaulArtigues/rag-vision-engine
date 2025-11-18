from backend.api.routers.events.logging import LoggerManager
from transformers import CLIPModel, CLIPProcessor
from backend.api.utils import properties
from typing import Optional
import torch
import os

class CLIPLocalModelLoader:
    """
    Singleton-style loader for a locally stored CLIP model.

    This class handles:
        - Verifying whether the CLIP model is already present locally.
        - Downloading the model and processor if missing.
        - Loading the model only once into memory.
        - Exposing the model, processor, and the computed vision embedding dimension.

    Attributes:
        local_dir (str): Path where the CLIP model is stored or downloaded.
        device (str): Device on which the model is loaded ('cuda' or 'cpu').
        clip_model (CLIPModel): Loaded CLIP model instance.
        clip_processor (CLIPProcessor): Processor associated with the CLIP model.
        vision_dim (int): Dimensionality of the vision model embeddings.
    """
    _instance: Optional["CLIPLocalModelLoader"] = None

    HF_MODEL_ID = properties.HF_MODEL_ID_CLIP

    def __new__(cls, *args, **kwargs):
        """
        Create and return a single shared instance of the loader.

        Ensures:
            Only one instance of `CLIPLocalModelLoader` is ever created.
        """
        if cls._instance is None:
            cls._instance = super(CLIPLocalModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, local_dir: str, device: str = None):
        """
        Initialize the CLIP model loader.

        Args:
            local_dir (str): Directory where the model should be stored or retrieved from.
            device (str, optional): Device override ('cpu' or 'cuda'). If None, auto-selected.

        Notes:
            Initialization is skipped if the instance has already been created.
        """
        if getattr(self, "_initialized", False):
            return

        self.local_dir = local_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.local_dir, exist_ok=True)

        LoggerManager.log_formatter(
            f"Initializing CLIP Loader. Local path: {self.local_dir}",
            "", 2000, level="INFO")

        self.clip_model, self.clip_processor = self._load_or_download_clip_model()
        self.vision_dim = self.clip_model.vision_model.config.hidden_size

        LoggerManager.log_formatter(
            f"CLIP model loaded on device: {self.device}",
            "", 2000, level="INFO")
        
        LoggerManager.log_formatter(
            f"CLIP vision dimension: {self.vision_dim}",
            "", 2000, level="INFO")

        self._initialized = True

    def _is_model_downloaded(self) -> bool:
        """
        Check if the required CLIP model files exist in the local directory.

        Returns:
            bool: True if all required configuration and weight files exist, False otherwise.
        """
        required = [
            "config.json",
            "preprocessor_config.json"
        ]

        files = os.listdir(self.local_dir)

        for req in required:
            if req not in files:
                return False

        has_weights = any(
            w in files for w in ("pytorch_model.bin", "model.safetensors")
        )

        return has_weights

    def _download_model(self):
        """
        Download the CLIP model and processor from Hugging Face Hub.

        The model is saved locally inside the `local_dir` path.
        """
        LoggerManager.log_formatter(
            "CLIP model not found locally — downloading...",
            "", 2001, level="WARNING")

        model = CLIPModel.from_pretrained(self.HF_MODEL_ID)
        processor = CLIPProcessor.from_pretrained(self.HF_MODEL_ID)

        model.save_pretrained(self.local_dir)
        processor.save_pretrained(self.local_dir)

        LoggerManager.log_formatter(
            "CLIP model download complete.",
            "", 2000, level="INFO")

    def _load_or_download_clip_model(self):
        """
        Load the local CLIP model or download it if missing.

        Returns:
            tuple:
                - CLIPModel: Loaded CLIP model.
                - CLIPProcessor: Associated processor used for preprocessing inputs.
        """
        if not self._is_model_downloaded():
            self._download_model()
        else:
            LoggerManager.log_formatter(
                f"Found existing CLIP model. Loading from {self.local_dir}",
                "", 2000, level="INFO")

        model = CLIPModel.from_pretrained(self.local_dir).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(self.local_dir)

        LoggerManager.log_formatter(
            f"CLIP model successfully loaded from {self.local_dir}","", 2000, level="INFO")

        return model, processor

    def get_model(self):
        """
        Retrieve the loaded CLIP model.

        Returns:
            CLIPModel: The loaded model instance.
        """
        return self.clip_model

    def get_processor(self):
        """
        Retrieve the CLIP processor.

        Returns:
            CLIPProcessor: The preprocessing and tokenization handler.
        """
        return self.clip_processor

    def get_vision_dim(self):
        """
        Retrieve the embedding dimension of the CLIP vision encoder.

        Returns:
            int: Vision feature dimension size.
        """
        return self.vision_dim