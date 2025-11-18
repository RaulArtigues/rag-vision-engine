from transformers import AutoProcessor, AutoModelForVision2Seq
from backend.api.routers.events.logging import LoggerManager
from backend.api.utils import properties
from typing import Optional
import torch
import os

class QwenLocalModelLoader:
    """
    Singleton-style loader for a locally stored Qwen2-VL-2B-Instruct model.

    This loader handles:
        - Checking whether the model already exists locally.
        - Downloading it from Hugging Face Hub if missing.
        - Loading the model only once into memory.
        - Exposing both the model and processor for downstream tasks.

    Attributes:
        local_dir (str): Directory where the model is stored or downloaded.
        device (str): Device used to load the model ('cpu' or 'cuda').
        vlm_model (AutoModelForVision2Seq): Loaded Qwen model instance.
        vlm_processor (AutoProcessor): Processor used for data preprocessing.
    """
    _instance: Optional["QwenLocalModelLoader"] = None

    HF_MODEL_ID = properties.HF_MODEL_ID_VLM

    def __new__(cls, *args, **kwargs):
        """
        Create and return a single shared instance of the loader.

        Ensures:
            Only one instance of `QwenLocalModelLoader` is ever created.
        """
        if cls._instance is None:
            cls._instance = super(QwenLocalModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, local_dir: str, device: str = None):
        """
        Initialize the Qwen model loader.

        Args:
            local_dir (str): Directory where the model is/will be stored.
            device (str, optional): Device to load the model onto.
                If None, automatically uses CUDA when available.

        Notes:
            If the loader was already initialized, this method exits early.
        """
        if getattr(self, "_initialized", False):
            return

        self.local_dir = local_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.local_dir, exist_ok=True)

        LoggerManager.log_formatter(
            f"Initializing Qwen Loader. Local path: {self.local_dir}",
            "", 2000, level="INFO")

        self.vlm_model, self.vlm_processor = self._load_or_download_model()

        LoggerManager.log_formatter(
            f"Qwen model loaded on device: {self.device}",
            "", 2000, level="INFO")

        self._initialized = True

    def _is_model_downloaded(self) -> bool:
        """
        Check whether the required model files are present locally.

        Returns:
            bool: True if configuration files and weight files exist, False otherwise.
        """
        files = os.listdir(self.local_dir)

        required = ["config.json", "preprocessor_config.json"]

        for req in required:
            if req not in files:
                return False

        has_weights = any(
            f.startswith("pytorch_model") or (f.startswith("model") and f.endswith(".safetensors"))
            for f in files
        )

        return has_weights

    def _download_model(self):
        """
        Download the Qwen model and processor from Hugging Face Hub.

        Saves all necessary files into the local directory.
        """
        LoggerManager.log_formatter(
            "Qwen model not found locally — downloading...",
            "", 2001, level="WARNING")

        processor = AutoProcessor.from_pretrained(self.HF_MODEL_ID)
        processor.save_pretrained(self.local_dir)

        model = AutoModelForVision2Seq.from_pretrained(
            self.HF_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(self.local_dir)

        LoggerManager.log_formatter(
            "Qwen model download complete.",
            "", 2000, level="INFO")

    def _load_or_download_model(self):
        """
        Load the Qwen model from disk, or download it if missing.

        Returns:
            tuple:
                - AutoModelForVision2Seq: Loaded Qwen model instance.
                - AutoProcessor: Processor for preprocessing inputs.
        """
        if not self._is_model_downloaded():
            self._download_model()
        else:
            LoggerManager.log_formatter(
                f"Found existing Qwen model. Loading from {self.local_dir}",
                "", 2000, level="INFO")

        processor = AutoProcessor.from_pretrained(self.local_dir)

        model = AutoModelForVision2Seq.from_pretrained(
            self.local_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

        LoggerManager.log_formatter(
            f"Qwen model successfully loaded from {self.local_dir}",
            "", 2000, level="INFO")

        return model, processor

    def get_model(self):
        """
        Retrieve the loaded Qwen model.

        Returns:
            AutoModelForVision2Seq: The loaded model instance.
        """
        return self.vlm_model

    def get_processor(self):
        """
        Retrieve the Qwen processor.

        Returns:
            AutoProcessor: The preprocessing and tokenization handler.
        """
        return self.vlm_processor