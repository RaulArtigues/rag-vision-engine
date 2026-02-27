from typing import Tuple, Optional
from io import BytesIO
from PIL import Image
import base64

class ImagePreprocessor:
    """
    Utility class responsible for decoding, preparing, and resizing input images
    for use in the RAG-Vision pipeline (CLIP retrieval + VLM reasoning).

    Core responsibilities:
        - Decode Base64 → PIL.Image
        - Preserve and return the original image resolution
        - Resize images to a fixed square shape suitable for vision encoders
        - Provide a clean hook for future preprocessing steps such as:
            * normalization
            * center-cropping
            * augmentation
    """
    def __init__(self, target_size: Optional[int] = 224):
        """
        Initialize the preprocessor.

        Args:
            target_size (int or None):
                If set to an integer, the image will be resized to 
                (target_size, target_size).  
                If set to None, the image is returned in its original resolution.
        """
        self.target_size = target_size

    def decode_base64_to_pil(self, b64_str: str) -> Image.Image:
        """
        Decode a Base64-encoded image into a PIL Image.

        Args:
            b64_str (str):
                A Base64-encoded image string. May or may not include 
                a data URI prefix such as "data:image/png;base64,...".

        Returns:
            Image.Image:
                A PIL RGB image decoded from the Base64 input.
        """
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        return img

    def preprocess(self, b64_image: str) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Perform the full preprocessing pipeline on a Base64 input image.

        Steps:
            1. Decode Base64 → PIL Image
            2. Capture the original dimensions (width, height)
            3. Resize the image to (target_size, target_size) if configured

        Args:
            b64_image (str):
                Base64 string representing the image to preprocess.

        Returns:
            Tuple[Image.Image, Tuple[int, int]]:
                - processed_image: The resized (or original) PIL image.
                - original_size: A tuple (width, height) representing the 
                  original image resolution before preprocessing.
        """
        img = self.decode_base64_to_pil(b64_image)
        orig_size = img.size

        if self.target_size is not None:
            img = img.resize((self.target_size, self.target_size), Image.BICUBIC)

        return img, orig_size