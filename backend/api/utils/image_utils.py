from io import BytesIO
from PIL import Image
import base64

def decode_base64_to_image(b64_string: str) -> Image.Image:
    """
    Decode a Base64 string into a PIL Image.

    This function supports Base64 strings that may include a data URI prefix
    (e.g., "data:image/png;base64,...") and strips it if present.

    Args:
        b64_string (str): The Base64-encoded image data.

    Returns:
        Image.Image: A PIL Image object in RGB format.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)

    return Image.open(BytesIO(img_bytes)).convert("RGB")

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file into a Base64 string.

    Args:
        image_path (str): Path to the image file to encode.

    Returns:
        str: A UTF-8 Base64-encoded representation of the image.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")