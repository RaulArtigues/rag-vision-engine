import base64
import os
import re

def read_file_as_b64(file_input):
    """
    Read a file from multiple possible input formats and return its Base64-encoded string.

    This utility is intentionally flexible to support different UI frameworks
    (Streamlit, Gradio, custom UIs, JSON-based inputs, etc.), which may represent
    uploaded files with different structures.

    Supported input formats:
        - An object with a `.name` attribute (e.g., some file wrappers)
        - A dictionary with a `"name"` key
        - A raw file path string

    Steps performed:
        1. Resolves the actual filesystem path from the input.
        2. Validates that the path exists and points to a readable file.
        3. Reads the file as bytes.
        4. Encodes its contents as Base64.
        5. Optionally sanitizes the Base64 string by removing any unexpected characters.

    Args:
        file_input (Any):
            Input representing a file. Can be:
            - dict with {"name": "/path/to/file.jpg"}
            - object with attribute `.name`
            - string "/path/to/file.jpg"

    Returns:
        str:
            A Base64-encoded string representing the file contents.

    Raises:
        ValueError:
            If `file_input` is in an unsupported format.
        FileNotFoundError:
            If the resolved path does not exist or cannot be accessed.

    Notes:
        - The Base64 cleanup step (`re.sub`) ensures compatibility across clients
          by removing any whitespace or unexpected characters that may be introduced
          by certain file encoders or transport layers.
    """
    if isinstance(file_input, dict) and "name" in file_input:
        path = file_input["name"]
    elif hasattr(file_input, "name"):
        path = file_input.name
    elif isinstance(file_input, str):
        path = file_input
    else:
        raise ValueError(f"Invalid file input format: {file_input}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File does not exist or cannot be accessed: {path}")

    with open(path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode("utf-8")

    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", b64)

    return b64