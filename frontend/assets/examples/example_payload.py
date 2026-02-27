from frontend.assets.examples.prompts.prompts import SYSTEM_PROMPT, USER_PROMPT
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE = os.path.abspath(os.path.join(CURRENT_DIR))

DIRTY_DIR = os.path.join(BASE, "support", "dirty")
CLEAN_DIR = os.path.join(BASE, "support", "clean")
QUERY_IMG = os.path.join(BASE, "query_image", "query.jpg")

def list_images(directory):
    """
    Return a sorted list of absolute file paths for all images in a directory.

    The function filters files by typical image extensions (jpg, jpeg, png).
    It also handles missing directories gracefully by returning an empty list.

    Args:
        directory (str):
            Path to the folder containing example images.

    Returns:
        list[str]:
            Sorted list of absolute paths to image files found in the directory.
            Returns an empty list if the directory does not exist.
    """
    if not os.path.exists(directory):
        return []

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort()

    return files


EXAMPLE_DIRTY = list_images(DIRTY_DIR)
EXAMPLE_CLEAN = list_images(CLEAN_DIR)

def build_example_payload():
    """
    Build a full example payload used by the RAG-Vision frontend demo.

    This payload is typically sent to the backend as an example request
    when users load the demo interface. It includes:
        - Example class names (“dirty”, “clean”)
        - Example image paths for each class
        - Predefined system and user prompts
        - Example query image

    Returns:
        dict:
            A dictionary bundling all example data required to run a sample
            inference request in the frontend.
    """
    return {
        "class1": "dirty",
        "class2": "clean",
        "files_class1": EXAMPLE_DIRTY,
        "files_class2": EXAMPLE_CLEAN,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "query_image": QUERY_IMG,
    }

def get_example_classes():
    """
    Return example class labels used in the frontend demo.

    Returns:
        tuple[str, str]:
            The two example classes: (“dirty”, “clean”).
    """
    return "dirty", "clean"

def get_example_prompts():
    """
    Retrieve predefined system and user prompts used in the example payload.

    Returns:
        tuple[str, str]:
            A tuple containing (system_prompt, user_prompt).
    """
    return SYSTEM_PROMPT, USER_PROMPT

def get_example_images():
    """
    Provide paths to the example images used in the demo interface.

    Returns:
        tuple[list[str], list[str], str]:
            - List of sample dirty images
            - List of sample clean images
            - Path to the example query image
    """
    return EXAMPLE_DIRTY, EXAMPLE_CLEAN, QUERY_IMG