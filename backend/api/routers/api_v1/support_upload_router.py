from fastapi import APIRouter, Request, UploadFile, File, Form
from backend.api.routers.events.logging import LoggerManager
import shutil
import json
import os

router = APIRouter()

IS_HF = "SPACE_ID" in os.environ

if IS_HF:
    BASE_DATA_DIR = "/data"
else:
    BASE_DATA_DIR = os.path.abspath("../../data")

SUPPORT_ROOT = os.path.join(BASE_DATA_DIR, "support")

os.makedirs(SUPPORT_ROOT, exist_ok=True)

def save_image_bytes(class_name: str, filename: str, img_bytes: bytes):
    """
    Save raw image bytes into the support dataset structure.

    This helper function stores an uploaded image in the correct class-specific
    directory under the SUPPORT_ROOT folder.

    Args:
        class_name (str):
            Name of the class the image belongs to. A folder for this class
            will be created if it does not already exist.

        filename (str):
            Name of the file to save, typically uniquely generated based
            on class name and incremental index.

        img_bytes (bytes):
            Raw bytes of the uploaded image file.

    Returns:
        str:
            Absolute path to the saved image file.
    """
    class_dir = os.path.join(SUPPORT_ROOT, class_name)
    os.makedirs(class_dir, exist_ok=True)

    filepath = os.path.join(class_dir, filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    return filepath

@router.post("/support/upload/image")
async def upload_single_image(
    request: Request,
    className: str = Form(...),
    classes: str = Form(...),
    index: int = Form(...), 
    file: UploadFile = File(...)
    ):
    """
    Upload a single support image with automatic class and index handling.

    This endpoint is used to dynamically build or update the support dataset
    used by the RAG-Vision pipeline. Images are uploaded one at a time with
    metadata describing their class assignment and upload order.

    Workflow:
        1. Parse and validate index.
        2. Parse the full list of classes from JSON.
        3. If index == 0:
            - Reset (wipe) the entire support folder.
            - Recreate the support directory structure.
        4. Validate that the assigned className is part of classes.
        5. Read and save the uploaded image bytes.

    Args:
        request (Request):
            The FastAPI request object (not used directly but required).

        className (str):
            The name of the class to associate with this image.

        classes (str):
            A JSON-encoded list of all classes. Required for validation and
            directory reset operations.

        index (int):
            Upload index of the image. If index == 0, the support directory
            is reset and rebuilt from scratch.

        file (UploadFile):
            The uploaded image file (JPEG/PNG/etc).

    Returns:
        dict:
            A structured response with:
                - success (bool)
                - className (str)
                - index (int)
                - filename (str)
            Or an error message if validation fails.

    Error Handling:
        - Invalid index format
        - Invalid JSON list for 'classes'
        - Unknown class assignment
        - Failure to read uploaded file
    """
    try:
        index_int = int(index)
    except:
        return {"success": False, "error": f"Invalid index: {index}"}

    try:
        all_classes = json.loads(classes)
        assert isinstance(all_classes, list)
    except Exception:
        return {"success": False, "error": "Invalid 'classes' format. Must be JSON list."}

    if index_int == 0:
        LoggerManager.log_formatter(
            f"[RESET] index=0 → cleaning support directory",
            "", 2000, level="INFO"
        )

        if os.path.exists(SUPPORT_ROOT):
            shutil.rmtree(SUPPORT_ROOT)
        os.makedirs(SUPPORT_ROOT, exist_ok=True)

        for cls in all_classes:
            os.makedirs(os.path.join(SUPPORT_ROOT, cls), exist_ok=True)

    if className not in all_classes:
        return {"success": False, "error": f"Image assigned to unknown class '{className}'"}

    try:
        img_bytes = await file.read()
    except Exception as e:
        return {"success": False, "error": f"Cannot read upload file: {e}"}

    filename = f"{className}_{index_int}.jpg"
    save_image_bytes(className, filename, img_bytes)

    LoggerManager.log_formatter(
        f"[UPLOAD] class={className} index={index_int} file={file.filename}",
        "", 2000, level="INFO"
    )

    return {
        "success": True,
        "className": className,
        "index": index_int,
        "filename": filename
    }