from pydantic import BaseModel, Field
from typing import List

class SupportUploadMeta(BaseModel):
    """
    Metadata schema for support image upload operations.

    This model provides contextual information about the class associated
    with a support image, the list of all available classes, and the index
    position used during upload or dataset organization.
    """
    className: str = Field(..., description="Name of the class to which the uploaded support image belongs.")
    classes: List[str] = Field(..., description="List of all available class names defined in the system or dataset.")
    index: int = Field(..., description="Numeric index used to order or place the uploaded support image within its class.")