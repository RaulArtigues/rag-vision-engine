from pydantic import BaseModel, Field
from typing import Optional

class RagVisionInput(BaseModel):
    """
    Input schema for the RAG Vision Engine.

    Defines all parameters required to process an image-based RAG request,
    including encoded images, prompts, retrieval settings, and generation
    configuration.
    """
    encodedImage: str = Field(..., description="Base64-encoded representation of the input image. May include a data URI prefix.")
    systemPrompt: str = Field(..., description="System-level instruction that defines behavior and sets constraints before user prompt execution.")
    userPrompt: str = Field(..., description="User instruction or query that the model should respond to using RAG and visual context.")
    kRetrieval: int = Field(..., description="Number of relevant patches/items to retrieve during RAG similarity lookup.")
    maxPatchesPerClass: int = Field(..., description="Maximum number of visual patches that can be retrieved per semantic class category.")
    maxNewTokens: int = Field(..., description="Maximum number of tokens the generative model is allowed to produce in its response.")
    inputResolution: int = Field(..., description="Resolution to which the input image will be resized before embedding or patch extraction.")
    supportRes: int = Field(..., description="Resolution applied to support images used during retrieval or CLIP-based similarity matching.")
    supportPatchSize: int = Field(..., description="Patch size (in pixels) extracted from support images for CLIP similarity scoring.")
    temperature: float = Field(..., description="Sampling temperature for generation. Higher values increase randomness.")
    topP: float = Field(..., description="Nucleus sampling parameter determining probability mass used during generation.")
    supportClipLocalDir: Optional[str] = Field(None, description="Optional local directory path containing the CLIP model used for support retrieval.")