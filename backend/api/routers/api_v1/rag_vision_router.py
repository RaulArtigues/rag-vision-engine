from backend.api.schema.ragvision_outputs import RagVisionOutput
from backend.api.schema.ragvision_inputs import RagVisionInput
from backend.api.routers.events.logging import LoggerManager
from backend.api.services.main import RagVisionService
from fastapi.responses import JSONResponse
from fastapi import APIRouter
import uuid
import time

router = APIRouter()
rag_service = RagVisionService()

@router.post("/ragvision/invocations", response_model=RagVisionOutput)
async def rag_vision_inference(payload: RagVisionInput):
    """
    Execute a full RAG-Vision pipeline inference request.

    This endpoint orchestrates all components of the RAG-Vision system:
        1. **Input validation**
           Ensures all required fields are included in the request payload.

        2. **Support retrieval with CLIP**
           Uses the SupportIndex to extract and compare visual patches from
           the support dataset.

        3. **Multimodal reasoning using Qwen2-VL-2B-Instruct**
           Builds the multimodal input (images + system prompt + user prompt)
           and performs generative reasoning.

        4. **Postprocessing**
           Extracts structured fields (flag, explanation) from the raw VLM output.

        5. **Response formatting**
           Returns a `RagVisionOutput` schema with structured prediction results.

    Args:
        payload (RagVisionInput):
            The request payload containing:
                - Base64-encoded image
                - System and user prompts
                - Retrieval configuration
                - Generation parameters
                - Image preprocessing settings

    Returns:
        JSONResponse:
            A JSON-compatible response shaped according to the `RagVisionOutput` model.
            Contains either:
                - A successful structured inference result, or
                - An error structure if execution failed.

    Notes:
        - Each request is assigned a unique `imageId` for tracking.
        - Errors are logged and returned with full context.
        - All execution times are measured and logged in milliseconds.
    """
    start_time = time.time()

    imageId = str(uuid.uuid4())

    required_fields = {
        "encodedImage": payload.encodedImage,
        "flagKey": payload.flagKey,
        "systemPrompt": payload.systemPrompt,
        "userPrompt": payload.userPrompt,
    }

    missing = [k for k, v in required_fields.items() if v is None]
    if missing:
        elapsed = int((time.time() - start_time) * 1000)
        LoggerManager.log_formatter(
            f"Missing required fields: {', '.join(missing)}",
            imageId, 4000, level="ERROR")
        
        return JSONResponse(
            RagVisionOutput.fails(
                imageId=imageId,
                flagKey=payload.flagKey,
                elapsedTimeMs=elapsed,
                err=f"Missing fields: {missing}",
            ),
            status_code=4000
        )

    try:
        LoggerManager.log_formatter(
            "Executing RAG-Vision service...",
            imageId, 2000, level="INFO"
        )


        result = rag_service.analyze(
            b64_image=payload.encodedImage,
            flagKey=payload.flagKey,
            system_prompt=payload.systemPrompt,
            user_prompt=payload.userPrompt,
            imageId=imageId,
            k_retrieval=payload.kRetrieval,
            max_patches_per_class=payload.maxPatchesPerClass,
            max_new_tokens=payload.maxNewTokens,
            input_resolution=payload.inputResolution,
            support_res=payload.supportRes,
            support_patch_size=payload.supportPatchSize,
            support_clip_local_dir=None,
            temperature=payload.temperature,
            top_p=payload.topP,
        )

    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)

        LoggerManager.log_formatter(
            f"RAG-Vision ERROR: {str(e)}",
            imageId, 5000, level="ERROR")

        return JSONResponse(
            RagVisionOutput.fails(
                imageId=imageId,
                flagKey=payload.flagKey,
                elapsedTimeMs=elapsed,
                err=str(e),
            ),
            status_code=2000
        )

    elapsed = int((time.time() - start_time) * 1000)

    LoggerManager.log_formatter(
        f"RAG-Vision inference completed in {elapsed} ms. Result: {result}",
        imageId, 2000, level="INFO")

    return JSONResponse(
        RagVisionOutput.success_output(
            imageId=imageId,
            flagKey=payload.flagKey,
            elapsedTimeMs=elapsed,
            flag=result.get("flag"),
            explanation=result.get("explanation"),
            classScores=result.get("class_scores"),
            rawResponse=result.get("raw_response"),
        )
    )