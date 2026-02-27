from fastapi.responses import JSONResponse
from fastapi import APIRouter, status

healthcheck_router = APIRouter()
@healthcheck_router.get("/ragvision/healthcheck", status_code=status.HTTP_201_CREATED, summary="Check service health",            
    responses={status.HTTP_201_CREATED: {"description": "Service is healthy","content": {"application/json": {"example": {"status": "ok"}}}}})
async def healthcheck():
    """
    Healthcheck endpoint for infrastructure monitoring.

    This endpoint verifies the API is alive and responsive.
    Used by ECS to determine container health.

    Returns:
        dict: A JSON with status "ok" and HTTP 201 if successful.
    """
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_201_CREATED)