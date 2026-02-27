from backend.api.routers.api_v1.healthcheck_ragvision import healthcheck_router
from backend.api.routers.api_v1 import support_upload_router
from backend.api.routers.api_v1 import rag_vision_router
from fastapi import APIRouter

# Main router that groups all sub-routes related to the RAG-Vision API
ragvision_router = APIRouter()

# Include the router responsible for uploading support images
ragvision_router.include_router(support_upload_router.router)

# Include the main RAG-Vision inference endpoint router
ragvision_router.include_router(rag_vision_router.router)

# Include the healthcheck endpoint router for monitoring service status
ragvision_router.include_router(healthcheck_router)