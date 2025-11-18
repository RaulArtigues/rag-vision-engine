import os

SPACE_ID = os.getenv("SPACE_ID")

if SPACE_ID:
    SPACE_NAME = SPACE_ID.replace("/", "-")
    BASE_URL = f"https://{SPACE_NAME}.hf.space"
else:
    BASE_URL = "http://localhost:7860"

# Common API prefix used by all backend endpoints.
API_PREFIX = f"{BASE_URL}/api"

# Endpoint used to upload support images for the RAG-Vision pipeline.
ENDPOINT_UPLOAD = f"{API_PREFIX}/support/upload/image"

# Endpoint used to invoke the RAG-Vision inference API.
ENDPOINT_RAG = f"{API_PREFIX}/ragvision/invocations"