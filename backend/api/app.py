from backend.api.routers.events import logging as logging_router
from backend.api.schema.ragvision_outputs import RagVisionOutput
from backend.api.routers.events.logging import LoggerManager
from starlette.middleware.base import BaseHTTPMiddleware
from backend.api.config.api_settings import APISettings
from backend.api.routers.api import ragvision_router
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
import logging

app = FastAPI()

class DebugRequestMiddleware(BaseHTTPMiddleware):
    """
    Middleware used for debugging raw incoming HTTP requests.

    Logs:
        - HTTP method
        - Request URL
        - Raw request body (byte length)

    It clones the body into `request._body` because FastAPI can only
    read the request body once; without cloning, downstream parsers would fail.
    """
    async def dispatch(self, request, call_next):
        print("\n========== RAW HTTP REQUEST DEBUG ==========")
        print("→ method:", request.method)
        print("→ url:", request.url)

        body = await request.body()
        request._body = body

        print("→ RAW BODY:", len(body), "bytes")

        response = await call_next(request)
        return response

app.add_middleware(DebugRequestMiddleware)

class CustomUvicornHandler(logging.Handler):
    """
    A custom logging handler that redirects Uvicorn log messages through the
    unified LoggerManager formatting system. Ensures consistent logging behavior
    across the entire API stack.
    """
    def emit(self, record):
        log_message = record.getMessage()
        try:
            LoggerManager.log_formatter(
                f'{log_message}', f'', '', level='INFO')
        except Exception as e:
            logging.error(f"Failed to log message: {e}")

def custom_openapi_schema():
    """
    Generate a customized OpenAPI schema for the RAG-Vision API.

    Responsibilities:
        - Use APISettings for title/version/description.
        - Remove Pydantic validation error schemas from OpenAPI output.
        - Rewrite internal Pydantic $defs references to valid OpenAPI references.
        - Insert custom "example" for failure responses using RagVisionOutput.fails().
        - Override HTTP 422 validation error schema with the custom fail model.

    Returns:
        dict: The OpenAPI schema with custom modifications.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=APISettings.app_name,
        version=APISettings.api_current_version,
        description=APISettings.app_description,
        routes=app.routes,
    )

    openapi_schema["components"]["schemas"].pop("ValidationError", None)
    openapi_schema["components"]["schemas"].pop("HTTPValidationError", None)

    for schema_name, schema in openapi_schema["components"]["schemas"].items():
        schema_str = str(schema)
        if "$defs" in schema_str:
            schema_str = schema_str.replace("#/$defs/", "#/components/schemas/")
        openapi_schema["components"]["schemas"][schema_name] = eval(schema_str)

    imageId = ''
    flagKey = ''
    elapsedTimeMs = 0
    err = ''
    fails_instance_dict = RagVisionOutput.fails(imageId, flagKey, elapsedTimeMs, err)
    fails_schema = RagVisionOutput.schema(ref_template="#/components/schemas/{model}")
    fails_schema['example'] = fails_instance_dict
    
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if "422" in openapi_schema["paths"][path][method]["responses"]:
                openapi_schema["paths"][path][method]["responses"]["422"] = {
                    "description": "Validation Error",
                    "content": {
                        "application/json": {"schema": fails_schema}
                    },
                }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi_schema

app.include_router(ragvision_router, tags=["RagVision"])
app.include_router(logging_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, log_level="debug")