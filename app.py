from fastapi.middleware.cors import CORSMiddleware
from backend.api.app import app as fastapi_backend
from gradio.routes import mount_gradio_app
from frontend.ui.layout import build_ui
from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api", fastapi_backend)

ui = build_ui()
mount_gradio_app(app, ui, path="/")