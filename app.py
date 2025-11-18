# app.py — Unified FastAPI + Gradio for Hugging Face Spaces

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

# backend FastAPI app
from backend.api.app import app as fastapi_backend

# frontend Gradio UI
from frontend.ui.layout import build_ui

# =====================================================
# MAIN ASGI APPLICATION (ONE SINGLE SERVER)
# =====================================================
app = FastAPI()

# CORS (opcional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# BACKEND mounted at /api/*
# =====================================================
app.mount("/api", fastapi_backend)

# =====================================================
# FRONTEND (Gradio) mounted at /
# =====================================================
ui = build_ui()
mount_gradio_app(app, ui, path="/")

# IMPORTANT:
# Do NOT add uvicorn.run(...) here.
# Hugging Face will run uvicorn via Dockerfile.