#!/bin/bash

# ====================================================
#     🚀 RAG-VISION ENGINE — FULL LAUNCH SCRIPT
# ====================================================

GREEN="\033[0;32m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${CYAN}"
echo "======================================================"
echo "       🚀 Starting FULL RAG-VISION ENGINE STACK       "
echo "======================================================"
echo -e "${RESET}"

PROJECT_ROOT="$(pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"

### 1 — CHECK CONDA ###
echo -e "${GREEN}[INFO] Checking conda...${RESET}"
if ! command -v conda &> /dev/null
then
    echo -e "${RED}[ERROR] Conda is not installed or not found.${RESET}"
    exit 1
fi

### 2 — ACTIVATE ENVIRONMENT ###
echo -e "${GREEN}[INFO] Activating conda environment: rag-vision-engine...${RESET}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rag-vision-engine

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to activate conda env 'rag-vision-engine'.${RESET}"
    exit 1
fi

### 3 — INSTALL REQUIREMENTS IF NEEDED ###
REQ_FILE="${PROJECT_ROOT}/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    echo -e "${GREEN}[INFO] Installing Python dependencies from requirements.txt...${RESET}"
    pip install -r "$REQ_FILE"
else
    echo -e "${RED}[WARNING] No requirements.txt found. Skipping dependency installation.${RESET}"
fi

### 4 — VALIDATE BACKEND AND FRONTEND ###
APP_BACKEND="${BACKEND_DIR}/api/app.py"
APP_FRONTEND="${FRONTEND_DIR}/app.py"

if [ ! -f "$APP_BACKEND" ]; then
    echo -e "${RED}[ERROR] Backend app.py not found at: ${APP_BACKEND}${RESET}"
    exit 1
fi

if [ ! -f "$APP_FRONTEND" ]; then
    echo -e "${RED}[ERROR] Frontend Gradio app.py not found at: ${APP_FRONTEND}${RESET}"
    exit 1
fi

echo -e "${GREEN}[INFO] Both backend and frontend found.${RESET}"

### 5 — START BACKEND ###
echo -e "${GREEN}[INFO] Starting BACKEND (FastAPI) on port 8000...${RESET}"
cd "$BACKEND_DIR"

uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level debug \
    --reload &

BACKEND_PID=$!
echo -e "${GREEN}[INFO] Backend running with PID: ${BACKEND_PID}${RESET}"

### 6 — START FRONTEND ###
echo -e "${GREEN}[INFO] Starting FRONTEND (Gradio UI) on port 7860...${RESET}"
cd "$FRONTEND_DIR"

python app.py --server_name 0.0.0.0 --server_port 7860 &

FRONTEND_PID=$!
echo -e "${GREEN}[INFO] Frontend running with PID: ${FRONTEND_PID}${RESET}"

### 7 — WAIT FOREVER (KEEP SCRIPT ALIVE) ###
echo -e "${CYAN}"
echo "======================================================"
echo "  🎉 RAG-VISION ENGINE IS FULLY RUNNING!"
echo "  Backend:  http://localhost:8000/docs"
echo "  Frontend: http://localhost:7860"
echo "======================================================"
echo -e "${RESET}"

wait