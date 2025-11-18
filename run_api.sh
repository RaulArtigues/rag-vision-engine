#!/bin/bash

# ====================================================
#   RAG-VISION ENGINE — LOCAL API LAUNCH SCRIPT
# ====================================================

# --- COLORS ---
GREEN="\033[0;32m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${CYAN}"
echo "==================================================="
echo "        🚀 Starting RAG-VISION ENGINE API          "
echo "==================================================="
echo -e "${RESET}"

# --- CHECK FOR CONDA ---
if ! command -v conda &> /dev/null
then
    echo -e "${RED}[ERROR] Conda is not installed or not in PATH.${RESET}"
    exit 1
fi

# --- ACTIVATE ENVIRONMENT ---
echo -e "${GREEN}[INFO] Activating conda environment: rag-vision-engine...${RESET}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rag-vision-engine

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to activate conda environment 'rag-vision-engine'.${RESET}"
    exit 1
fi

# --- COMPUTE PROJECT PATHS ---
PROJECT_ROOT="$(pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
APP_PATH="${BACKEND_DIR}/api/app.py"

echo -e "${GREEN}[INFO] Current directory: ${PROJECT_ROOT}${RESET}"

# --- VALIDATE APP PATH ---
if [ ! -f "$APP_PATH" ]; then
    echo -e "${RED}[ERROR] app.py not found at: ${APP_PATH}${RESET}"
    echo -e "${RED}Make sure this script is executed from the project root directory.${RESET}"
    exit 1
fi

# --- START API ---
echo -e "${GREEN}[INFO] Launching FastAPI server (development mode)...${RESET}"
echo -e "${GREEN}[INFO] API Docs available at: http://localhost:8000/docs${RESET}"

cd "$BACKEND_DIR"

uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level debug \
    --reload