GREEN="\033[0;32m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${CYAN}"
echo "======================================================"
echo "        🚀 Starting Unified RAG-VISION ENGINE         "
echo "======================================================"
echo -e "${RESET}"

PROJECT_ROOT="$(pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/backend:${PROJECT_ROOT}/frontend"

echo -e "${GREEN}[INFO] Checking conda...${RESET}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[ERROR] Conda is not installed.${RESET}"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rag-vision-engine || {
    echo -e "${RED}[ERROR] Could not activate conda env 'rag-vision-engine'.${RESET}"
    exit 1
}

REQ_FILE="${PROJECT_ROOT}/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo -e "${GREEN}[INFO] Installing Python dependencies...${RESET}"
    pip install -r "$REQ_FILE"
else
    echo -e "${RED}[WARNING] No requirements.txt found.${RESET}"
fi

echo -e "${GREEN}[INFO] Starting Unified FastAPI + Gradio app...${RESET}"

uvicorn app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --log-level debug \
    --reload &

APP_PID=$!

echo -e "${GREEN}[INFO] Unified App running with PID: ${APP_PID}${RESET}"

echo -e "${CYAN}"
echo "======================================================"
echo "  🎉 RAG-VISION ENGINE IS RUNNING!"
echo "  API Docs:   http://localhost:7860/api/docs"
echo "  Frontend:   http://localhost:7860"
echo "======================================================"
echo -e "${RESET}"

wait