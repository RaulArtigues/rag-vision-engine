#!/usr/bin/env bash
set -euo pipefail

# Simple tunnel launcher for exposing localhost (e.g., Gradio on 7860) to the internet.
# Usage examples:
#   ./scripts/start_tunnel.sh ngrok 7860
#   ./scripts/start_tunnel.sh cloudflare 7860
#
# Providers:
#   ngrok       -> requires ngrok installed; will auto-use $NGROK_AUTHTOKEN if set
#   cloudflare  -> requires cloudflared installed; uses ephemeral trycloudflare.com URL

provider=${1:-ngrok}
port=${2:-7860}

err() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

case "$provider" in
  ngrok)
    command -v ngrok >/dev/null 2>&1 || err "ngrok no está instalado. Instala con: brew install ngrok/ngrok/ngrok"

    if [[ -n "${NGROK_AUTHTOKEN-}" ]]; then
      info "Configurando authtoken de ngrok..."
      ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null
    fi

    info "Levantando túnel ngrok -> http://localhost:${port}"
    exec ngrok http "${port}"
    ;;

  cloudflare|cloudflared)
    command -v cloudflared >/dev/null 2>&1 || err "cloudflared no está instalado. Instala con: brew install cloudflared"

    info "Levantando túnel Cloudflare -> http://localhost:${port}"
    exec cloudflared tunnel --url "http://localhost:${port}"
    ;;

  *)
    err "Proveedor '${provider}' no soportado. Usa 'ngrok' o 'cloudflare'."
    ;;
esac
