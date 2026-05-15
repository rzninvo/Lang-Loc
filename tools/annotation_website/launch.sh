#!/usr/bin/env bash
# Launch the LangLoc annotation site behind a public Cloudflare Tunnel
# (default) or ngrok. Default is cloudflared because ngrok's free tier
# hits a monthly bandwidth cap quickly once we start serving 100-MB
# meshes.
#
# Reads .env for LANGLOC_COOKIE_SECRET / LANGLOC_ADMIN_TOKEN.
# Boots uvicorn on 0.0.0.0:$PORT (default 8000), then opens the tunnel
# and prints the public URL.
#
# Usage:
#   ./launch.sh [--port 8000] [--tunnel cloudflared|ngrok|none]
#
# Stop with Ctrl-C; both processes are torn down.

set -euo pipefail
cd "$(dirname "$0")"

PORT=8000
TUNNEL=cloudflared
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-ngrok|--no-tunnel) TUNNEL=none; shift ;;
    --tunnel) TUNNEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# //'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -f .env ]]; then
  set -a; source .env; set +a
else
  echo "[WARN] no .env file; using insecure default secrets" >&2
fi

PY="${LANGLOC_PYTHON:-$HOME/miniconda3/envs/langloc/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python)"

# Sanity: at least one per-dataset pool prepared?
if ! ls data/scenes_*.json >/dev/null 2>&1; then
  echo "[ERROR] no data/scenes_<dataset>.json files; run scripts/prepare_keyframes.py + compute_difficulty.py for each dataset first." >&2
  exit 1
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── boot uvicorn ──────────────────────────────────────────────────────────
echo "[launch] starting uvicorn on 0.0.0.0:$PORT"
"$PY" -m uvicorn server.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --forwarded-allow-ips '*' \
    --log-level info \
    > "$LOG_DIR/uvicorn.log" 2>&1 &
UVICORN_PID=$!

TUN_PID=""
cleanup() {
  echo
  echo "[launch] stopping…"
  kill "$UVICORN_PID" 2>/dev/null || true
  if [[ -n "$TUN_PID" ]]; then
    kill "$TUN_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  echo "[launch] stopped."
}
trap cleanup INT TERM EXIT

# wait for uvicorn to bind
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if ss -tln | grep -q ":$PORT"; then break; fi
  sleep 0.5
done
if ! ss -tln | grep -q ":$PORT"; then
  echo "[ERROR] uvicorn failed to bind. tail of log:"
  tail -25 "$LOG_DIR/uvicorn.log" >&2
  exit 1
fi
echo "[launch] uvicorn ready (PID $UVICORN_PID); local http://127.0.0.1:$PORT"

# ── tunnel ────────────────────────────────────────────────────────────────
PUB_URL=""

case "$TUNNEL" in
  cloudflared)
    # cloudflared lives in either /usr/local/bin or ~/.local/bin
    CFD="$(command -v cloudflared 2>/dev/null || echo "$HOME/.local/bin/cloudflared")"
    if [[ ! -x "$CFD" ]]; then
      echo "[ERROR] cloudflared not on PATH and not at $HOME/.local/bin/cloudflared." >&2
      echo "        Install: curl -sSL -o ~/.local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/.local/bin/cloudflared" >&2
      echo "        Or run with --tunnel ngrok" >&2
      exit 1
    fi
    echo "[launch] opening Cloudflare Tunnel (no account, ephemeral URL)"
    "$CFD" tunnel --url "http://localhost:$PORT" --no-autoupdate \
        > "$LOG_DIR/cloudflared.log" 2>&1 &
    TUN_PID=$!
    # poll the log for the trycloudflare URL it prints on startup
    for _ in $(seq 1 30); do
      PUB_URL=$(grep -oE "https://[a-z0-9.-]+\.trycloudflare\.com" "$LOG_DIR/cloudflared.log" 2>/dev/null | head -1)
      [[ -n "$PUB_URL" ]] && break
      sleep 0.6
    done
    if [[ -z "$PUB_URL" ]]; then
      echo "[ERROR] cloudflared did not produce a public URL after 18s. tail of log:" >&2
      tail -30 "$LOG_DIR/cloudflared.log" >&2
      exit 1
    fi
    ;;

  ngrok)
    if ! command -v ngrok >/dev/null; then
      echo "[ERROR] ngrok not on PATH; install or use --tunnel cloudflared" >&2
      exit 1
    fi
    echo "[launch] opening ngrok tunnel"
    ngrok http "$PORT" --log=stdout > "$LOG_DIR/ngrok.log" 2>&1 &
    TUN_PID=$!
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      PUB_URL=$(curl -sS http://127.0.0.1:4040/api/tunnels 2>/dev/null \
                | "$PY" -c "import sys, json
try:
  d = json.load(sys.stdin)
  for t in d.get('tunnels', []):
    if t.get('proto') == 'https':
      print(t['public_url']); break
except Exception:
  pass" 2>/dev/null) || true
      [[ -n "$PUB_URL" ]] && break
      sleep 0.6
    done
    if [[ -z "$PUB_URL" ]]; then
      echo "[ERROR] ngrok did not produce a public URL after 9s." >&2
      tail -25 "$LOG_DIR/ngrok.log" >&2
      exit 1
    fi
    ;;

  none) ;;
  *) echo "[ERROR] unknown --tunnel value: $TUNNEL" >&2; exit 2 ;;
esac

if [[ -n "$PUB_URL" ]]; then
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo " LangLoc annotation site live at:"
  echo "   $PUB_URL"
  echo "──────────────────────────────────────────────────────────────"
  echo " admin (gated on LANGLOC_ADMIN_TOKEN):"
  echo "   $PUB_URL/admin/coverage"
  echo "──────────────────────────────────────────────────────────────"
  echo
  echo "Logs:    $LOG_DIR/{uvicorn,cloudflared,ngrok}.log"
  echo "Stop:    Ctrl-C"
fi

# wait on uvicorn so trap handles teardown
wait "$UVICORN_PID"
