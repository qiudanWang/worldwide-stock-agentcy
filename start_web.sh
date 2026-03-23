#!/bin/bash
# Global Tech Market — Start web server

PROJECT_DIR="/Users/cathewan/Desktop/stocks/worldwide-stock-agentcy"
VENV="$PROJECT_DIR/venv/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "Starting Global Tech Market on port 5000..."
cd "$PROJECT_DIR"
"$VENV" -m flask --app web.app run --host=0.0.0.0 --port=5000 &
FLASK_PID=$!
log "Flask started (PID $FLASK_PID)"
log "  Main app: http://localhost:5000"

cleanup() {
    log "Shutting down..."
    [ -n "$FLASK_PID" ] && kill "$FLASK_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

wait $FLASK_PID
