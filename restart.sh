#!/bin/bash
# Global Tech Monitor — Daily Pipeline Runner
# Usage: ./restart.sh [module]
#   ./restart.sh          Run full daily pipeline
#   ./restart.sh universe Rebuild stock pool
#   ./restart.sh market   Fetch market data + volume alerts only
#   ./restart.sh peers    Rebuild peer mapping
#   ./restart.sh news     Fetch news only
#   ./restart.sh capital  Fetch capital flow only
#   ./restart.sh finance  Fetch financials only

PROJECT_DIR="/Users/cathewan/Desktop/stocks"
VENV="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/data/logs"
DATE=$(date +%Y%m%d)

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/pipeline_$DATE.log"
}

run_script() {
    local name="$1"
    local script="$2"
    log "START: $name"
    "$VENV" "$PROJECT_DIR/scripts/$script" >> "$LOG_DIR/pipeline_$DATE.log" 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "DONE:  $name"
    else
        log "FAIL:  $name (exit code $exit_code)"
    fi
    return $exit_code
}

MODULE="${1:-all}"

log "=== Pipeline start (module=$MODULE) ==="

case "$MODULE" in
    universe)
        run_script "Universe Builder" "build_universe.py"
        ;;
    market)
        run_script "Market Data" "fetch_market_data.py"
        ;;
    peers)
        run_script "Peer Mapping" "build_peers.py"
        ;;
    news)
        run_script "News" "fetch_news.py"
        ;;
    capital)
        run_script "Capital Flow" "fetch_capital_flow.py"
        ;;
    finance)
        run_script "Financials" "fetch_financials.py"
        ;;
    all)
        run_script "Daily Pipeline" "run_daily_pipeline.py"
        ;;
    *)
        log "Unknown module: $MODULE"
        echo "Usage: ./restart.sh [all|universe|market|peers|news|capital|finance]"
        exit 1
        ;;
esac

log "=== Pipeline end ==="
