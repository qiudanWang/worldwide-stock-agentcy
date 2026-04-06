#!/bin/bash
# Global Tech Market — Multi-Agent Pipeline Runner
# Usage:
#   ./run.sh                                    Full pipeline (all 40 agents, cached)
#   ./run.sh --force                            Full pipeline, bypass all caches
#   ./run.sh --market CN                        Run CN market agents only
#   ./run.sh --market CN --force                Run CN agents, bypass cache
#   ./run.sh --agent CN_data                    Run a single agent
#   ./run.sh --agent CN_data --force            Run single agent, bypass cache
#   ./run.sh --type data                        Run all data agents
#   ./run.sh --type data --force                Run all data agents, bypass cache
#   ./run.sh --type news                        Run all news agents
#   ./run.sh --type signal                      Run all signal agents
#   ./run.sh --type global                      Run global agent only
#   ./run.sh --legacy                           Run legacy pipeline (backward compat)

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/data/logs"
DATE=$(date +%Y%m%d)

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/pipeline_$DATE.log"
}

# Parse arguments
AGENT=""
MARKET=""
AGENT_TYPE=""
LEGACY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --market)
            MARKET="$2"
            shift 2
            ;;
        --type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --legacy)
            LEGACY=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run.sh [--agent NAME] [--market CODE] [--type TYPE] [--force] [--legacy]"
            exit 1
            ;;
    esac
done

if [ "$LEGACY" = true ]; then
    log "=== Legacy Pipeline Start ==="
    "$VENV" "$PROJECT_DIR/scripts/run_daily_pipeline.py" >> "$LOG_DIR/pipeline_$DATE.log" 2>&1
    log "=== Legacy Pipeline End ==="
    exit $?
fi

log "=== Multi-Agent Pipeline Start ==="

"$VENV" -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')

from src.agents.orchestrator import Orchestrator

orch = Orchestrator()
orch.load_all_agents()

force = '$FORCE' == 'true'
if force:
    orch.set_force(True)
    print('Force mode: all caches bypassed')

agent = '$AGENT'
market = '$MARKET'
agent_type = '$AGENT_TYPE'

if agent:
    print(f'Running single agent: {agent}')
    orch.run_agent(agent)
elif market:
    print(f'Running market: {market}')
    orch.run_market(market)
elif agent_type:
    print(f'Running agent type: {agent_type}')
    orch.run_agent_type(agent_type)
else:
    print('Running full pipeline...')
    orch.run_all()
" 2>&1 | tee -a "$LOG_DIR/pipeline_$DATE.log"

log "=== Multi-Agent Pipeline End ==="
