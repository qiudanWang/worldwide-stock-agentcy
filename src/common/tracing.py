"""TraceRoot initialization and observe decorator export.

Sets up tracing against the staging endpoint. Import observe from here
so every module gets a consistent, already-initialized decorator.
"""

import os
from pathlib import Path

# Load .env from project root if present
_env_path = Path(__file__).parents[2] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("TRACEROOT_HOST_URL", "https://staging.traceroot.ai/")

try:
    import traceroot
    from traceroot import Integration
    traceroot.initialize(
        integrations=[Integration.OPENAI, Integration.OPENAI_AGENTS],
        git_repo="https://github.com/qiudanWang/worldwide-stock-agentcy",
    )
    from traceroot import observe
except ImportError:
    def observe(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def _decorator(fn):
            return fn
        return _decorator
