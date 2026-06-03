"""TraceRoot initialization and observe decorator export.

Sets up tracing against the staging endpoint. Import observe from here
so every module gets a consistent, already-initialized decorator.
"""

import os

os.environ.setdefault("TRACEROOT_API_KEY", "tr-795818bf-8dca-4039-8125-0eeea8d71e21")
os.environ.setdefault("TRACEROOT_HOST_URL", "https://staging.traceroot.ai/")

try:
    import traceroot
    from traceroot import Integration
    traceroot.initialize(
        integrations=[Integration.OPENAI],
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
