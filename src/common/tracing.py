"""TraceRoot initialization and observe decorator export.

Sets up tracing against the staging endpoint. Import observe from here
so every module gets a consistent, already-initialized decorator.
"""

import os

os.environ.setdefault("TRACEROOT_API_KEY", "tr-5de4fc1a-4ca6-457b-9edf-0ce9ea40d839")
os.environ.setdefault("TRACEROOT_HOST_URL", "https://staging.traceroot.ai/")

try:
    import traceroot
    from traceroot import Integration
    traceroot.initialize(integrations=[Integration.OPENAI])
    from traceroot import observe
except ImportError:
    def observe(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def _decorator(fn):
            return fn
        return _decorator
