import concurrent.futures

_settings_cache = None


def _get_timeouts():
    global _settings_cache
    if _settings_cache is None:
        try:
            from src.common.config import get_settings
            _settings_cache = get_settings().get("timeouts", {})
        except Exception:
            _settings_cache = {}
    return _settings_cache


def default_timeout():
    return _get_timeouts().get("default", 15)


def bulk_timeout():
    return _get_timeouts().get("bulk", 30)


def llm_timeout():
    return _get_timeouts().get("llm", 60)


def call_with_timeout(fn, *args, timeout=None, **kwargs):
    """Run fn(*args, **kwargs) in a thread; raise TimeoutError if it exceeds timeout seconds.

    timeout defaults to the configured timeouts.default value (settings.yaml).
    """
    if timeout is None:
        timeout = default_timeout()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args, **kwargs).result(timeout=timeout)
