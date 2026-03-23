"""Global rate limiter for external API calls (yfinance, akshare, etc.).

All yfinance calls should go through `yf_limiter` to avoid hammering the API
when 13 agents run in parallel.

Usage:
    from src.common.rate_limiter import yf_limiter

    with yf_limiter:
        data = yf.download(...)
"""

import time
import threading


class RateLimiter:
    """Token-bucket style rate limiter with concurrency cap.

    - max_concurrent: max simultaneous calls (semaphore)
    - min_interval:   minimum seconds between any two calls (global throttle)
    """

    def __init__(self, max_concurrent: int = 3, min_interval: float = 1.0):
        self._sem = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = min_interval

    def __enter__(self):
        self._sem.acquire()
        with self._lock:
            now = time.time()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
        return self

    def __exit__(self, *args):
        self._sem.release()


# Shared limiter for all yfinance calls:
# - max 3 concurrent requests
# - at least 1 second between any two calls globally
yf_limiter = RateLimiter(max_concurrent=3, min_interval=1.0)
