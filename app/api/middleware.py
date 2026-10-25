
"""Security middleware and dependency injection for API authentication and rate limiting."""

import logging
import threading
import time
import uuid
from collections import defaultdict

from fastapi import HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

api_key_header = APIKeyHeader(name=settings.api_key_header_name, auto_error=False)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a unique correlation ID to every request for end-to-end trace tracking."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RateLimitBackend:
    """Abstract interface for rate limiting backends."""

    def is_allowed(self, client_ip: str, rpm: int) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MemoryRateLimitBackend(RateLimitBackend):
    """Process-local sliding-window rate limiter per client IP address.

    Note: When deploying multiple Uvicorn worker processes without a shared
    cache, rate limit counters are maintained independently per worker.
    For shared multi-worker or multi-replica rate limiting, configure Redis.
    """

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str, rpm: int) -> bool:
        now = time.time()
        window_start = now - 60.0

        with self._lock:
            # Purge timestamps outside sliding 60-second window
            timestamps = [t for t in self._requests[client_ip] if t > window_start]
            if len(timestamps) >= rpm:
                self._requests[client_ip] = timestamps
                return False

            timestamps.append(now)
            self._requests[client_ip] = timestamps
            return True

    def reset(self) -> None:
        """Clear all in-memory counters (useful for unit test isolation)."""
        with self._lock:
            self._requests.clear()


class RedisRateLimitBackend(RateLimitBackend):
    """Shared Redis-backed rate limiter for multi-worker and multi-container deployments."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client = None
        self._fallback_memory = MemoryRateLimitBackend()

        try:
            import redis

            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info("Successfully connected to Redis rate-limit backend at %s", redis_url)
        except Exception as e:
            logger.warning(
                "Failed to initialize Redis rate-limit backend (%s). Falling back to process-local in-memory limiter.",
                e,
            )
            self._client = None

    def is_allowed(self, client_ip: str, rpm: int) -> bool:
        if self._client is None:
            return self._fallback_memory.is_allowed(client_ip, rpm)

        key = f"ratelimit:{client_ip}:{int(time.time() // 60)}"
        try:
            current = self._client.incr(key)
            if current == 1:
                self._client.expire(key, 65)
            return current <= rpm
        except Exception as e:
            logger.warning("Redis rate limit check failed (%s). Falling back to memory backend.", e)
            return self._fallback_memory.is_allowed(client_ip, rpm)

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass


class RateLimiter:
    """High-level rate limiting orchestrator with pluggable backends."""

    def __init__(
        self,
        backend: RateLimitBackend | None = None,
        enabled: bool = True,
        default_rpm: int = 60,
    ) -> None:
        self.backend = backend or MemoryRateLimitBackend()
        self.enabled = enabled
        self.default_rpm = default_rpm

    def is_allowed(self, client_ip: str, rpm: int | None = None) -> bool:
        if not self.enabled:
            return True
        max_reqs = rpm or self.default_rpm
        return self.backend.is_allowed(client_ip, max_reqs)

    def close(self) -> None:
        self.backend.close()


def create_rate_limiter(cfg: Settings) -> RateLimiter:
    """Factory creating configured RateLimiter instance."""
    backend: RateLimitBackend
    if cfg.rate_limit_backend == "redis" and cfg.redis_url:
        backend = RedisRateLimitBackend(cfg.redis_url)
    else:
        backend = MemoryRateLimitBackend()

    return RateLimiter(
        backend=backend,
        enabled=cfg.rate_limit_enabled,
        default_rpm=cfg.rate_limit_requests_per_minute,
    )


# Default global rate limiter instance
rate_limiter = create_rate_limiter(settings)
# Keep InMemoryRateLimiter as alias for backward compatibility with existing tests
InMemoryRateLimiter = RateLimiter


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing rate limits per client IP."""
    app_settings = getattr(request.app.state, "settings", settings)
    if not app_settings.rate_limit_enabled:
        return

    limiter = getattr(request.app.state, "rate_limiter", rate_limiter)
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.is_allowed(client_ip, rpm=app_settings.rate_limit_requests_per_minute):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.warning("Rate limit exceeded for IP %s [Request: %s]", client_ip, req_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: maximum {app_settings.rate_limit_requests_per_minute} requests per minute.",
            headers={"Retry-After": "60", "X-Request-ID": req_id},
        )


async def verify_api_key(
    request: Request,
    api_key_from_header: str = Security(api_key_header),
) -> str | None:
    """FastAPI dependency enforcing API Key authentication if configured in settings."""
    app_settings = getattr(request.app.state, "settings", settings)
    configured_key = app_settings.api_key_str
    if not configured_key:
        return None  # Authentication disabled if API_KEY is not set

    if not api_key_from_header or api_key_from_header != configured_key:
        req_id = getattr(request.state, "request_id", "unknown")
        logger.warning("Unauthorized request attempt [Request: %s]", req_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key in request header",
            headers={"WWW-Authenticate": app_settings.api_key_header_name, "X-Request-ID": req_id},
        )
    return api_key_from_header
