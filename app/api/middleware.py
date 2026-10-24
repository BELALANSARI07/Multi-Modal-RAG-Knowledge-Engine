"""Security middleware and dependency injection for API authentication and rate limiting."""

from collections import defaultdict
import logging
import time
from typing import Dict, List, Optional
import uuid

from fastapi import HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

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


class InMemoryRateLimiter:
    """Sliding-window rate limiter per client IP address."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = True) -> None:
        self.default_rpm = requests_per_minute
        self.enabled = enabled
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str, rpm: Optional[int] = None) -> bool:
        if not self.enabled:
            return True

        max_reqs = rpm or self.default_rpm
        now = time.time()
        window_start = now - 60.0

        # Purge timestamps outside sliding 60-second window
        timestamps = [t for t in self._requests[client_ip] if t > window_start]
        self._requests[client_ip] = timestamps

        if len(timestamps) >= max_reqs:
            return False

        self._requests[client_ip].append(now)
        return True


rate_limiter = InMemoryRateLimiter(
    requests_per_minute=settings.rate_limit_requests_per_minute,
    enabled=settings.rate_limit_enabled,
)


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing rate limits per client IP."""
    app_settings = getattr(request.app.state, "settings", settings)
    if not app_settings.rate_limit_enabled:
        return

    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip, rpm=app_settings.rate_limit_requests_per_minute):
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
) -> Optional[str]:
    """FastAPI dependency enforcing API Key authentication if configured in settings."""
    app_settings = getattr(request.app.state, "settings", settings)
    if not app_settings.api_key:
        return None  # Authentication disabled if API_KEY is not set

    if not api_key_from_header or api_key_from_header != app_settings.api_key:
        req_id = getattr(request.state, "request_id", "unknown")
        logger.warning("Unauthorized request attempt [Request: %s]", req_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key in request header",
            headers={"WWW-Authenticate": app_settings.api_key_header_name, "X-Request-ID": req_id},
        )
    return api_key_from_header
