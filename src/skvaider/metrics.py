"""Prometheus metrics for the gateway."""

from prometheus_client import Counter, Gauge, Histogram

# Request metrics
gateway_requests_total = Counter(
    "skvaider_gateway_requests_total",
    "Total number of gateway requests",
    ["endpoint", "status", "model", "streaming"],
)

gateway_request_duration_seconds = Histogram(
    "skvaider_gateway_request_duration_seconds",
    "Duration of gateway requests in seconds",
    ["endpoint", "model", "streaming"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Backend metrics
gateway_backend_requests_total = Counter(
    "skvaider_gateway_backend_requests_total",
    "Total number of requests to backends",
    ["backend", "endpoint", "model", "status", "streaming"],
)

gateway_backend_retry_total = Counter(
    "skvaider_gateway_backend_retry_total",
    "Total number of backend request retries",
    ["backend", "endpoint", "model", "reason", "streaming"],
)

# Active request tracking
gateway_active_requests = Gauge(
    "skvaider_gateway_active_requests",
    "Number of currently active requests",
    ["endpoint", "model", "streaming"],
)
