from prometheus_client import Counter, Gauge, Histogram

# Request metrics
inference_requests_total = Counter(
    "skvaider_inference_requests_total",
    "Total number of inference requests",
    ["model", "endpoint", "status"],
)

inference_request_duration_seconds = Histogram(
    "skvaider_inference_request_duration_seconds",
    "Duration of inference requests in seconds",
    ["model", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Token metrics
inference_tokens_generated = Counter(
    "skvaider_inference_tokens_generated_total",
    "Total number of tokens generated",
    ["model"],
)

inference_tokens_prompt = Counter(
    "skvaider_inference_tokens_prompt_total",
    "Total number of prompt tokens processed",
    ["model"],
)

# Model metrics
inference_model_load_duration_seconds = Histogram(
    "skvaider_inference_model_load_duration_seconds",
    "Duration of model loading in seconds",
    ["model"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

inference_model_status = Gauge(
    "skvaider_inference_model_status",
    "Model status (1=running, 0=stopped)",
    ["model"],
)

# Resource metrics
inference_memory_bytes = Gauge(
    "skvaider_inference_memory_bytes",
    "Memory usage in bytes",
    ["model", "type"],  # type: model, total, used, free
)

inference_vram_bytes = Gauge(
    "skvaider_inference_vram_bytes",
    "VRAM usage in bytes",
    ["backend", "type"],  # type: total, used, free
)
