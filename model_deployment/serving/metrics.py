"""Conditional operational metrics for the shared serving application."""
from __future__ import annotations

import os


class NoOpMetric:
    """Provide the metric methods used by inference without requiring Prometheus."""

    def labels(self, **labels):
        """Return the no-op metric for any labels."""
        del labels
        return self

    def inc(self, *args, **kwargs) -> None:
        """Ignore counter increments when metrics are disabled."""
        del args, kwargs

    def observe(self, *args, **kwargs) -> None:
        """Ignore latency observations when metrics are disabled."""
        del args, kwargs

    def set(self, *args, **kwargs) -> None:
        """Ignore gauge updates when metrics are disabled."""
        del args, kwargs


class NoOpMetrics:
    """Metrics bundle used when Prometheus instrumentation is disabled."""

    enabled = False
    content_type = None

    def __init__(self) -> None:
        """Create disabled metric objects for the serving code."""
        self.requests = NoOpMetric()
        self.errors = NoOpMetric()
        self.latency = NoOpMetric()
        self.model_up = NoOpMetric()

    def render(self) -> bytes:
        """Refuse to produce fake Prometheus output."""
        raise RuntimeError("Prometheus metrics are disabled for this deployment backend.")


class PrometheusMetrics:
    """Prometheus metric bundle loaded only when instrumentation is enabled."""

    enabled = True

    def __init__(self) -> None:
        """Create the Prometheus collectors used by the model server."""
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
            generate_latest,
        )

        self.content_type = CONTENT_TYPE_LATEST
        self.registry = CollectorRegistry()
        self.generate_latest = generate_latest
        self.requests = Counter(
            "inference_requests_total",
            "Inference requests",
            ["model", "version"],
            registry=self.registry,
        )
        self.errors = Counter(
            "inference_errors_total",
            "Inference errors",
            ["model", "version"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "inference_duration_seconds",
            "Inference request duration",
            ["model", "version"],
            registry=self.registry,
        )
        self.model_up = Gauge(
            "model_up",
            "Whether the model loaded successfully",
            ["model", "version"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render the current Prometheus exposition payload."""
        return self.generate_latest(self.registry)


def prometheus_metrics_enabled() -> bool:
    """Return whether the serving image enabled Prometheus instrumentation."""
    return os.environ.get("ENABLE_PROMETHEUS_METRICS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def create_metrics():
    """Create a Prometheus or no-op metrics bundle without unconditional imports."""
    if prometheus_metrics_enabled():
        return PrometheusMetrics()
    return NoOpMetrics()
