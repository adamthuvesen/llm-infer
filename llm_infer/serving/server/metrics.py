"""Hand-rolled Prometheus counters, gauges, and histograms for ``GET /metrics``."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

# Latency buckets in seconds: sub-millisecond through ~30s covers a CPU tiny-model TTFT and a
# loaded GPU request alike. The implicit ``+Inf`` bucket is appended by the renderer.
_DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)


def _format_value(value: float) -> str:
    """Render a metric value Prometheus-style: integers bare, floats compact, +Inf literal."""
    if value == float("inf"):
        return "+Inf"
    if isinstance(value, int) or value.is_integer():
        return str(int(value))
    return repr(value)


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{val}"' for key, val in sorted(labels.items()))
    return "{" + inner + "}"


class Counter:
    """A monotonically increasing total, optionally split into labelled child series."""

    def __init__(self, name: str, help_text: str, *, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help_text = help_text
        self._label_names = label_names
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()
        if not label_names:
            self._values[()] = 0.0

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self._label_names):
            raise ValueError(
                f"counter {self.name!r} expects labels {self._label_names}, got {tuple(labels)}"
            )
        return tuple(labels[name] for name in self._label_names)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        for key, value in items:
            labels = dict(zip(self._label_names, key, strict=True))
            lines.append(f"{self.name}{_format_labels(labels)} {_format_value(value)}")
        return lines


@dataclass
class Gauge:
    """A point-in-time value sourced from a callback so a scrape reads live engine state."""

    name: str
    help_text: str
    source: Callable[[], float]

    def render(self) -> list[str]:
        return [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} gauge",
            f"{self.name} {_format_value(float(self.source()))}",
        ]


@dataclass
class SourceCounter:
    """A monotonic counter whose value is read from a callback (e.g. the engine's own tally).

    Used where the canonical running total already lives on the engine — preemptions are
    counted in the step loop — so the metric reflects that real number at scrape time rather
    than a separately-incremented copy that could drift.
    """

    name: str
    help_text: str
    source: Callable[[], float]

    def render(self) -> list[str]:
        return [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
            f"{self.name} {_format_value(float(self.source()))}",
        ]


@dataclass
class Histogram:
    """Cumulative bucket counts plus running sum/count for a latency distribution."""

    name: str
    help_text: str
    buckets: tuple[float, ...] = _DEFAULT_LATENCY_BUCKETS
    _counts: list[int] = field(init=False)
    _sum: float = field(default=0.0, init=False)
    _count: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if list(self.buckets) != sorted(self.buckets):
            raise ValueError(f"histogram {self.name!r} buckets must be ascending: {self.buckets}")
        self._counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        # Record the value in its single smallest-fitting bucket; render() turns these
        # per-bucket tallies into the cumulative "le" counts Prometheus expects. A value
        # above every bound lands only in the implicit +Inf bucket (the total count).
        with self._lock:
            self._sum += value
            self._count += 1
            for index, upper in enumerate(self.buckets):
                if value <= upper:
                    self._counts[index] += 1
                    break

    def render(self) -> list[str]:
        with self._lock:
            counts = list(self._counts)
            total = self._count
            total_sum = self._sum
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        # Bucket counts are cumulative ("le" = less-than-or-equal): each bound includes all
        # observations at or below it, and the final +Inf bucket equals the total count.
        cumulative = 0
        for upper, count in zip(self.buckets, counts, strict=True):
            cumulative += count
            lines.append(f'{self.name}_bucket{{le="{_format_value(upper)}"}} {cumulative}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {total}')
        lines.append(f"{self.name}_sum {_format_value(total_sum)}")
        lines.append(f"{self.name}_count {total}")
        return lines


class Registry:
    """Holds the server's instruments and renders them as Prometheus text exposition."""

    def __init__(self) -> None:
        self._instruments: list[Counter | Gauge | SourceCounter | Histogram] = []

    def register(self, instrument: Counter | Gauge | SourceCounter | Histogram) -> None:
        self._instruments.append(instrument)

    def render(self) -> str:
        """Emit the full exposition: every instrument's HELP/TYPE block, blank-line separated."""
        blocks = ["\n".join(instrument.render()) for instrument in self._instruments]
        return "\n".join(blocks) + "\n"


@dataclass
class ServerMetrics:
    """The serving instruments, wired to real request/token/engine state.

    Counters are bumped at the request and token choke points in the async wrapper and request
    handlers; gauges read live off the engine at scrape time via the callbacks passed to
    :meth:`bind_engine_gauges`. This object owns the registry and exposes :meth:`render`.
    """

    registry: Registry = field(default_factory=Registry)

    requests_total: Counter = field(init=False)
    requests_admitted_total: Counter = field(init=False)
    requests_completed_total: Counter = field(init=False)
    generated_tokens_total: Counter = field(init=False)
    stream_tokens_total: Counter = field(init=False)
    queue_time_seconds: Histogram = field(init=False)
    ttft_seconds: Histogram = field(init=False)
    itl_seconds: Histogram = field(init=False)
    request_latency_seconds: Histogram = field(init=False)

    def __post_init__(self) -> None:
        self.requests_total = Counter(
            "llm_infer_requests_total", "Requests accepted by the server."
        )
        self.requests_admitted_total = Counter(
            "llm_infer_requests_admitted_total",
            "Requests admitted from the server queue into the scheduler.",
        )
        self.requests_completed_total = Counter(
            "llm_infer_requests_completed_total",
            "Requests that finished generation, by finish reason.",
            label_names=("finish_reason",),
        )
        self.generated_tokens_total = Counter(
            "llm_infer_generated_tokens_total", "Output tokens generated across all requests."
        )
        self.stream_tokens_total = Counter(
            "llm_infer_stream_tokens_total",
            "Generated token stream items delivered to request consumers.",
        )
        self.queue_time_seconds = Histogram(
            "llm_infer_queue_time_seconds",
            "Time from server acceptance until scheduler admission.",
        )
        self.ttft_seconds = Histogram(
            "llm_infer_ttft_seconds", "Time from request arrival to its first generated token."
        )
        self.itl_seconds = Histogram(
            "llm_infer_itl_seconds",
            "Time between generated tokens delivered to a request consumer.",
        )
        self.request_latency_seconds = Histogram(
            "llm_infer_request_latency_seconds", "End-to-end request latency, arrival to finish."
        )
        for counter in (
            self.requests_total,
            self.requests_admitted_total,
            self.requests_completed_total,
            self.generated_tokens_total,
            self.stream_tokens_total,
        ):
            self.registry.register(counter)
        self.registry.register(self.queue_time_seconds)
        self.registry.register(self.ttft_seconds)
        self.registry.register(self.itl_seconds)
        self.registry.register(self.request_latency_seconds)

    def bind_engine_gauges(
        self,
        *,
        preemptions: Callable[[], float],
        grouped_decode_steps: Callable[[], float],
        running_requests: Callable[[], float],
        waiting_requests: Callable[[], float],
        kv_blocks_used: Callable[[], float],
        kv_blocks_free: Callable[[], float],
        kv_blocks_total: Callable[[], float],
        kv_utilization: Callable[[], float],
    ) -> None:
        """Register the live-read engine instruments. Called once at app build, after wiring.

        Preemptions are a monotonic counter read straight off the engine's own tally; the rest
        are point-in-time gauges. Both reflect real engine state at scrape time.
        """
        self.registry.register(
            SourceCounter(
                "llm_infer_preemptions_total",
                "Running requests preempted (evicted by recompute) under KV pressure.",
                preemptions,
            )
        )
        self.registry.register(
            SourceCounter(
                "llm_infer_grouped_decode_steps_total",
                "Decode window steps run by engine-owned grouped graphs (0 when disabled; "
                "a grouped deployment stuck at 0 is silently falling back).",
                grouped_decode_steps,
            )
        )
        gauges = (
            (
                "llm_infer_running_requests",
                "Requests currently in the scheduler's running set.",
                running_requests,
            ),
            (
                "llm_infer_waiting_requests",
                "Requests queued and waiting for admission (queue depth).",
                waiting_requests,
            ),
            (
                "llm_infer_kv_blocks_used",
                "Physical KV-cache blocks currently allocated.",
                kv_blocks_used,
            ),
            (
                "llm_infer_kv_blocks_free",
                "Physical KV-cache blocks currently free in the pool.",
                kv_blocks_free,
            ),
            (
                "llm_infer_kv_blocks_total",
                "Total physical KV-cache blocks in the pool.",
                kv_blocks_total,
            ),
            (
                "llm_infer_kv_utilization_ratio",
                "Fraction of the KV-cache block pool currently in use (0..1).",
                kv_utilization,
            ),
        )
        for name, help_text, source in gauges:
            self.registry.register(Gauge(name, help_text, source))

    def render(self) -> str:
        return self.registry.render()
