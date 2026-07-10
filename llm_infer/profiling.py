"""Lightweight timing buckets for benchmark decomposition.

CUDA timings use events so recording spans does not synchronize the device mid-decode.
Reading the summary is the explicit boundary where elapsed times become host data.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

PHASE_BUCKETS: dict[str, tuple[str, ...]] = {
    "prefill": ("prefill",),
    "decode": ("decode",),
    "page_planning": ("paged_attention_plan",),
    "attention": ("attention", "paged_attention"),
    "sampling": ("sampling",),
    "window_flushing": ("window_flushing",),
}


def attach_host_method_profile(
    profiler: TimingProfiler,
    target: object,
    bucket: str,
    method_names: tuple[str, ...],
) -> None:
    """Wrap diagnostic-only instance methods in one host timing bucket."""
    for method_name in method_names:
        original = getattr(target, method_name)
        if not callable(original):
            raise TypeError(f"{type(target).__name__}.{method_name} is not callable")

        def measured(
            *args: object,
            _original: Callable[..., object] = original,
            **kwargs: object,
        ) -> object:
            with profiler.host(bucket):
                return _original(*args, **kwargs)

        setattr(target, method_name, measured)


@dataclass
class TimingBucket:
    """One named timing bucket accumulated across a run."""

    name: str
    calls: int = 0
    total_ms: float = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms

    def as_dict(self) -> dict[str, int | float | str]:
        return {"name": self.name, "calls": self.calls, "total_ms": self.total_ms}


@dataclass
class TimingSummary:
    """JSON-ready profile for one engine run."""

    device: str
    buckets: list[TimingBucket] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        by_name = {bucket.name: bucket for bucket in self.buckets}
        phases = {}
        for phase, source_names in PHASE_BUCKETS.items():
            sources = [by_name[name] for name in source_names if name in by_name]
            phases[phase] = {
                "available": bool(sources),
                "calls": sum(bucket.calls for bucket in sources),
                "total_ms": sum(bucket.total_ms for bucket in sources),
                "mean_ms_per_call": (
                    sum(bucket.total_ms for bucket in sources)
                    / sum(bucket.calls for bucket in sources)
                    if sources
                    else None
                ),
                "source_buckets": [bucket.name for bucket in sources],
            }
        return {
            "device": self.device,
            "buckets": [bucket.as_dict() for bucket in self.buckets],
            # These spans are nested: decode includes page planning and attention. They are
            # attribution records, not values that may be summed into total wall time.
            "phases": phases,
            "phase_timings_are_nested": True,
        }


@dataclass
class _CudaSpan:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class TimingProfiler:
    """Named timing spans with deferred CUDA synchronization."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._buckets: dict[str, TimingBucket] = defaultdict(lambda: TimingBucket(""))
        self._cuda_spans: list[_CudaSpan] = []

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        """Record device work without synchronizing CUDA."""
        if self._cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_spans.append(_CudaSpan(name, start, end))
            return

        start_s = time.perf_counter()
        try:
            yield
        finally:
            self._add(name, (time.perf_counter() - start_s) * 1000.0)

    @contextmanager
    def host(self, name: str) -> Iterator[None]:
        """Record host-side work, including intentional tensor materialization boundaries."""
        start_s = time.perf_counter()
        try:
            yield
        finally:
            self._add(name, (time.perf_counter() - start_s) * 1000.0)

    def summary(self) -> TimingSummary:
        """Return accumulated timings, synchronizing CUDA events only here."""
        for span in self._cuda_spans:
            span.end.synchronize()
            self._add(span.name, span.start.elapsed_time(span.end))
        self._cuda_spans.clear()
        return TimingSummary(
            device=str(self.device),
            buckets=sorted(self._buckets.values(), key=lambda bucket: bucket.name),
        )

    def _add(self, name: str, elapsed_ms: float) -> None:
        bucket = self._buckets[name]
        if not bucket.name:
            bucket.name = name
        bucket.add(elapsed_ms)
