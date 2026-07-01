"""Lightweight timing buckets for benchmark decomposition.

CUDA timings use events so recording spans does not synchronize the device mid-decode.
Reading the summary is the explicit boundary where elapsed times become host data.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


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
        return {
            "device": self.device,
            "buckets": [bucket.as_dict() for bucket in self.buckets],
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
