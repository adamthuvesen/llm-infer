"""Drive a running llm-infer server with N concurrent requests and report real latency.

A load generator, not a benchmark harness: it points async ``httpx`` at a live server's
OpenAI-compatible ``/v1/chat/completions`` endpoint, fires ``--concurrency`` requests at a
time until ``--num-requests`` have run, and reports the numbers a serving demo needs —
per-request and aggregate tokens/s, TTFT (p50/p99), end-to-end latency (p50/p99), total
throughput, and an error count. Streaming is the default (so TTFT is the time to the first
SSE delta); ``--no-stream`` measures a single blocking call where TTFT equals total latency.

This talks to the server over HTTP exactly as any OpenAI client would; it does not import the
engine. Start a server first (``python -m llm_infer.serve``) and point ``--base-url`` at it, or
in tests run it against the in-process ASGI app over ``httpx.ASGITransport``.

Usage::

    uv run python scripts/loadgen.py --base-url http://127.0.0.1:8000 \\
        --concurrency 16 --num-requests 64 --max-tokens 64 \\
        --model Qwen/Qwen2.5-Coder-3B-Instruct --prompt "Write a haiku about caches."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    """One completed request's measured timings, or its error."""

    ttft_s: float | None
    latency_s: float
    output_tokens: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.ok or self.latency_s <= 0:
            return None
        return self.output_tokens / self.latency_s


def _chat_payload(model: str, prompt: str, max_tokens: int, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }


async def _run_streaming(client: httpx.AsyncClient, payload: dict) -> RequestResult:
    """One streaming request: TTFT is the first content delta, tokens counted from deltas."""
    start = time.perf_counter()
    ttft: float | None = None
    output_tokens = 0
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", "replace")[:200]
            error = f"HTTP {resp.status_code}: {body}"
            return RequestResult(None, time.perf_counter() - start, 0, error=error)
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                output_tokens += 1
    return RequestResult(ttft, time.perf_counter() - start, output_tokens)


async def _run_blocking(client: httpx.AsyncClient, payload: dict) -> RequestResult:
    """One non-streaming request: TTFT equals total latency, tokens from the usage block."""
    start = time.perf_counter()
    resp = await client.post("/v1/chat/completions", json=payload)
    latency = time.perf_counter() - start
    if resp.status_code != 200:
        return RequestResult(None, latency, 0, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
    body = resp.json()
    output_tokens = body.get("usage", {}).get("completion_tokens", 0)
    return RequestResult(latency, latency, output_tokens)


async def run_load(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    stream: bool,
) -> tuple[list[RequestResult], float]:
    """Fire ``num_requests`` through a ``concurrency``-wide semaphore; return results + wall-clock.

    The semaphore caps in-flight requests so the server sees exactly ``concurrency`` concurrent
    streams, which is what makes continuous batching visible. The returned wall-clock is the span
    from the first request launched to the last one finished — the basis for total throughput.
    """
    payload = _chat_payload(model, prompt, max_tokens, stream)
    run_one = _run_streaming if stream else _run_blocking
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded() -> RequestResult:
        async with semaphore:
            try:
                return await run_one(client, payload)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                return RequestResult(None, 0.0, 0, error=f"{type(exc).__name__}: {exc}")

    wall_start = time.perf_counter()
    results = await asyncio.gather(*(guarded() for _ in range(num_requests)))
    wall_s = time.perf_counter() - wall_start
    return list(results), wall_s


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a non-empty list (pct in [0, 100])."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = max(0, min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[rank]


def summarize(results: list[RequestResult], wall_s: float) -> dict[str, object]:
    """Aggregate per-request timings into the headline serving numbers."""
    ok = [r for r in results if r.ok]
    errors = len(results) - len(ok)
    total_tokens = sum(r.output_tokens for r in ok)
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    latencies = [r.latency_s for r in ok]
    per_req_tps = [tps for r in ok if (tps := r.tokens_per_second) is not None]
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": errors,
        "wall_s": wall_s,
        "total_output_tokens": total_tokens,
        "throughput_tok_s": (total_tokens / wall_s) if wall_s > 0 else float("nan"),
        "ttft_p50": _percentile(ttfts, 50) if ttfts else float("nan"),
        "ttft_p99": _percentile(ttfts, 99) if ttfts else float("nan"),
        "latency_p50": _percentile(latencies, 50) if latencies else float("nan"),
        "latency_p99": _percentile(latencies, 99) if latencies else float("nan"),
        "per_request_tok_s_mean": statistics.fmean(per_req_tps) if per_req_tps else float("nan"),
    }


def format_summary(summary: dict[str, object], *, concurrency: int, stream: bool) -> str:
    """A clean fixed-width summary table for the terminal."""
    mode = "streaming" if stream else "blocking"
    rows = [
        ("requests (ok/total)", f"{summary['ok']}/{summary['requests']}"),
        ("errors", f"{summary['errors']}"),
        ("concurrency", f"{concurrency} ({mode})"),
        ("wall-clock", f"{summary['wall_s']:.3f} s"),
        ("output tokens", f"{summary['total_output_tokens']}"),
        ("throughput", f"{summary['throughput_tok_s']:.1f} tok/s"),
        ("per-request tok/s (mean)", f"{summary['per_request_tok_s_mean']:.1f}"),
        (
            "TTFT p50 / p99",
            f"{summary['ttft_p50'] * 1000:.1f} / {summary['ttft_p99'] * 1000:.1f} ms",
        ),
        ("latency p50 / p99", f"{summary['latency_p50']:.3f} / {summary['latency_p99']:.3f} s"),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["llm-infer load generator", "-" * (width + 24)]
    lines += [f"{label.ljust(width)}  {value}" for label, value in rows]
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> int:
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        results, wall_s = await run_load(
            client,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            stream=not args.no_stream,
        )
    summary = summarize(results, wall_s)
    print(format_summary(summary, concurrency=args.concurrency, stream=not args.no_stream))
    return 1 if summary["errors"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent load generator for the llm-infer server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--prompt", default="Write a short haiku about paged attention.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--no-stream", action="store_true", help="use a single blocking call per request"
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
