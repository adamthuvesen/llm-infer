"""Drive a running llm-infer server with N concurrent requests and report real latency.

A load generator, not a benchmark harness: it points async ``httpx`` at a live server's
OpenAI-compatible ``/v1/chat/completions`` endpoint, fires ``--concurrency`` requests at a
time until ``--num-requests`` have run, and reports the numbers a serving demo needs —
per-request and aggregate output rate, TTFT, inter-token latency (ITL), end-to-end latency,
queue time, total throughput, and an error count. Streaming is the default (so TTFT is the time
to the first SSE delta); ``--no-stream`` measures a single blocking call where TTFT equals total
latency and client-side ITL is unavailable.

Both modes read true completion-token counts from the response ``usage`` block. Streaming asks
the server for the final usage chunk and also records the arrival time of every visible content
delta. Client ITL is therefore HTTP-visible content-chunk spacing; the server's ``/metrics``
histogram reports generated-token ITL at the engine-to-handler boundary. Queue percentiles are
bucket upper bounds derived from the before/after server histogram, not invented client values.

This talks to the server over HTTP exactly as any OpenAI client would; it does not import the
engine. Start a server first (``python -m llm_infer.serve``) and point ``--base-url`` at it, or
in tests run it against the in-process ASGI app over ``httpx.ASGITransport``.

``--model`` must equal the id the target server actually serves (the server 404s any other
id). It defaults to ``esme-214m-chat`` for the documented Esme path
(``python -m llm_infer.serve --backend esme --bundle <path>``); pass
``Qwen/Qwen2.5-Coder-3B-Instruct`` when targeting a Qwen reference server.

Usage::

    # Esme server
    uv run python scripts/loadgen.py --base-url http://127.0.0.1:8000 \\
        --concurrency 16 --num-requests 64 --max-tokens 64 \\
        --model esme-214m-chat --prompt "Write a haiku about caches."

    # Qwen reference server
    uv run python scripts/loadgen.py --base-url http://127.0.0.1:8000 \\
        --concurrency 16 --num-requests 64 --max-tokens 64 \\
        --model Qwen/Qwen2.5-Coder-3B-Instruct --prompt "Write a haiku about caches."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class RequestResult:
    """One completed request's measured timings, plus its output size.

    ``output_count`` is true completion tokens when the response includes ``usage``. A streaming
    server that omits usage falls back to content deltas and labels the unit accordingly.
    """

    ttft_s: float | None
    latency_s: float
    output_count: int
    output_unit: str
    token_times_s: tuple[float, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def output_per_second(self) -> float | None:
        """Output units per second for this request (tokens blocking, deltas streaming)."""
        if not self.ok or self.latency_s <= 0:
            return None
        return self.output_count / self.latency_s

    @property
    def itls_s(self) -> tuple[float, ...]:
        return tuple(
            max(0.0, right - left)
            for left, right in zip(self.token_times_s, self.token_times_s[1:], strict=False)
        )


def _chat_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


async def _run_streaming(client: httpx.AsyncClient, payload: dict) -> RequestResult:
    """Measure one streaming request, including visible-delta TTFT and ITL timestamps."""
    start = time.perf_counter()
    ttft: float | None = None
    deltas = 0
    completion_tokens: int | None = None
    token_times: list[float] = []
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", "replace")[:200]
            error = f"HTTP {resp.status_code}: {body}"
            return RequestResult(None, time.perf_counter() - start, 0, "tokens", error=error)
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                completion_tokens = int(usage.get("completion_tokens", 0))
                continue
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                arrived_s = time.perf_counter()
                if ttft is None:
                    ttft = arrived_s - start
                deltas += 1
                token_times.append(arrived_s)
    output_count = completion_tokens if completion_tokens is not None else deltas
    output_unit = "tokens" if completion_tokens is not None else "deltas"
    return RequestResult(
        ttft,
        time.perf_counter() - start,
        output_count,
        output_unit,
        tuple(token_times),
    )


async def _run_blocking(client: httpx.AsyncClient, payload: dict) -> RequestResult:
    """One non-streaming request: TTFT equals total latency, true tokens from the usage block."""
    start = time.perf_counter()
    resp = await client.post("/v1/chat/completions", json=payload)
    latency = time.perf_counter() - start
    if resp.status_code != 200:
        return RequestResult(
            None,
            latency,
            0,
            "tokens",
            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )
    body = resp.json()
    completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
    return RequestResult(latency, latency, completion_tokens, "tokens")


async def run_load(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    stream: bool,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
) -> tuple[list[RequestResult], float]:
    """Fire ``num_requests`` through a ``concurrency``-wide semaphore; return results + wall-clock.

    The semaphore caps in-flight requests so the server sees exactly ``concurrency`` concurrent
    streams, which is what makes continuous batching visible. The returned wall-clock is the span
    from the first request launched to the last one finished — the basis for total throughput.
    """
    if concurrency < 1:
        # A zero-permit semaphore would block every task forever; reject it loudly.
        raise ValueError(f"concurrency must be >= 1; got {concurrency}")
    if num_requests < 1:
        raise ValueError(f"num_requests must be >= 1; got {num_requests}")
    payload = _chat_payload(
        model,
        prompt,
        max_tokens,
        stream,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    run_one = _run_streaming if stream else _run_blocking
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded() -> RequestResult:
        async with semaphore:
            try:
                return await run_one(client, payload)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                return RequestResult(
                    None,
                    0.0,
                    0,
                    "tokens" if not stream else "unknown",
                    error=f"{type(exc).__name__}: {exc}",
                )

    wall_start = time.perf_counter()
    results = await asyncio.gather(*(guarded() for _ in range(num_requests)))
    wall_s = time.perf_counter() - wall_start
    return list(results), wall_s


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile over a non-empty list (pct in [0, 100])."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = 0 if pct <= 0 else math.ceil(pct / 100.0 * len(ordered)) - 1
    rank = max(0, min(len(ordered) - 1, rank))
    return ordered[rank]


def summarize(
    results: list[RequestResult],
    wall_s: float,
    *,
    server_metrics_before: dict[str, float] | None = None,
    server_metrics_after: dict[str, float] | None = None,
) -> dict[str, object]:
    """Aggregate per-request timings into the headline serving numbers.

    ``total_output`` and the rates use true tokens when every successful response has usage;
    otherwise they retain the response unit and label it.
    """
    ok = [r for r in results if r.ok]
    errors = len(results) - len(ok)
    total_output = sum(r.output_count for r in ok)
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    latencies = [r.latency_s for r in ok]
    itls = [itl for result in ok for itl in result.itls_s]
    per_req_rates = [rate for r in ok if (rate := r.output_per_second) is not None]
    before = server_metrics_before or {}
    after = server_metrics_after or {}
    output_units = {result.output_unit for result in ok}
    output_unit = output_units.pop() if len(output_units) == 1 else "mixed"
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": errors,
        "wall_s": wall_s,
        "total_output": total_output,
        "output_unit": output_unit,
        "throughput_per_s": (total_output / wall_s) if wall_s > 0 else None,
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p95": _percentile(ttfts, 95),
        "ttft_p99": _percentile(ttfts, 99),
        "itl_p50": _percentile(itls, 50),
        "itl_p95": _percentile(itls, 95),
        "latency_p50": _percentile(latencies, 50),
        "latency_p95": _percentile(latencies, 95),
        "latency_p99": _percentile(latencies, 99),
        "queue_time_p50_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_queue_time_seconds", 50
        ),
        "queue_time_p95_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_queue_time_seconds", 95
        ),
        "server_itl_p50_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_itl_seconds", 50
        ),
        "server_itl_p95_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_itl_seconds", 95
        ),
        "per_request_per_s_mean": (statistics.fmean(per_req_rates) if per_req_rates else None),
    }


def format_summary(summary: dict[str, object], *, concurrency: int, stream: bool) -> str:
    """A clean fixed-width summary table for the terminal.

    A normal llm-infer server returns true completion tokens through ``usage`` in both modes.
    If another compatible server omits streaming usage, the caller receives a clearly labelled
    delta count instead.
    """
    mode = "streaming" if stream else "blocking"
    unit = "tok" if summary.get("output_unit") == "tokens" else str(summary.get("output_unit"))

    def number(key: str) -> float | None:
        value = summary.get(key)
        return float(value) if isinstance(value, int | float) else None

    def pair(first: str, second: str, *, scale: float, suffix: str) -> str:
        left = number(first)
        right = number(second)
        if left is None or right is None:
            return "n/a"
        return f"{left * scale:.1f} / {right * scale:.1f} {suffix}"

    throughput = number("throughput_per_s")
    mean_rate = number("per_request_per_s_mean")
    rows = [
        ("requests (ok/total)", f"{summary['ok']}/{summary['requests']}"),
        ("errors", f"{summary['errors']}"),
        ("concurrency", f"{concurrency} ({mode})"),
        ("wall-clock", f"{summary['wall_s']:.3f} s"),
        (f"output {unit}s", f"{summary['total_output']}"),
        ("throughput", f"{throughput:.1f} {unit}/s" if throughput is not None else "n/a"),
        (
            f"per-request {unit}/s (mean)",
            f"{mean_rate:.1f}" if mean_rate is not None else "n/a",
        ),
        (
            "TTFT p50 / p95",
            pair("ttft_p50", "ttft_p95", scale=1000, suffix="ms"),
        ),
        (
            "client ITL p50 / p95",
            pair("itl_p50", "itl_p95", scale=1000, suffix="ms"),
        ),
        (
            "latency p50 / p95",
            pair("latency_p50", "latency_p95", scale=1, suffix="s"),
        ),
    ]
    queue_p50 = summary.get("queue_time_p50_bucket_upper_s")
    queue_p95 = summary.get("queue_time_p95_bucket_upper_s")
    if isinstance(queue_p50, float) and isinstance(queue_p95, float):
        rows.append(
            (
                "queue p50 / p95 (bucket upper)",
                f"<= {queue_p50 * 1000:.1f} / <= {queue_p95 * 1000:.1f} ms",
            )
        )
    width = max(len(label) for label, _ in rows)
    lines = ["llm-infer load generator", "-" * (width + 24)]
    lines += [f"{label.ljust(width)}  {value}" for label, value in rows]
    return "\n".join(lines)


async def _scrape_metrics(client: httpx.AsyncClient) -> dict[str, float] | None:
    try:
        response = await client.get("/metrics")
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    samples: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            samples[name] = float(value)
        except ValueError:
            continue
    return samples


def _histogram_delta_quantile(
    before: dict[str, float],
    after: dict[str, float],
    metric: str,
    percentile: float,
) -> float | None:
    """Return the first finite bucket whose run-local cumulative delta reaches percentile."""
    count_name = f"{metric}_count"
    total = after.get(count_name, 0.0) - before.get(count_name, 0.0)
    if total <= 0:
        return None
    target = total * percentile / 100.0
    prefix = f'{metric}_bucket{{le="'
    buckets: list[tuple[float, float]] = []
    for name, value in after.items():
        if not name.startswith(prefix):
            continue
        upper_text = name[len(prefix) :].split('"', 1)[0]
        if upper_text == "+Inf":
            continue
        upper = float(upper_text)
        delta = value - before.get(name, 0.0)
        buckets.append((upper, delta))
    for upper, cumulative in sorted(buckets):
        if cumulative >= target:
            return upper
    return None


async def _main(args: argparse.Namespace) -> int:
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=args.concurrency,
            max_keepalive_connections=args.concurrency,
        ),
    ) as client:
        warmup_s = 0.0
        if args.warmup_requests:
            _, warmup_s = await run_load(
                client,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                concurrency=min(args.concurrency, args.warmup_requests),
                num_requests=args.warmup_requests,
                stream=not args.no_stream,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
            )
        metrics_before = await _scrape_metrics(client)
        results, wall_s = await run_load(
            client,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            stream=not args.no_stream,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        metrics_after = await _scrape_metrics(client)
    summary = summarize(
        results,
        wall_s,
        server_metrics_before=metrics_before,
        server_metrics_after=metrics_after,
    )
    summary["warmup_requests"] = args.warmup_requests
    summary["warmup_s"] = warmup_s
    summary["measurement_scope"] = "persistent_server_requests_only"
    summary["sampling"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    print(format_summary(summary, concurrency=args.concurrency, stream=not args.no_stream))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 1 if summary["errors"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent load generator for the llm-infer server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        default="esme-214m-chat",
        help=(
            "model id the target server serves; defaults to Esme, use "
            "'Qwen/Qwen2.5-Coder-3B-Instruct' for a Qwen reference server"
        ),
    )
    parser.add_argument("--prompt", default="Write a short haiku about paged attention.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--no-stream", action="store_true", help="use a single blocking call per request"
    )
    args = parser.parse_args()
    for name in ("concurrency", "num_requests", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1; got {getattr(args, name)}")
    if args.warmup_requests < 0:
        parser.error(f"--warmup-requests must be >= 0; got {args.warmup_requests}")
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
