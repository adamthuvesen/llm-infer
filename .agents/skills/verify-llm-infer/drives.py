"""llm-infer drives: the DRIVES table, the control, and the subprocess primitives they share."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
PACKAGE = "llm_infer"
PACKAGE_DIR = ROOT / "llm_infer"
INSTALL_HINT = "make install"
SHARED = ["pyproject.toml", "uv.lock", ".agents/skills/verify-llm-infer/**"]
TRIPWIRE_TOOLS = ["modal", "wandb", "huggingface-cli", "hf", "nvidia-smi"]
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
MODEL_ID = "tiny-dense"
GREEDY_REPLY = "tok_6 tok_6 tok_6 tok_6"


class DriveError(Exception):
    pass


@dataclass(frozen=True)
class Drive:
    proves: str
    paths: list[str]
    run: Callable[[Path, dict[str, str]], str]


def _record(
    evidence: Path, name: str, argv: list[str], env: dict[str, str], code: int | None
) -> None:
    redacted = {
        k: ("<redacted>" if any(m in k for m in SECRET_MARKERS) else v) for k, v in env.items()
    }
    payload = {"argv": argv, "env": redacted, "exit": code}
    (evidence / f"{name}.cmd.json").write_text(json.dumps(payload, indent=2))


def run(
    evidence: Path, env: dict[str, str], name: str, argv: list[str], timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        raise DriveError(f"{name}: timed out after {timeout}s") from None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    _record(evidence, name, argv, env, proc.returncode)
    (evidence / f"{name}.stdout").write_text(out)
    (evidence / f"{name}.stderr").write_text(err)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise DriveError(message)


def ok(result: subprocess.CompletedProcess[str], name: str) -> None:
    tail = result.stderr.strip()[-400:]
    expect(result.returncode == 0, f"{name} exited {result.returncode}: {tail}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(base: str, path: str, body: Mapping[str, object] | None = None) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, headers={"content-type": "application/json"}
    )
    try:
        with DIRECT.open(request, timeout=20) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


@contextmanager
def server(evidence: Path, env: dict[str, str]) -> Generator[str]:
    """Serve the tiny random-weight Esme bundle through the real `python -m llm_infer.serve`."""
    bundle_root = Path(tempfile.mkdtemp(dir=env["HOME"]))
    writer = "from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as w"
    write = [
        str(PYTHON),
        "-c",
        f"import sys, pathlib; {writer}; print(w(pathlib.Path(sys.argv[1])))",
    ]
    bundle = run(evidence, env, "bundle", [*write, str(bundle_root)])
    ok(bundle, "tiny bundle writer")
    port = free_port()
    argv = [str(PYTHON), "-m", "llm_infer.serve", "--backend", "esme"]
    argv += ["--bundle", bundle.stdout.strip(), "--port", str(port), "--device", "cpu"]
    log = (evidence / "server.log").open("w")
    proc = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    _record(evidence, "server", argv, env, None)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            expect(
                proc.poll() is None,
                f"server exited {proc.returncode}; see {evidence / 'server.log'}",
            )
            try:
                if http(base, "/health")[0] == 200:
                    break
            except OSError:
                pass
            expect(time.monotonic() < deadline, "server did not answer /health within 30 s")
            time.sleep(0.2)
        yield base
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log.close()


def drive_serve_http(
    evidence: Path, env: dict[str, str], expected_reply: str = GREEDY_REPLY
) -> str:
    with server(evidence, env) as base:
        status, body = http(base, "/v1/models")
        expect(json.loads(body)["data"][0]["id"] == MODEL_ID, f"/v1/models served {body}")
        chat = {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        status, body = http(base, "/v1/chat/completions", chat)
        (evidence / "chat.json").write_text(body)
        expect(status == 200, f"chat returned {status}: {body[:300]}")
        reply = json.loads(body)
        content = reply["choices"][0]["message"]["content"]
        expect(
            content == expected_reply, f"expected greedy reply {expected_reply!r}, got {content!r}"
        )
        expect(
            reply["usage"]["completion_tokens"] == 4,
            f"expected 4 completion tokens: {reply['usage']}",
        )
        status, stream = http(base, "/v1/chat/completions", chat | {"stream": True})
        (evidence / "stream.txt").write_text(stream)
        expect(
            stream.rstrip().endswith("data: [DONE]"),
            f"stream did not end with [DONE]: {stream[-200:]}",
        )
        status, body = http(base, "/v1/chat/completions", chat | {"model": "other"})
        expect(
            status == 404 and MODEL_ID in body, f"wrong model id returned {status}: {body[:200]}"
        )
        status, page = http(base, "/")
        expect(status == 200 and "<title>llm-infer</title>" in page, "web UI did not render")
        status, metrics = http(base, "/metrics")
        (evidence / "metrics.txt").write_text(metrics)
        expect(
            'llm_infer_requests_completed_total{finish_reason="length"} 2' in metrics,
            "metrics miss 2 completions",
        )
    return (
        "tiny Esme server: exact greedy chat, SSE to [DONE], 404 on a wrong model, web UI, metrics"
    )


def control_serve_http(evidence: Path, env: dict[str, str]) -> str:
    return drive_serve_http(evidence, env, expected_reply="tok_1 tok_2 tok_3 tok_4")


def drive_loadgen(evidence: Path, env: dict[str, str]) -> str:
    summary_path = evidence / "loadgen.json"
    with server(evidence, env) as base:
        argv = [str(PYTHON), "scripts/loadgen.py", "--base-url", base, "--model", MODEL_ID]
        argv += ["--concurrency", "4", "--num-requests", "8", "--max-tokens", "4"]
        local = env | {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        argv += ["--output", str(summary_path)]
        ok(run(evidence, local, "loadgen", argv, timeout=40), "loadgen")
    summary = json.loads(summary_path.read_text())
    (evidence / "loadgen.summary.txt").write_text(json.dumps(summary, indent=2))
    counts = {key: summary.get(key) for key in ("requests", "ok", "errors", "total_output")}
    want = {"requests": 8, "ok": 8, "errors": 0, "total_output": 32}
    expect(counts == want, f"expected loadgen {want}, got {counts}")
    return "loadgen streamed 8 requests at concurrency 4: 8 ok, 0 errors, 32 output tokens"


def drive_launch_plan(evidence: Path, env: dict[str, str]) -> str:
    probe = [str(PYTHON), "-W", "ignore", str(HERE / "launch_plan.py")]
    result = run(evidence, env | {"PYTHONPATH": str(ROOT)}, "launch-plan", probe, timeout=40)
    ok(result, "launch plan")
    plan = json.loads(result.stdout.strip().splitlines()[-1])
    (evidence / "launch-plan.json").write_text(json.dumps(plan, indent=2))
    serve = plan.pop("modal_esme_serve", {}).get("import", "")
    serve_command = "LLM_INFER_SERVE_APPROVED=1 modal serve scripts/modal_esme_serve.py"
    expect(serve.endswith(serve_command), f"serve app did not refuse to import: {serve!r}")
    entrypoints = gpu_functions = 0
    bad = []
    for script, members in plan.items():
        for name, (kind, gpu, outcome) in members.items():
            if kind == "function" and gpu is None:
                continue
            refused = outcome.startswith("Refusing to start") and outcome.rstrip().endswith(
                "--approved"
            )
            if not refused or f"modal run scripts/{script}.py" not in outcome:
                bad.append(f"{script}::{name}: {outcome[:160]!r}")
            if kind == "entrypoint":
                entrypoints += 1
            else:
                gpu_functions += 1
                if gpu != "A100-80GB":
                    bad.append(f"{script}::{name} runs on {gpu}, not A100-80GB")
    expect(not bad, "paths that do not refuse:\n  " + "\n  ".join(bad))
    expect(
        entrypoints >= 15 and gpu_functions >= 30,
        f"found {entrypoints} entrypoints, {gpu_functions} GPU fns",
    )
    return f"{entrypoints} entrypoints, {gpu_functions} A100 functions, and the serve app refuse"


DRIVES = {
    "serve-http": Drive(
        "python -m llm_infer.serve answers chat, streaming, models, web UI, and metrics on CPU",
        [
            "llm_infer/serve.py",
            "llm_infer/serving/**",
            "llm_infer/model/**",
            "llm_infer/scheduler/**",
            "llm_infer/kv_cache/**",
            "llm_infer/kernels/**",
            "llm_infer/fixtures/**",
        ],
        drive_serve_http,
    ),
    "loadgen": Drive(
        "scripts/loadgen.py drives concurrent streaming requests against a live server",
        ["scripts/loadgen.py", "llm_infer/serving/server/**"],
        drive_loadgen,
    ),
    "launch-plan": Drive(
        "every Modal entrypoint, A100 function, and the serve app refuse without approval",
        ["scripts/modal_*.py"],
        drive_launch_plan,
    ),
}
SMOKE = "serve-http"
# The control passes only when its drive fails on exactly this planted expectation.
CONTROL_FAILURE = "expected greedy reply 'tok_1 tok_2 tok_3 tok_4', got "
CONTROL = ("serve-http", control_serve_http)
