"""Process and network modules stay in the files that own them.

A process call or an HTTP client in a new module can start a server, reach the network, or spend
without anyone noticing. Add a file to `OWNERS` only when it needs the module.
"""

from __future__ import annotations

import ast

from tests.support.repo_files import ROOT, candidate_files, is_python_file

BANNED_MODULES = (
    "subprocess",
    "asyncio.subprocess",
    "pty",
    "socket",
    "http.client",
    "urllib.request",
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "modal",
)
OS_PROCESS_CALLS = {
    *(f"exec{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
    *(f"spawn{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
    "system",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "fork",
    "forkpty",
}
# Event-loop and asyncio methods that start a process or open a connection, on any receiver.
ASYNC_CALLS = {
    **dict.fromkeys(
        (
            "create_subprocess_exec",
            "create_subprocess_shell",
            "subprocess_exec",
            "subprocess_shell",
        ),
        "asyncio.subprocess",
    ),
    **dict.fromkeys(
        ("open_connection", "open_unix_connection", "create_connection", "create_unix_connection"),
        "asyncio.connection",
    ),
}
# Modal bills. The spend-gate scans and the launch-plan drive read only these files.
MODAL_OWNERS = ("scripts/modal_*.py", ".agents/skills/verify-llm-infer/launch_plan.py")
VERIFY = ".agents/skills/verify-llm-infer"
OWNERS = {
    f"{VERIFY}/drives.py": {"socket", "subprocess", "urllib.request"},
    f"{VERIFY}/verify": {"subprocess"},
    "llm_infer/benchmarks/report.py": {"subprocess"},
    "scripts/check_benchmark_evidence.py": {"subprocess"},
    "scripts/esme_serving_eval.py": {"httpx", "socket"},
    "scripts/loadgen.py": {"httpx"},
    "scripts/modal_esme_decode_profile.py": {"subprocess"},
    "scripts/modal_esme_fused_startup.py": {"subprocess"},
    "scripts/modal_esme_vllm_baseline.py": {"socket", "subprocess"},
    "scripts/modal_reference_check.py": {"subprocess"},
    "tests/support/repo_files.py": {"subprocess"},
    "tests/benchmarks/test_evidence_contract.py": {"subprocess"},
    "tests/serving/test_esme_kv_trace.py": {"subprocess"},
    **{
        f"tests/{name}.py": {"httpx"}
        for name in (
            "model/test_runtime",
            "serving/test_api",
            "serving/test_decode_graph_serving",
            "serving/test_esme_serving",
            "serving/test_loadgen",
            "serving/test_metrics",
            "serving/test_prefix_cache_serving",
            "serving/test_stop",
            "serving/test_webui",
        )
    },
}


def _banned(module: str) -> str | None:
    return next((b for b in BANNED_MODULES if module == b or module.startswith(f"{b}.")), None)


def process_and_network_uses(tree: ast.Module) -> set[str]:
    uses: set[str] = set()
    os_names = {"os"} | {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "os" and alias.asname
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            uses |= {b for alias in node.names if (b := _banned(alias.name))}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if banned := _banned(node.module) or _banned(f"{node.module}.{alias.name}"):
                    uses.add(banned)
                elif node.module == "os" and alias.name in OS_PROCESS_CALLS:
                    uses.add(f"os.{alias.name}")
                elif node.module == "asyncio" and alias.name in ASYNC_CALLS:
                    uses.add(ASYNC_CALLS[alias.name])
        elif isinstance(node, ast.Attribute):
            if node.attr in ASYNC_CALLS:
                uses.add(ASYNC_CALLS[node.attr])
            elif (
                isinstance(node.value, ast.Name)
                and node.value.id in os_names
                and node.attr in OS_PROCESS_CALLS
            ):
                uses.add(f"os.{node.attr}")
    return uses


def test_process_and_network_modules_stay_in_their_owners() -> None:
    strays = []
    for path in candidate_files():
        if not is_python_file(path):
            continue
        name = path.relative_to(ROOT).as_posix()
        uses = process_and_network_uses(ast.parse(path.read_text(encoding="utf-8")))
        if any(path.relative_to(ROOT).full_match(pattern) for pattern in MODAL_OWNERS):
            uses.discard("modal")
        if extra := uses - OWNERS.get(name, set()):
            strays.append(f"{name}: {', '.join(sorted(extra))}")
    assert not strays, (
        "Process or network use outside its owner files. Call an owner instead, or add the file "
        "to `OWNERS` in tests/test_process_network_owners.py if it must own the call:\n  "
        + "\n  ".join(strays)
    )


def test_the_scan_finds_each_form() -> None:
    source = (
        "import subprocess\nfrom http import client\nfrom urllib.request import urlopen\n"
        "import httpx as h\nfrom os import system\nimport asyncio\nimport os\n"
        "asyncio.create_subprocess_exec('x')\nos.execvp('x', [])\nfrom . import socket\n"
        "import os as o\no.system('x')\nloop.subprocess_exec(f)\nasyncio.open_connection(h, p)\n"
        "import modal\n"
    )
    assert process_and_network_uses(ast.parse(source)) == {
        "subprocess",
        "http.client",
        "urllib.request",
        "httpx",
        "os.system",
        "os.execvp",
        "asyncio.subprocess",
        "asyncio.connection",
        "modal",
    }


def test_every_owner_still_exists() -> None:
    missing = [name for name in OWNERS if not (ROOT / name).is_file()]
    assert not missing, f"Remove stale owners from OWNERS: {missing}"
