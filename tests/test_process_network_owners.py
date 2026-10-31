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
ASYNCIO_PROCESS_CALLS = {"create_subprocess_exec", "create_subprocess_shell"}
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            uses |= {b for alias in node.names if (b := _banned(alias.name))}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if banned := _banned(node.module) or _banned(f"{node.module}.{alias.name}"):
                    uses.add(banned)
                elif node.module == "os" and alias.name in OS_PROCESS_CALLS:
                    uses.add(f"os.{alias.name}")
                elif node.module == "asyncio" and alias.name in ASYNCIO_PROCESS_CALLS:
                    uses.add("asyncio.subprocess")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in OS_PROCESS_CALLS:
                uses.add(f"os.{node.attr}")
            elif node.value.id == "asyncio" and node.attr in ASYNCIO_PROCESS_CALLS:
                uses.add("asyncio.subprocess")
    return uses


def test_process_and_network_modules_stay_in_their_owners() -> None:
    strays = []
    for path in candidate_files():
        if not is_python_file(path):
            continue
        name = path.relative_to(ROOT).as_posix()
        uses = process_and_network_uses(ast.parse(path.read_text(encoding="utf-8")))
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
    )
    assert process_and_network_uses(ast.parse(source)) == {
        "subprocess",
        "http.client",
        "urllib.request",
        "httpx",
        "os.system",
        "os.execvp",
        "asyncio.subprocess",
    }


def test_every_owner_still_exists() -> None:
    missing = [name for name in OWNERS if not (ROOT / name).is_file()]
    assert not missing, f"Remove stale owners from OWNERS: {missing}"
