from __future__ import annotations

import subprocess
import sys
from typing import Any


def run_tests(repo_path: str, target: str = ".", timeout_s: int = 60) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", target, "-q", "--tb=short"]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        output = (proc.stdout + proc.stderr)[-4000:]
        return {"ok": proc.returncode == 0, "output": output, "timeout": False}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return {"ok": False, "output": f"timeout after {timeout_s}s\n{out}", "timeout": True}
