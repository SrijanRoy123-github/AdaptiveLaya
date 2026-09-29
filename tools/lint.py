from __future__ import annotations

import subprocess
import sys
from typing import Any


def lint(repo_path: str, path: str = ".", timeout_s: int = 30) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "ruff", "check", path, "--output-format", "concise"]
    try:
        proc = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout_s, check=False)
        output = (proc.stdout + proc.stderr)[-4000:]
        return {"ok": proc.returncode == 0, "output": output or "no issues", "timeout": False}
    except FileNotFoundError:
        return {"ok": False, "output": "ruff not installed", "timeout": False}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"lint timeout after {timeout_s}s", "timeout": True}
