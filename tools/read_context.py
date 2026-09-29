from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def read_context(repo_path: str, path: str, max_chars: int = 4000) -> dict[str, Any]:
    full = Path(repo_path) / path
    try:
        resolved = full.resolve()
        repo_resolved = str(Path(repo_path).resolve())
        resolved_str = str(resolved)
        if not (resolved_str == repo_resolved or resolved_str.startswith(repo_resolved + os.sep)):
            return {"ok": False, "output": "path escapes repository", "timeout": False}
        text = resolved.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n...[truncated]..."
        return {"ok": True, "output": text, "timeout": False, "truncated": truncated}
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return {"ok": False, "output": f"not found: {path}", "timeout": False}
