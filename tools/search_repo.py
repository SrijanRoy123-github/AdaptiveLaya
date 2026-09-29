from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def search_repository(repo_path: str, pattern: str, max_hits: int = 50) -> dict[str, Any]:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return {"ok": False, "output": f"bad pattern: {exc}", "timeout": False}
    hits = []
    root = Path(repo_path)
    for file in root.rglob("*.py"):
        if ".git" in file.parts or "site-packages" in file.parts:
            continue
        try:
            for i, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    rel = file.relative_to(root)
                    hits.append(f"{rel}:{i}: {line.strip()}")
                    if len(hits) >= max_hits:
                        break
        except OSError:
            continue
        if len(hits) >= max_hits:
            break
    output = "\n".join(hits) if hits else "no matches"
    return {"ok": True, "output": output, "timeout": False}
