from __future__ import annotations

from typing import Any

from tools.lint import lint
from tools.read_context import read_context
from tools.run_tests import run_tests
from tools.search_repo import search_repository

TOOL_NAMES = ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]

_EXECUTORS = {
    "RUN_TESTS": lambda repo, args: run_tests(
        repo, args.get("target", "."), int(args.get("timeout_s", 60))
    ),
    "LINT": lambda repo, args: lint(repo, args.get("path", ".")),
    "READ_CONTEXT": lambda repo, args: read_context(repo, args.get("path", ".")),
    "SEARCH_REPOSITORY": lambda repo, args: search_repository(repo, args.get("pattern", "")),
}


def execute(tool: str, args: dict[str, Any], repo_path: str) -> dict[str, Any]:
    if tool not in _EXECUTORS:
        return {"ok": False, "output": f"unknown tool: {tool}", "timeout": False}
    return _EXECUTORS[tool](repo_path, args or {})
