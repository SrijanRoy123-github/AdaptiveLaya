from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

HF_CACHE = Path(os.environ.get("ADAPTIVE_LAYA_HF_CACHE", "data/raw/hf"))
KAGGLE_CACHE = Path(os.environ.get("ADAPTIVE_LAYA_KAGGLE_CACHE", "data/raw/kaggle"))
LOCAL_DIR = Path(os.environ.get("ADAPTIVE_LAYA_LOCAL_DIR", ""))

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DOMAIN_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("vlsi", re.compile(r"\b(spice|verilog|vhdl|fpga|asic|skywater|synthesis|layout|route|netlist|transistor)\b", re.IGNORECASE)),
    ("cad", re.compile(r"\b(cad|kicad|openroad|drc|floorplan|place|route|board|pcb|design rule)\b", re.IGNORECASE)),
    ("math", re.compile(r"\b(calculate|compute|solve|prove|integrate|derivative|equation|math|theorem|algebra)\b", re.IGNORECASE)),
    ("reasoning", re.compile(r"\b(reason|deduce|logic|puzzle|riddle|chain.of.thought|think step)\b", re.IGNORECASE)),
    ("book", re.compile(r"\b(book|chapter|character|plot|summary|novel|literature|author)\b", re.IGNORECASE)),
    ("science", re.compile(r"\b(experiment|hypothesis|measure|observe|analyze|formula|physics|chemistry|biology|science)\b", re.IGNORECASE)),
    ("coding", re.compile(r"\b(code|function|class|def |import |test|bug|fix|lint|run)\b", re.IGNORECASE)),
]

_TEXT_FIELDS = (
    "text", "content", "question", "instruction", "prompt",
    "problem", "description", "summary", "topology", "netlist_json",
)
_SUMMARY_FIELDS = ("context", "support", "summary", "response", "output")

_DOMAIN_ACTIONS: dict[str, tuple[str, str]] = {
    "vlsi": ("TOOL", "VLSI_SPICE"),
    "cad": ("TOOL", "CAD_DRC"),
    "circuit": ("TOOL", "VLSI_SPICE"),
    "math": ("ANSWER", "MATH_COMPUTE"),
    "reasoning": ("ASK", "REASON_DEDUCE"),
    "book": ("ANSWER", "BOOK_SUMMARIZE"),
    "science": ("ASK", "SCIENCE_QUERY"),
}


def local_mode() -> bool:
    if LOCAL_DIR:
        return True
    return not os.environ.get("KAGGLE_KERNEL_SESSION")


def _domain_from_text(text: str) -> str:
    for domain, pat in DOMAIN_KEYWORDS:
        if pat.search(text):
            return domain
    return "coding"


def _row_text(row: dict[str, Any]) -> str:
    for key in _TEXT_FIELDS:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _row_summary(row: dict[str, Any]) -> str:
    for key in _SUMMARY_FIELDS:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _heuristic_label(text: str, domain: str | None = None) -> dict[str, Any]:
    if domain is None:
        domain = _domain_from_text(text)
    action, tool = _DOMAIN_ACTIONS.get(domain, ("ANSWER", "RUN_TESTS"))
    return {"domain": domain, "label_action": action, "label_tool": tool, "label_terminal": "DONE", "label_confidence": 0.8}


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(str(path))
            for row in table.to_pylist():
                if isinstance(row, dict):
                    yield row
        except (OSError, ValueError, TypeError):
            return
    elif path.suffix == ".csv":
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                yield from csv.DictReader(fh)
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            return
    else:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            return


def _build_record(row: dict[str, Any], slug: str, index: int, domain: str) -> dict[str, Any]:
    text = _row_text(row)
    labeled = _heuristic_label(text, domain=domain)
    rec = {
        "id": f"real-{slug.replace('/', '_')}-{index:06d}",
        "user_request": text[:2000],
        "repository_summary": _row_summary(row),
        "relevant_file_context": str(row.get("context") or row.get("support") or ""),
        "recent_changes": "",
        "previous_actions": [],
        "test_results": "",
        "previous_tool_result": "",
        "step_index": 0,
        "previous_confidence": 0.0,
        "hard_negative": False,
        "task_family": labeled["domain"],
        "difficulty": row.get("difficulty", "medium"),
        "repository": f"hf/{slug}",
    }
    rec.update(labeled)
    rec["branches"] = {}
    return rec


def download_hf(
    slug: str,
    dst: Path,
    max_records: int | None = None,
    token: str | None = None,
    domain: str | None = None,
) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "decisions.jsonl"
    if out.exists():
        return out
    try:
        path = snapshot_download(repo_id=slug, local_dir=str(dst), repo_type="dataset", token=token)
    except Exception as exc:
        raise RuntimeError(
            f"HF download failed for {slug}. Public datasets need no token. "
            f"If private, set HF_TOKEN env var. Error: {exc}"
        ) from exc
    data_files = (
        sorted(Path(path).rglob("*.jsonl"))
        + sorted(Path(path).rglob("*.parquet"))
        + sorted(Path(path).rglob("*.csv"))
    )
    if not data_files:
        raise RuntimeError(f"no jsonl/parquet/csv in {slug}")
    label_domain = domain or _domain_from_text(slug)
    records: list[dict[str, Any]] = []
    done = False
    for f in data_files:
        for row in _iter_rows(f):
            if not isinstance(row, dict):
                continue
            records.append(_build_record(row, slug, len(records), label_domain))
            if max_records and len(records) >= max_records:
                done = True
                break
        if done:
            break
    payload = "\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in records) + "\n"
    out.write_bytes(payload.encode("utf-8"))
    manifest = {"count": len(records), "sha256": hashlib.sha256(payload.encode()).hexdigest(), "source": slug, "version": "v1"}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def download_kaggle(slug: str, dst: Path) -> Path:
    import kagglehub

    dst.mkdir(parents=True, exist_ok=True)
    try:
        path = kagglehub.dataset_download(slug)
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle download failed for {slug}. Public datasets need no auth. "
            f"If private, set KAGGLE_API_TOKEN env var (https://www.kaggle.com/settings/account). "
            f"Error: {exc}"
        ) from exc
    for item in Path(path).rglob("*"):
        if item.is_file():
            target = dst / item.name
            target.write_bytes(item.read_bytes())
    return dst


def ingest(dataset: dict[str, Any], hf_token: str | None = None) -> Path | None:
    domain = dataset["domain"]
    source = dataset["source"]
    slug = dataset["slug"]
    max_records = dataset.get("max_records")
    cache = HF_CACHE if source == "hf" else KAGGLE_CACHE
    out = cache / domain / slug.replace("/", "_")
    if local_mode() and (out / "decisions.jsonl").exists():
        return out / "decisions.jsonl"
    try:
        if source == "hf":
            download_hf(slug, out, max_records=max_records, token=hf_token, domain=domain)
        elif source == "kaggle":
            download_kaggle(slug, out)
        else:
            raise ValueError(f"unknown source {source}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: skip {domain} ({slug}): {exc}")
        return None
    return out / "decisions.jsonl"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/worker.yaml")
    ap.add_argument("--domain", type=str, default=None)
    ap.add_argument("--domains", type=str, default=None, help="comma-separated domain filter")
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    import yaml

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", {}).get("real", [])
    if args.domain:
        datasets = [d for d in datasets if d["domain"] == args.domain]
    if args.domains:
        keep = {d.strip() for d in args.domains.split(",") if d.strip()}
        datasets = [d for d in datasets if d["domain"] in keep]
    for ds in datasets:
        if args.max_records:
            ds = {**ds, "max_records": args.max_records}
        path = ingest(ds, hf_token=hf_token)
        if path:
            print(f"{ds['domain']}: {path}")


if __name__ == "__main__":
    main()
