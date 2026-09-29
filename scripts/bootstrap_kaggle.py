from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.kaggle_io import download


def verify_checksum(jsonl_path: Path, expected_sha: str | None) -> None:
    payload = jsonl_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha and digest != expected_sha:
        raise SystemExit(f"checksum mismatch: {digest} != {expected_sha}")


def ensure_shards(decisions_dir: Path, out_dir: Path, worker_id: int) -> Path:
    shard = out_dir / f"shard-{worker_id}.jsonl"
    if shard.exists():
        return shard
    subprocess.run(
        [sys.executable, "scripts/make_shards.py",
         "--input", str(decisions_dir / "decisions.jsonl"),
         "--out", str(out_dir)],
        check=True,
    )
    return shard


def _get_secret(name: str, default: str = "") -> str:
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name) or default
    except Exception:  # noqa: BLE001
        return os.environ.get(name, default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--mode", type=str, default="probe", choices=["probe", "lora"])
    ap.add_argument("--data-dir", type=str, default="data/decisions_live")
    ap.add_argument("--shards-dir", type=str, default="data/processed/shards")
    ap.add_argument("--repo-url", type=str, default="")
    ap.add_argument("--commit", type=str, default="")
    args = ap.parse_args()

    repo_url = args.repo_url or _get_secret("ADAPTIVE_LAYA_REPO")
    commit = args.commit or _get_secret("ADAPTIVE_LAYA_COMMIT")

    if repo_url and not Path("configs/worker.yaml").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
        token = _get_secret("GITHUB_TOKEN")
        clone_url = repo_url
        if token and repo_url.startswith("https://github.com/"):
            clone_url = repo_url.replace("https://", f"https://{token}@")
        subprocess.run(["git", "clone", clone_url, "."], check=True)
        if commit:
            subprocess.run(["git", "checkout", commit], check=True)

    workers_cfg = yaml.safe_load(Path("configs/worker.yaml").read_text(encoding="utf-8"))
    decisions_slug = workers_cfg["datasets"]["decisions"]

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not (data_dir / "decisions.jsonl").exists():
        download(decisions_slug, dst=data_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_checksum(data_dir / "decisions.jsonl", manifest.get("sha256"))
    else:
        manifest = {}

    shard = ensure_shards(data_dir, Path(args.shards_dir), args.worker_id)
    print(json.dumps({
        "worker_id": args.worker_id,
        "mode": args.mode,
        "shard": str(shard),
        "records_sha": manifest.get("sha256", "unverified"),
    }, indent=2))


if __name__ == "__main__":
    main()