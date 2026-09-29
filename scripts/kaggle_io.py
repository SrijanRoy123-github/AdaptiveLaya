from __future__ import annotations

import os
import shutil
from pathlib import Path


def local_mode() -> bool:
    if os.environ.get("ADAPTIVE_LAYA_LOCAL_DIR"):
        return True
    if Path("/kaggle").exists():
        return False
    return not os.environ.get("KAGGLE_KERNEL_SESSION")


def _local_store() -> Path:
    return Path(os.environ.get("ADAPTIVE_LAYA_LOCAL_DIR", "data/kaggle_local_store"))


def _slug_dir(slug: str) -> str:
    return slug.replace("/", "_")


def publish(src_dir: str | Path, slug: str, message: str, local_dir_override: str | None = None) -> None:
    src = Path(src_dir)
    if local_dir_override or local_mode():
        root = Path(local_dir_override) if local_dir_override else _local_store()
        dest = root / _slug_dir(slug)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"[local] published {src} -> {dest}")
        return
    import kagglehub

    kagglehub.dataset_upload(slug, str(src), version_notes=message)
    print(f"[kaggle] published {src} -> {slug}")


def download(slug: str, dst: str | Path | None = None) -> Path:
    if local_mode():
        root = _local_store()
        candidate = root / _slug_dir(slug)
        if candidate.exists():
            if dst:
                Path(dst).mkdir(parents=True, exist_ok=True)
                shutil.copytree(candidate, Path(dst), dirs_exist_ok=True)
                return Path(dst)
            return candidate
        raise FileNotFoundError(f"no local artifact for {slug} under {root}")
    import kagglehub

    path = Path(kagglehub.dataset_download(slug))
    if dst:
        Path(dst).mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, Path(dst), dirs_exist_ok=True)
        return Path(dst)
    return path
