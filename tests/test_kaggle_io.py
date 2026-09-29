import sys
from pathlib import Path

from scripts.kaggle_io import download, local_mode, publish


def test_local_mode_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("KAGGLE_KERNEL_SESSION", raising=False)
    assert local_mode() is True
    src = tmp_path / "artifact"
    src.mkdir()
    (src / "router.pt").write_bytes(b"weights")
    publish(src, "adaptive-laya/does-not-matter", "test push")
    got = download("adaptive-laya/does-not-matter")
    assert (Path(got) / "router.pt").exists()


def test_publish_uses_slug_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", str(tmp_path / "store"))
    src = tmp_path / "w0"
    src.mkdir()
    (src / "manifest.json").write_text("{}")
    publish(src, "adaptive-laya/adaptive-laya-worker-0", "round 0")
    base = tmp_path / "store" / "adaptive-laya_adaptive-laya-worker-0"
    assert (base / "manifest.json").exists()


def test_download_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("KAGGLE_KERNEL_SESSION", raising=False)
    try:
        download("adaptive-laya/nothing")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_publish_remote_uses_dataset_upload(tmp_path, monkeypatch):
    monkeypatch.delenv("ADAPTIVE_LAYA_LOCAL_DIR", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_SESSION", "1")
    calls: list[tuple[str, str, str]] = []

    class _Stub:
        @staticmethod
        def dataset_upload(slug: str, path: str, version_notes: str = "") -> None:
            calls.append((slug, path, version_notes))

        @staticmethod
        def dataset_download(slug: str) -> str:
            raise AssertionError(f"download should not be called: {slug}")

    monkeypatch.setitem(sys.modules, "kagglehub", _Stub())
    assert local_mode() is False
    src = tmp_path / "artifact"
    src.mkdir()
    (src / "router.pt").write_bytes(b"weights")
    publish(src, "judata1/adaptive-laya-worker-0", "round 0")
    assert calls == [("judata1/adaptive-laya-worker-0", str(src), "round 0")]