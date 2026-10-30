import zipfile

import pytest

from video_cls import d2_city


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


@pytest.fixture()
def d2_zip(tmp_path):
    return make_zip(tmp_path / "d2.zip", {"1003/a.mp4": b"aaaa", "1003/b.mp4": b"bb", "1003/notes.txt": b"x"})


def test_download_and_extract_makes_no_copies(tmp_path, d2_zip, monkeypatch):
    monkeypatch.setattr(d2_city, "run", lambda **kwargs: {"output_path": str(d2_zip)})   # no network
    root = tmp_path / "d2_city"
    (root / "benchmark").mkdir(parents=True)
    (root / "benchmark" / "evt_000001.mp4").write_bytes(b"left by an older run")     # must not block

    result = d2_city.download_and_extract(root=root)

    assert result["source_count"] == 2
    assert sorted(p.name for p in (root / "raw").rglob("*.mp4")) == ["a.mp4", "b.mp4"]
    assert [p.name for p in (root / "benchmark").iterdir()] == ["evt_000001.mp4"]     # untouched


def test_download_and_extract_refuses_non_empty_raw(tmp_path, d2_zip, monkeypatch):
    monkeypatch.setattr(d2_city, "run", lambda **kwargs: {"output_path": str(d2_zip)})
    root = tmp_path / "d2_city"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "existing.mp4").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        d2_city.download_and_extract(root=root)


def test_prepare_benchmark_still_makes_copies(tmp_path, d2_zip):
    result = d2_city.prepare_benchmark(d2_zip, root=tmp_path / "out", copies=2, workers=2)
    copies = sorted(p.name for p in (tmp_path / "out" / "benchmark").glob("*.mp4"))
    assert result["file_count"] == 4 and copies == [f"evt_00000{i}.mp4" for i in range(1, 5)]
    assert (tmp_path / "out" / "benchmark" / "manifest.csv").is_file()


def test_unsafe_zip_member_is_rejected(tmp_path):
    bad = make_zip(tmp_path / "bad.zip", {"../escape.mp4": b"x"})
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        d2_city.extract_raw(bad, tmp_path / "out")
    assert not (tmp_path / "escape.mp4").exists()
