"""Tests for the dataset download helper (imu2text/download.py).

Nothing here touches the network. The catalog is checked for internal
consistency, and the archive handling is exercised against ZIPs built in a
temp directory - including a malicious one, since these archives are fetched
over the network and unpacked on a developer's machine.
"""

import os
import zipfile

import pytest

from imu2text import download as D


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def test_catalog_is_not_empty():
    assert len(D.DATASETS) >= 15


def test_every_entry_has_an_https_url_under_the_onhw_base():
    """A plain-HTTP or off-host URL would silently download something else."""
    for key, spec in D.DATASETS.items():
        assert spec.url.startswith("https://"), f"{key} is not HTTPS"
        assert spec.url.startswith(D.ONHW_BASE + "/"), f"{key} points off-host"


def test_every_entry_is_described():
    for key, spec in D.DATASETS.items():
        assert spec.description.strip(), f"{key} has no description"
        assert spec.size.strip(), f"{key} has no size"


def test_urls_are_unique():
    """Two keys pointing at one archive means one of them is a typo."""
    urls = [spec.url for spec in D.DATASETS.values()]
    assert len(urls) == len(set(urls))


def test_the_small_smoke_test_archives_are_present():
    """The docs tell people to start with these; they must exist as keys."""
    for key in ("onhw_chars_L", "onhw_symbols_L", "onhw_words500_indep_L"):
        assert key in D.DATASETS


def test_download_one_rejects_an_unknown_key(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset"):
        D.download_one("not_a_dataset", str(tmp_path))


# --------------------------------------------------------------------------- #
# Archive handling
# --------------------------------------------------------------------------- #
def _make_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return path


def test_safe_extract_unpacks_a_normal_archive(tmp_path):
    zip_path = _make_zip(tmp_path / "ok.zip", ["Root/a.pkl", "Root/sub/b.pkl"])
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        D._safe_extract(zf, str(out))
    assert (out / "Root" / "a.pkl").exists()
    assert (out / "Root" / "sub" / "b.pkl").exists()


def test_safe_extract_refuses_a_path_traversal_member(tmp_path):
    """Zip slip: a member named ../evil must never be written outside out_dir."""
    zip_path = _make_zip(tmp_path / "evil.zip", ["Root/a.pkl", "../evil.pkl"])
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        with pytest.raises(ValueError, match="refusing to extract"):
            D._safe_extract(zf, str(out))
    assert not (tmp_path / "evil.pkl").exists()


def test_safe_extract_refuses_a_deeply_nested_escape(tmp_path):
    zip_path = _make_zip(tmp_path / "evil2.zip", ["Root/../../escaped.pkl"])
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        with pytest.raises(ValueError, match="refusing to extract"):
            D._safe_extract(zf, str(out))
    assert not (tmp_path.parent / "escaped.pkl").exists()


def test_safe_extract_refuses_an_absolute_member(tmp_path):
    """ZipFile strips leading slashes, but check the guard holds regardless."""
    zip_path = tmp_path / "abs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        info = zipfile.ZipInfo("/tmp/absolute_escape.pkl")
        zf.writestr(info, b"x")
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        try:
            D._safe_extract(zf, str(out))
        except ValueError:
            pass  # refused outright is fine
    assert not os.path.exists("/tmp/absolute_escape.pkl")


def test_archive_root_finds_the_single_top_level_folder(tmp_path):
    zip_path = _make_zip(
        tmp_path / "one.zip", ["OnHW-chars_L/all_gt.pkl", "OnHW-chars_L/list_ids.pkl"]
    )
    with zipfile.ZipFile(zip_path) as zf:
        assert D._archive_root(zf) == "OnHW-chars_L"


def test_archive_root_is_empty_when_there_are_several_roots(tmp_path):
    """Reading namelist()[0] would have guessed one of them and been wrong."""
    zip_path = _make_zip(tmp_path / "two.zip", ["A/x.pkl", "B/y.pkl"])
    with zipfile.ZipFile(zip_path) as zf:
        assert D._archive_root(zf) == ""


def test_archive_root_is_empty_for_a_flat_archive(tmp_path):
    zip_path = _make_zip(tmp_path / "flat.zip", ["x.pkl", "y.pkl"])
    with zipfile.ZipFile(zip_path) as zf:
        assert D._archive_root(zf) == ""


def test_download_one_skips_an_existing_archive(tmp_path, monkeypatch):
    """skip_existing must not re-fetch; the network call would fail the test."""

    def explode(*a, **k):
        raise AssertionError("urlretrieve should not have been called")

    monkeypatch.setattr(D.urllib.request, "urlretrieve", explode)

    zip_path = tmp_path / "onhw_chars_L.zip"
    _make_zip(zip_path, ["OnHW-chars_L/all_gt.pkl"])
    result = D.download_one("onhw_chars_L", str(tmp_path), skip_existing=True)
    assert result.endswith("OnHW-chars_L")
    assert (tmp_path / "OnHW-chars_L" / "all_gt.pkl").exists()
