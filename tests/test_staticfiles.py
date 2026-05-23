"""
Tests for backend/staticfiles.py (Phase 14).

NoCacheStaticFiles should attach a Cache-Control header on every file it
serves, so the browser revalidates after every code change instead of
silently serving a stale file from disk-cache.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.staticfiles import NoCacheStaticFiles


def _make_client(tmp_path):
    (tmp_path / "index.html").write_text("<html><body>hi</body></html>")
    (tmp_path / "app.js").write_text("// hello")
    app = FastAPI()
    app.mount("/", NoCacheStaticFiles(directory=str(tmp_path), html=True), name="static")
    return TestClient(app)


def test_no_cache_header_on_html(tmp_path):
    client = _make_client(tmp_path)
    r = client.get("/index.html")
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "no-cache" in cc.lower()
    assert "must-revalidate" in cc.lower()


def test_no_cache_header_on_js(tmp_path):
    client = _make_client(tmp_path)
    r = client.get("/app.js")
    assert r.status_code == 200
    assert "no-cache" in r.headers.get("cache-control", "").lower()


def test_etag_still_works_for_304(tmp_path):
    """The no-cache header revalidates — but ETag should still cut a
    matched request down to a cheap 304 Not Modified."""
    client = _make_client(tmp_path)
    first = client.get("/app.js")
    etag = first.headers.get("etag")
    assert etag, "StaticFiles should send an ETag"
    second = client.get("/app.js", headers={"If-None-Match": etag})
    assert second.status_code == 304
