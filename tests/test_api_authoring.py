"""
API-level tests for /strategies/validate and /strategies/translate (Phase 6).
Groq is monkeypatched — no network calls.
"""
from __future__ import annotations

import pytest

from backend.llm import groq_client


GOOD_CODE = """
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 5))
    long = int(params.get('long', 20))
    return (data['close'].rolling(short).mean() > data['close'].rolling(long).mean()).astype(float)
"""

BANNED_IMPORT = "import os\n\ndef strategy(data, params):\n    return data['close'] * 0\n"


# ---- /strategies/validate ----------------------------------------------

def test_validate_strategy_happy(api_client):
    resp = api_client.post("/strategies/validate", json={"code": GOOD_CODE})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["errors"] == []
    assert body["strategy_name"] == "strategy"


def test_validate_strategy_rejects_banned_import(api_client):
    resp = api_client.post("/strategies/validate", json={"code": BANNED_IMPORT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert any("Disallowed import" in e or "os" in e for e in body["errors"])


def test_validate_strategy_rejects_empty_code(api_client):
    # Pydantic rejects empty strings via min_length=1 → 422.
    resp = api_client.post("/strategies/validate", json={"code": ""})
    assert resp.status_code == 422, resp.text


def test_validate_strategy_does_not_save(api_client):
    # After validate, the strategies list is still empty.
    api_client.post("/strategies/validate", json={"code": GOOD_CODE})
    listed = api_client.get("/strategies").json()
    assert listed == []


# ---- /strategies/translate (mocked Groq) -------------------------------

def _install_fake_groq(monkeypatch, content: str):
    from types import SimpleNamespace

    class _Comp:
        def create(self, **_kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=_Comp()))
    monkeypatch.setattr(groq_client, "_get_client", lambda: fake)


def test_translate_strategy_happy(api_client, monkeypatch):
    _install_fake_groq(monkeypatch, GOOD_CODE)
    resp = api_client.post("/strategies/translate", json={"description": "SMA crossover"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "def strategy" in body["code"]
    assert body["validation_result"]["ok"] is True


def test_translate_strategy_returns_validation_for_bad_output(api_client, monkeypatch):
    """When the LLM emits code that fails validation, the route still
    returns 200 — the UI shows the errors and lets the user fix or
    regenerate."""
    _install_fake_groq(monkeypatch, BANNED_IMPORT)
    resp = api_client.post("/strategies/translate", json={"description": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["validation_result"]["ok"] is False
    assert body["validation_result"]["errors"]


def test_translate_strategy_groq_failure_returns_502(api_client, monkeypatch):
    class _Boom:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **_kw):
            raise RuntimeError("groq down")
    monkeypatch.setattr(groq_client, "_get_client", lambda: _Boom())
    resp = api_client.post("/strategies/translate", json={"description": "anything"})
    assert resp.status_code == 502, resp.text
    assert "groq down" in resp.json()["detail"].lower() or "LLM translation failed" in resp.json()["detail"]


# ---- Phase 16.2 — strategy versioning ----------------------------------

UPDATED_CODE = """
import pandas as pd

def strategy(data, params):
    # Faster crossover than GOOD_CODE — different code, so a new version.
    return (data['close'].rolling(3).mean() > data['close'].rolling(10).mean()).astype(float)
"""


def _create_strategy(api_client, name="V test"):
    resp = api_client.post(
        "/strategies",
        json={"name": name, "description": "", "code": GOOD_CODE},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_create_strategy_returns_code_hash_and_inserts_initial_version(api_client):
    body = _create_strategy(api_client)
    assert body["code_hash"] is not None
    sid = body["strategy_id"]

    versions = api_client.get(f"/strategies/{sid}/versions").json()
    assert versions["current_hash"] == body["code_hash"]
    assert len(versions["versions"]) == 1
    assert versions["versions"][0]["is_current"] is True


def test_update_strategy_creates_new_version(api_client):
    body = _create_strategy(api_client)
    sid, original_hash = body["strategy_id"], body["code_hash"]

    upd = api_client.put(
        f"/strategies/{sid}",
        json={"name": "renamed", "description": "edited", "code": UPDATED_CODE},
    )
    assert upd.status_code == 200, upd.text
    new_hash = upd.json()["code_hash"]
    assert new_hash != original_hash

    versions = api_client.get(f"/strategies/{sid}/versions").json()
    assert versions["current_hash"] == new_hash
    assert len(versions["versions"]) == 2
    hashes = [v["code_hash"] for v in versions["versions"]]
    assert hashes == [new_hash, original_hash]  # newest first


def test_update_strategy_with_identical_code_is_version_noop(api_client):
    body = _create_strategy(api_client)
    sid, original_hash = body["strategy_id"], body["code_hash"]

    upd = api_client.put(
        f"/strategies/{sid}",
        json={"name": "renamed", "description": "edited", "code": GOOD_CODE},
    )
    assert upd.status_code == 200
    assert upd.json()["code_hash"] == original_hash

    versions = api_client.get(f"/strategies/{sid}/versions").json()
    assert len(versions["versions"]) == 1


def test_update_unknown_strategy_404(api_client):
    resp = api_client.put(
        "/strategies/nope",
        json={"name": "x", "description": "", "code": GOOD_CODE},
    )
    assert resp.status_code == 404


def test_update_rejects_invalid_code(api_client):
    body = _create_strategy(api_client)
    sid = body["strategy_id"]
    resp = api_client.put(
        f"/strategies/{sid}",
        json={"name": "x", "description": "", "code": "import os\n"},
    )
    assert resp.status_code == 400


def test_get_strategy_version_round_trip(api_client):
    body = _create_strategy(api_client)
    sid, original_hash = body["strategy_id"], body["code_hash"]
    api_client.put(
        f"/strategies/{sid}",
        json={"name": "x", "description": "", "code": UPDATED_CODE},
    )

    # The old version's code is still retrievable by its hash.
    detail = api_client.get(f"/strategies/{sid}/versions/{original_hash}").json()
    assert detail["code_hash"] == original_hash
    assert detail["is_current"] is False
    assert "def strategy" in detail["code"]


def test_get_strategy_detail_includes_current_hash(api_client):
    body = _create_strategy(api_client)
    sid = body["strategy_id"]
    detail = api_client.get(f"/strategies/{sid}").json()
    assert detail["code_hash"] == body["code_hash"]


# ---- Phase 16.4 — strategy fork ----------------------------------------


def test_fork_creates_independent_copy(api_client):
    src = _create_strategy(api_client, name="SMA")
    sid = src["strategy_id"]

    resp = api_client.post(f"/strategies/{sid}/fork")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    new_id = body["strategy_id"]
    assert new_id != sid

    # Both strategies appear in the list; the copy carries the suffix.
    listed = {s["id"]: s for s in api_client.get("/strategies").json()}
    assert new_id in listed
    assert listed[new_id]["name"] == "SMA (copy)"

    # Code matches the source. Hash also matches because the copy's
    # initial version is keyed on (strategy_id, code_hash) — independent
    # strategy_id rows share a code body but live in separate buckets.
    detail = api_client.get(f"/strategies/{new_id}").json()
    src_detail = api_client.get(f"/strategies/{sid}").json()
    assert detail["code"] == src_detail["code"]
    assert detail["code_hash"] == src_detail["code_hash"]

    # The copy has exactly one version row.
    versions = api_client.get(f"/strategies/{new_id}/versions").json()
    assert len(versions["versions"]) == 1


def test_fork_does_not_copy_favourite(api_client):
    src = _create_strategy(api_client)
    sid = src["strategy_id"]
    api_client.post(f"/strategies/{sid}/favourite", json={"favourite": True})

    fork = api_client.post(f"/strategies/{sid}/fork").json()
    new_id = fork["strategy_id"]
    listed = {s["id"]: s for s in api_client.get("/strategies").json()}
    assert listed[new_id]["favourite"] is False
    assert listed[sid]["favourite"] is True


def test_fork_unknown_strategy_404(api_client):
    resp = api_client.post("/strategies/nope/fork")
    assert resp.status_code == 404


def test_fork_deleted_strategy_404(api_client):
    src = _create_strategy(api_client)
    sid = src["strategy_id"]
    api_client.delete(f"/strategies/{sid}")
    resp = api_client.post(f"/strategies/{sid}/fork")
    assert resp.status_code == 404
