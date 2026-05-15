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
