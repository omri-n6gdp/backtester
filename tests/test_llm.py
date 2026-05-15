"""
Tests for backend/llm/groq_client.py. Five tests, zero real Groq calls.

We monkeypatch `_get_client` to return a fake whose chat.completions.create()
returns a canned response. This isolates us from the network and lets us
assert what gets sent to the model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.llm import groq_client


class _FakeChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str, capture: list):
        self._content = content
        self._capture = capture

    def create(self, **kwargs):
        self._capture.append(kwargs)
        return _FakeCompletion(self._content)


class _FakeClient:
    def __init__(self, content: str = "def strategy(data, params):\n    return data['close'] * 0\n"):
        self.capture: list = []
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(content, self.capture))


@pytest.fixture
def fake_client(monkeypatch):
    """Default fake — tests can override `content` by accessing the fixture's
    returned client and rebuilding chat.completions before calling."""
    client = _FakeClient()
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    return client


# ---- output cleanup -----------------------------------------------------

def test_translate_strips_markdown_fences(monkeypatch):
    fenced = "```python\nimport pandas as pd\n\ndef strategy(data, params):\n    return data['close'] * 0\n```"
    client = _FakeClient(content=fenced)
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    out = groq_client.translate_strategy("anything")
    assert "```" not in out
    assert out.startswith("import pandas as pd")
    assert out.rstrip().endswith("return data['close'] * 0")


def test_translate_passes_through_unfenced_code(fake_client):
    """Plain code (no fences) should be returned essentially as-is, only stripped."""
    out = groq_client.translate_strategy("constant flat position")
    assert out.startswith("def strategy")
    assert "```" not in out


# ---- message wiring -----------------------------------------------------

def test_translate_sends_description_to_llm(fake_client):
    description = "Buy when RSI crosses above 30."
    groq_client.translate_strategy(description)
    assert len(fake_client.capture) == 1
    sent = fake_client.capture[0]
    # User message must contain the description verbatim.
    user_messages = [m for m in sent["messages"] if m["role"] == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == description


def test_translate_uses_documented_model(fake_client):
    groq_client.translate_strategy("anything")
    assert fake_client.capture[0]["model"] == "llama-3.3-70b-versatile"


# ---- error propagation --------------------------------------------------

def test_translate_propagates_groq_errors(monkeypatch):
    class _Boom:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **_kw):
            raise RuntimeError("upstream offline")
    monkeypatch.setattr(groq_client, "_get_client", lambda: _Boom())
    with pytest.raises(RuntimeError, match="upstream offline"):
        groq_client.translate_strategy("any description")


def test_translate_rejects_empty_description(fake_client):
    with pytest.raises(ValueError, match="empty"):
        groq_client.translate_strategy("")
    with pytest.raises(ValueError, match="empty"):
        groq_client.translate_strategy("   ")
