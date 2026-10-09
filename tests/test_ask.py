"""Smoke tests for the HR assistant.

The real integration path depends on external cloud services. To keep the
project runnable in local/offline development, we also verify the fallback
mode that answers from the local HR policy text file when credentials are
missing or invalid.
"""

from hr_assistant import config
from hr_assistant.pipeline import ask, build_hr_assistant


def test_ask_returns_a_real_answer():
    agent = build_hr_assistant()
    answer = ask(agent, "How many days of paid annual leave do I get per year?")

    assert isinstance(answer, str)
    assert answer.strip() != ""


def test_offline_fallback_works(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "JINA_API_KEY", "")
    monkeypatch.setattr(config, "PORTKEY_API_KEY", "")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")

    agent = build_hr_assistant()
    answer = ask(agent, "How many paid annual leave days do I get?")

    assert isinstance(agent, dict)
    assert agent.get("offline") is True
    assert isinstance(answer, str)
    assert "leave" in answer.lower()
