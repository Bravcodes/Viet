"""Streamlit AppTest runs of the real app script, with Groq faked."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend_v3 as backend  # noqa: E402

APP = str(ROOT / "streamlit_app.py")

PLAN = {
    "choices": [
        {"metric": "average_net_price", "weight": 5, "explanation": "You asked for great aid."},
        {"metric": "DEBT_MDN", "weight": 3, "explanation": "Low debt."},
        {"metric": "MD_EARN_WNE_P10", "weight": 2, "explanation": "Job outcomes."},
    ],
    "unsupported": ["Prestigious employer reputation", "Data Science program quality"],
    "cautions": ["Average net price is not a personalized aid offer."],
    "requested_major": "Data Science",
}
REPORT = {
    "original_winner_explanation": "ORIGINAL WINNER TEXT",
    "custom_winner_explanation": "CUSTOM WINNER TEXT",
    "why_rankings_changed": "CHANGED TEXT",
    "key_tradeoffs": ["TRADEOFF"],
}


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, model, messages, response_format, temperature):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        body = REPORT if "colleges" in payload else PLAN
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))])


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def fake_ai(monkeypatch):
    completions = FakeCompletions()
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        backend, "make_client", lambda key: SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    return completions


def run(at):
    # Accepts an AppTest or a clicked widget; both have .run()
    at = at.run(timeout=30)
    assert not at.exception, [e.message for e in at.exception]
    return at


def all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [w.value for w in at.warning] + [i.value for i in at.info] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


def button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def test_data_only_mode_full_flow(no_key):
    at = run(AppTest.from_file(APP))
    assert "Data-only demo" in all_text(at)

    run(button(at, "⚖️ Balanced").click())
    assert at.radio(key="k_step").value.startswith("2")
    assert len(at.slider) == 4

    # Raise earnings importance and see matches change
    run(button(at, "See my matches").click())
    top_before = at.dataframe[0].value["College"].tolist()
    assert "1,988" in all_text(at)

    at.radio(key="k_step").set_value("2 · Review")
    run(at)
    at.slider(key="k_w_MD_EARN_WNE_P10").set_value(10)
    at.slider(key="k_w_average_net_price").set_value(1)
    run(at)
    at.radio(key="k_step").set_value("3 · Matches")
    run(at)
    assert at.dataframe[0].value["College"].tolist() != top_before

    # Filter to an empty result without crashing
    at.toggle(key="k_price_on").set_value(True)
    run(at)
    at.slider(key="k_max_price").set_value(0)
    run(at)
    assert "No scored colleges match" in all_text(at)

    at.toggle(key="k_price_on").set_value(False)
    run(at)
    at.radio(key="k_step").set_value("4 · Compare")
    run(at)
    assert "Points add up to the match score" in all_text(at)


def test_ai_flow_calls_agents_once(fake_ai):
    at = run(AppTest.from_file(APP))
    at.text_area(key="k_prompt").set_value(
        "I want Data Science with great financial aid and prestigious employer reputation"
    )
    run(button(at, "🔎 Find my matches").click())
    assert len(fake_ai.calls) == 2  # Agent 1 + Agent 2

    text = all_text(at)
    assert at.radio(key="k_step").value.startswith("2")
    assert "Not measured yet" in text
    assert "Prestigious employer reputation" in text
    assert set(at.multiselect(key="k_codes").value) == {"30.7001", "30.7099"}

    # Sliders and filters never call the AI
    at.slider(key="k_w_MD_EARN_WNE_P10").set_value(10)
    run(at)
    at.radio(key="k_step").set_value("3 · Matches")
    run(at)
    at.multiselect(key="k_states").set_value(["CA"])
    run(at)
    assert len(fake_ai.calls) == 2
    assert "reported 2024 bachelor's graduates in Data Science" in all_text(at)
    assert set(at.dataframe[0].value["State"]) == {"CA"}

    # Agent 3 runs only on request, with verified evidence
    run(button(at, "✨ Explain my top matches").click())
    assert len(fake_ai.calls) == 3
    assert fake_ai.calls[2]["unsupported_priorities"] == PLAN["unsupported"]
    assert "CUSTOM WINNER TEXT" in all_text(at)

    # Changing a weight marks the explanation as stale, still no new call
    at.radio(key="k_step").set_value("2 · Review")
    run(at)
    at.slider(key="k_w_DEBT_MDN").set_value(0)
    run(at)
    at.radio(key="k_step").set_value("3 · Matches")
    run(at)
    assert "changed your choices" in all_text(at)
    assert len(fake_ai.calls) == 3

    # Reset restores the AI's weights
    at.radio(key="k_step").set_value("2 · Review")
    run(at)
    run(button(at, "↺ Reset AI suggestions").click())
    assert at.slider(key="k_w_DEBT_MDN").value == 6


def test_ai_failure_falls_back(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def boom(*_args, **_kwargs):
        raise backend.AIServiceError("quota", "The AI service quota or rate limit was reached.")

    monkeypatch.setattr(backend, "interpret_and_audit", boom)
    at = run(AppTest.from_file(APP))
    at.text_area(key="k_prompt").set_value("cheap school")
    run(button(at, "🔎 Find my matches").click())
    assert "quota" in all_text(at)
    run(button(at, "Set my priorities by hand").click())
    assert at.radio(key="k_step").value.startswith("2")


def test_key_never_rendered(fake_ai):
    at = run(AppTest.from_file(APP))
    assert "test-key-not-real" not in all_text(at)
