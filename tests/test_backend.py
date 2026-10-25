"""Deterministic backend tests (no network, no API key).

Tests 1–8 mirror the notebook's "RANKMYCOLLEGE BACKEND TESTS" cell.
The rest cover the README section 7 checks and the frontend adapter.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend_v3 as backend  # noqa: E402

FOUR = {"average_net_price": 5, "DEBT_MDN": 4, "MD_EARN_WNE_P10": 4, "graduation_rate_150pct": 3}
TWO = {"average_net_price": 5, "graduation_rate_150pct": 5}


@pytest.fixture(scope="module")
def ctx():
    return backend.load_context()


# ---------- notebook tests 1–8 ----------

def test_1_all_colleges_accounted_for(ctx):
    ranked, incomplete = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    assert len(ranked) + len(incomplete) == len(ctx.reference_pool)


def test_2_no_duplicates(ctx):
    ranked, incomplete = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    assert ranked["unitid"].is_unique and incomplete["unitid"].is_unique
    assert set(ranked["unitid"]).isdisjoint(incomplete["unitid"])


def test_3_scores_in_range(ctx):
    ranked, _ = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    assert ranked["final_score"].notna().all()
    assert ranked["final_score"].between(0, 100).all()


def test_4_contributions_sum_to_score(ctx):
    ranked, _ = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    cols = [f"{m}_contribution" for m in FOUR]
    np.testing.assert_allclose(ranked[cols].sum(axis=1), ranked["final_score"], atol=1e-8)


def test_5_scaling_weights_is_harmless(ctx):
    ranked, _ = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    scaled, _ = backend.rank_colleges_v2(
        ctx.reference_pool, {m: w * 10 for m, w in FOUR.items()}, ctx.rankable_metrics
    )
    assert ranked["unitid"].tolist() == scaled["unitid"].tolist()
    np.testing.assert_allclose(ranked["final_score"], scaled["final_score"], atol=1e-8)


def test_6_fewer_metrics_more_coverage(ctx):
    four, _ = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    two, _ = backend.rank_colleges_v2(ctx.reference_pool, TWO, ctx.rankable_metrics)
    assert set(four["unitid"]).issubset(two["unitid"])


@pytest.mark.parametrize("bad", [
    {"average_net_price": -5},
    {"average_net_price": float("nan")},
    {"made_up_metric": 5},
    {"average_net_price": 0},
    {"average_net_price": True},
    {"acceptance_rate": 5},  # "preference" direction, not rankable
])
def test_7_invalid_weights_rejected(ctx, bad):
    with pytest.raises(ValueError):
        backend.rank_colleges_v2(ctx.reference_pool, bad, ctx.rankable_metrics)


def test_8_reproducible(ctx):
    a, _ = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    b, _ = backend.rank_colleges_v2(ctx.reference_pool, FOUR, ctx.rankable_metrics)
    pd.testing.assert_frame_equal(a[["unitid", "rank", "final_score"]], b[["unitid", "rank", "final_score"]])


# ---------- data and historical counts ----------

def test_historical_counts(ctx):
    assert len(ctx.colleges) == 2796
    assert len(ctx.reference_pool) == 2490
    assert backend.calculate_rankings(ctx, FOUR)["counts"]["reference_scored"] == 1988
    assert backend.calculate_rankings(ctx, TWO)["counts"]["reference_scored"] == 2217
    cs = backend.calculate_rankings(ctx, FOUR, {"11.0701"})
    assert cs["counts"]["major_scored"] == 702


def test_missing_file_reported(tmp_path):
    assert set(backend.missing_data_files(tmp_path)) == set(backend.REQUIRED_FILES)
    with pytest.raises(FileNotFoundError):
        backend.load_context(tmp_path)


def test_import_has_no_side_effects():
    out = subprocess.run(
        [sys.executable, "-c", "import backend_v3"],
        cwd=ROOT, capture_output=True, text=True, env={"PATH": ""},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""


# ---------- README section 7 ----------

def test_counts_add_up_before_filters(ctx):
    r = backend.calculate_rankings(ctx, FOUR, {"30.7001"}, {"states": ["CA"]})
    c = r["counts"]
    assert c["reference_scored"] + c["reference_incomplete"] == c["reference_total"]


def test_filters_do_not_change_scores(ctx):
    full = backend.calculate_rankings(ctx, FOUR)["ranked"].set_index("unitid")["final_score"]
    filtered = backend.calculate_rankings(
        ctx, FOUR, {"11.0701"},
        {"states": ["MA", "NY"], "institution_types": ["Public 4-year"], "min_enrollment": 1000},
    )
    f = filtered["ranked"]
    assert len(f) > 0
    assert f["state"].isin(["MA", "NY"]).all()
    assert (f["institution_type"] == "Public 4-year").all()
    assert (f["undergraduate_enrollment"] >= 1000).all()
    np.testing.assert_allclose(f.set_index("unitid")["final_score"], full.loc[f["unitid"]], atol=1e-10)
    assert f["rank"].tolist() == list(range(1, len(f) + 1))
    assert f["overall_rank"].is_monotonic_increasing


def test_major_filter_uses_reported_completions(ctx):
    r = backend.calculate_rankings(ctx, FOUR, {"30.7001", "30.7099"})
    ids = set(ctx.reported_programs.loc[
        ctx.reported_programs["cipcode"].isin({"30.7001", "30.7099"}), "unitid"
    ])
    assert set(r["ranked"]["unitid"]).issubset(ids)
    assert (r["ranked"]["matching_major_awards_2024"] > 0).all()


def test_changing_weights_changes_order(ctx):
    a = backend.calculate_rankings(ctx, FOUR)["ranked"]["unitid"].head(10).tolist()
    b = backend.calculate_rankings(ctx, {**FOUR, "MD_EARN_WNE_P10": 40})["ranked"]["unitid"].head(10).tolist()
    assert a != b


def test_empty_filter_result_does_not_crash(ctx):
    r = backend.calculate_rankings(ctx, FOUR, None, {"max_average_net_price": 1})
    assert r["ranked"].empty
    assert r["counts"]["shown_scored"] == 0


def test_ambiguous_major_gives_suggestions_not_a_guess(ctx):
    with pytest.raises(backend.MajorNotFoundError) as exc:
        backend.resolve_major("Nursing", ctx.major_catalog)
    assert "Registered Nursing/Registered Nurse" in exc.value.suggestions["major_name"].tolist()


def test_data_science_resolves_to_official_codes(ctx):
    codes = set(backend.resolve_major("Data Science", ctx.major_catalog)["cipcode"])
    assert codes == {"30.7001", "30.7099"}


def test_breakdown_sums_to_score(ctx):
    r = backend.calculate_rankings(ctx, FOUR)
    row = r["ranked"].iloc[0]
    assert backend.score_breakdown(row, r["weights"])["points"].sum() == pytest.approx(row["final_score"])
    assert len(backend.top_reasons(row, r["weights"])) == 2


# ---------- adapter with a fake Groq client ----------

class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, model, messages, response_format, temperature):
        assert response_format == {"type": "json_object"}
        assert "JSON Schema" in messages[0]["content"]
        self.calls.append(json.loads(messages[1]["content"]))
        content = json.dumps(self.responses.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_client(*responses):
    completions = FakeCompletions(responses)
    # .models mirrors .chat.completions so tests can read .calls
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), models=completions)


PLAN = {
    "choices": [
        {"metric": "average_net_price", "weight": 5, "explanation": "aid"},
        {"metric": "MD_EARN_WNE_P10", "weight": 3, "explanation": "jobs"},
    ],
    "unsupported": ["Program quality", "Employer reputation"],
    "cautions": ["Net price is not personalized aid."],
    "requested_major": "Data Science",
}


def test_interpret_and_audit_validates_and_resolves(ctx):
    bad_audit = {**PLAN, "choices": PLAN["choices"] + [
        {"metric": "prestige_score", "weight": 9, "explanation": "made up"},
        {"metric": "average_net_price", "weight": 2, "explanation": "dup"},
    ]}
    client = fake_client(PLAN, bad_audit)
    plan = backend.interpret_and_audit(client, ctx, "Data Science with great aid and prestige", "test-model")

    assert len(client.models.calls) == 2
    assert set(plan["ai_weights"]) == {"average_net_price", "MD_EARN_WNE_P10"}
    assert sum(plan["ai_weights"].values()) == pytest.approx(1)
    assert len(plan["dropped"]) == 2
    assert plan["unsupported"] == ["Program quality", "Employer reputation"]
    assert {m["cipcode"] for m in plan["major_matches"]} == {"30.7001", "30.7099"}
    json.dumps(plan)  # must be plain data for session state


def test_interpret_ambiguous_major_returns_suggestions(ctx):
    nursing = {**PLAN, "requested_major": "Nursing"}
    plan = backend.interpret_and_audit(fake_client(nursing, nursing), ctx, "nursing please")
    assert plan["major_matches"] == []
    assert plan["major_suggestions"]


def test_interpret_rejects_empty_prompt(ctx):
    with pytest.raises(ValueError):
        backend.interpret_and_audit(fake_client(), ctx, "   ")


def test_explainer_gets_only_verified_evidence(ctx):
    report = {
        "original_winner_explanation": "a", "custom_winner_explanation": "b",
        "why_rankings_changed": "c", "key_tradeoffs": ["d"],
    }
    client = fake_client(report)
    ai = backend.calculate_rankings(ctx, FOUR, {"30.7001"})
    custom = backend.calculate_rankings(ctx, {**FOUR, "MD_EARN_WNE_P10": 20}, {"30.7001"})
    out = backend.explain_verified_results(
        client, ctx, "req", ai, custom, FOUR, custom["weights"],
        "Data Science", ["Program quality"],
    )
    assert out.key_tradeoffs == ["d"]
    evidence = client.models.calls[0]
    assert evidence["unsupported_priorities"] == ["Program quality"]
    assert evidence["requested_major"] == "Data Science"
    names = set(ai["ranked"]["college_name"].head(3)) | set(custom["ranked"]["college_name"].head(3))
    assert {c["college_name"] for c in evidence["colleges"]} == names


def test_no_key_is_a_clear_error():
    with pytest.raises(backend.AIServiceError) as exc:
        backend.make_client("")
    assert exc.value.kind == "no_key"


@pytest.mark.parametrize("code,kind", [(404, "model_not_found"), (429, "quota"), (503, "unavailable"), (403, "bad_key")])
def test_api_errors_are_classified(code, kind, monkeypatch):
    import groq
    import httpx

    class Failing:
        def create(self, **_):
            response = httpx.Response(code, request=httpx.Request("POST", "https://api.groq.com"))
            raise groq.APIStatusError("x", response=response, body=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Failing()))
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    with pytest.raises(backend.AIServiceError) as exc:
        backend.preference_agent_v3(client, "hi", {}, "m")
    assert exc.value.kind == kind


def test_malformed_reply_is_a_clear_error(ctx):
    with pytest.raises(backend.AIServiceError) as exc:
        backend.interpret_and_audit(fake_client({"not": "a plan"}), ctx, "cheap school")
    assert exc.value.kind == "bad_response"
