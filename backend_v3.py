"""RankMyCollege V3 backend.

Ported from the latest Colab notebook (RankMyCollege_AI_Agents.ipynb).
The last tested definition of each function was kept; test-only print
loops and sample prompts were left out so importing this module never
calls the AI service or prints anything.

Changes from the notebook:
- Notebook globals (rankable_metrics, reference_pool, reported_programs,
  major_catalog) live in a BackendContext built by load_context().
- The Groq client (replacing the notebook's Gemini client) is created
  from an API key passed in explicitly
  (no google.colab.userdata).
- A small adapter (interpret_and_audit, calculate_rankings,
  explain_verified_results) separates the AI calls from the
  deterministic scoring so sliders never trigger the AI.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_PROMPT_CHARS = 1000

REQUIRED_FILES = (
    "colleges_plus_scorecard.csv",
    "metrics.json",
    "college_programs_named.csv",
)

PROGRAM_EVIDENCE_NOTE = (
    "Reported 2024 bachelor's completions by CIP code; "
    "not a measure of program quality or proof that "
    "other colleges do not offer the major."
)

# Plain-language labels for students. "fmt" controls value display.
METRIC_LABELS = {
    "graduation_rate_150pct": {
        "label": "Graduation rate",
        "short": "Graduation rate",
        "help": "Share of students who finish within 150% of normal time (about 6 years for a 4-year degree).",
        "fmt": "pct",
    },
    "bachelors_4yr_completion_rate": {
        "label": "On-time graduation (4 years)",
        "short": "4-year grad rate",
        "help": "Share of students who finish a bachelor's degree within four years.",
        "fmt": "pct",
    },
    "retention_rate": {
        "label": "Students who come back for year 2",
        "short": "Retention",
        "help": "First-year students who return the next fall. It is not a direct measure of happiness.",
        "fmt": "pct",
    },
    "student_faculty_ratio": {
        "label": "Students per faculty member",
        "short": "Student:faculty",
        "help": "Fewer students per faculty member may mean more access to professors.",
        "fmt": "ratio",
    },
    "average_net_price": {
        "label": "Average net price (after grants)",
        "short": "Avg net price",
        "help": "A college-wide average yearly cost after grant aid. Your own price may be very different.",
        "fmt": "money",
    },
    "MD_EARN_WNE_P10": {
        "label": "Reported earnings 10 years after entering college",
        "short": "Earnings (10 yrs after entry)",
        "help": "Median earnings of former students 10 years after they STARTED college, across all majors. Not a salary prediction.",
        "fmt": "money",
    },
    "MD_EARN_WNE_P6": {
        "label": "Reported earnings 6 years after entering college",
        "short": "Earnings (6 yrs after entry)",
        "help": "Median earnings of former students 6 years after they STARTED college, across all majors. Not a salary prediction.",
        "fmt": "money",
    },
    "DEBT_MDN": {
        "label": "Median student-loan debt",
        "short": "Median debt",
        "help": "Median federal loan debt among students who borrowed. It does not predict your own debt.",
        "fmt": "money",
    },
    "GRAD_DEBT_MDN": {
        "label": "Median debt of graduates who borrowed",
        "short": "Graduate debt",
        "help": "Median federal loan debt among graduating borrowers. It does not predict your own debt.",
        "fmt": "money",
    },
}


def metric_label(metric, short=False):
    info = METRIC_LABELS.get(metric)
    if info is None:
        return metric
    return info["short"] if short else info["label"]


def format_metric_value(metric, value):
    if value is None or pd.isna(value):
        return "Not reported"
    fmt = METRIC_LABELS.get(metric, {}).get("fmt")
    value = float(value)
    if fmt == "money":
        return f"${value:,.0f}"
    if fmt == "pct":
        return f"{value:.0f}%"
    if fmt == "ratio":
        return f"{value:.0f} : 1"
    return f"{value:,.2f}"


# ==========================================
# OUTPUT DATA CLASSES
# ==========================================

class MetricChoice(BaseModel):
    metric: str
    weight: float
    explanation: str


class MajorAwarePlan(BaseModel):
    choices: list[MetricChoice]
    unsupported: list[str]
    cautions: list[str]
    requested_major: str | None = None


class ExplanationReport(BaseModel):
    original_winner_explanation: str
    custom_winner_explanation: str
    why_rankings_changed: str
    key_tradeoffs: list[str]


# ==========================================
# DATA LOADING (replaces notebook globals)
# ==========================================

@dataclass
class BackendContext:
    colleges: pd.DataFrame
    metrics: dict
    rankable_metrics: dict
    reference_pool: pd.DataFrame
    reported_programs: pd.DataFrame
    major_catalog: pd.DataFrame


def missing_data_files(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    return [name for name in REQUIRED_FILES if not (data_dir / name).exists()]


def prepare_reference_pool(colleges, catalog):
    df = colleges.copy()

    df["undergraduate_enrollment"] = pd.to_numeric(
        df["undergraduate_enrollment"],
        errors="coerce"
    )

    # Keep colleges with reported undergraduates
    df = df[
        df["undergraduate_enrollment"] > 0
    ].copy()

    # Convert ranking metrics into numbers
    for metric in catalog:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

    # Remove invalid negative net prices
    if "average_net_price" in df.columns:
        df.loc[
            df["average_net_price"] < 0,
            "average_net_price"
        ] = np.nan

    return df.reset_index(drop=True)


def build_major_catalog(reported_programs):
    return (
        reported_programs
        .dropna(subset=["major_name"])
        .groupby(["cipcode", "major_name"], as_index=False)
        .agg(
            colleges_reporting=("unitid", "nunique"),
            bachelors_awarded=("bachelors_awarded_2024", "sum")
        )
    )


def load_context(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    missing = missing_data_files(data_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing required data file(s) in {data_dir}: {', '.join(missing)}"
        )

    with open(data_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    colleges = pd.read_csv(data_dir / "colleges_plus_scorecard.csv")

    rankable_metrics = {
        name: info
        for name, info in metrics.items()
        if info.get("direction") in ("higher", "lower")
        and name in colleges.columns
    }

    reference_pool = prepare_reference_pool(colleges, rankable_metrics)

    programs_named = pd.read_csv(
        data_dir / "college_programs_named.csv",
        dtype={"cipcode": "string"}
    )
    programs_named["bachelors_awarded_2024"] = pd.to_numeric(
        programs_named["bachelors_awarded_2024"],
        errors="coerce"
    )

    # Keep programs with reported bachelor's graduates
    reported_programs = programs_named[
        programs_named["bachelors_awarded_2024"] > 0
    ].copy()

    return BackendContext(
        colleges=colleges,
        metrics=metrics,
        rankable_metrics=rankable_metrics,
        reference_pool=reference_pool,
        reported_programs=reported_programs,
        major_catalog=build_major_catalog(reported_programs),
    )


# ==========================================
# VALIDATION AND MAJOR LOOKUP
# ==========================================

def validate_weights(plan, rankable_metrics):
    weights = {}

    for choice in plan.choices:
        metric = choice.metric
        weight = choice.weight

        if metric not in rankable_metrics:
            raise ValueError(f"Unknown metric: {metric}")

        if metric in weights:
            raise ValueError(f"Duplicate metric: {metric}")

        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid weight: {weight}")

        weights[metric] = weight

    if not weights:
        raise ValueError("No valid metrics selected.")

    total = sum(weights.values())

    return {
        metric: weight / total
        for metric, weight in weights.items()
    }


def sanitize_plan(plan, rankable_metrics, max_metrics=6):
    """Drop AI metric choices that would fail validate_weights.

    Returns (clean_plan, dropped) where dropped lists human-readable
    reasons. The model never gets to score an unknown metric.
    """
    kept = []
    seen = set()
    dropped = []

    for choice in plan.choices:
        if choice.metric not in rankable_metrics:
            dropped.append(f"'{choice.metric}' is not a metric in our data")
        elif choice.metric in seen:
            dropped.append(f"'{choice.metric}' was listed twice")
        elif not math.isfinite(choice.weight) or choice.weight <= 0:
            dropped.append(f"'{choice.metric}' had an invalid weight")
        elif len(kept) >= max_metrics:
            dropped.append(f"'{choice.metric}' exceeded the {max_metrics}-metric limit")
        else:
            kept.append(choice)
            seen.add(choice.metric)

    return plan.model_copy(update={"choices": kept}), dropped


def search_majors(keyword, major_catalog, limit=20):
    matches = major_catalog[
        major_catalog["major_name"].str.contains(
            keyword,
            case=False,
            regex=False,
            na=False
        )
    ]

    return matches.sort_values(
        "colleges_reporting",
        ascending=False
    ).head(limit).reset_index(drop=True)


class MajorNotFoundError(ValueError):
    def __init__(self, major_name, suggestions):
        self.major_name = major_name
        self.suggestions = suggestions
        names = suggestions["major_name"].tolist()[:5]
        super().__init__(
            f"No exact match for '{major_name}'. Possible matches: {names}"
        )


def resolve_major(major_name, major_catalog):
    if not major_name:
        return pd.DataFrame(
            columns=[
                "cipcode",
                "major_name",
                "colleges_reporting"
            ]
        )

    name = major_name.strip().casefold()

    # Exact name and its General/Other classifications
    accepted_names = {
        name,
        f"{name}, general",
        f"{name}, other"
    }

    matches = major_catalog[
        major_catalog["major_name"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .isin(accepted_names)
    ].copy()

    if matches.empty:
        suggestions = search_majors(major_name, major_catalog)
        # Try each word too, e.g. "nursing degree" -> "nursing"
        if suggestions.empty:
            for word in major_name.split():
                if len(word) >= 4:
                    suggestions = search_majors(word, major_catalog)
                    if not suggestions.empty:
                        break
        raise MajorNotFoundError(major_name, suggestions)

    return matches.reset_index(drop=True)


# ==========================================
# DETERMINISTIC RANKING ENGINE V2
# ==========================================

def rank_colleges_v2(pool, selected_weights, catalog):

    # 1. Validate selected metrics and weights
    active = {}

    for metric, raw_weight in selected_weights.items():
        if metric not in catalog:
            raise ValueError(f"Unknown metric: {metric}")

        if metric not in pool.columns:
            raise ValueError(f"Missing column: {metric}")

        direction = catalog[metric]["direction"]

        if direction not in ("higher", "lower"):
            raise ValueError(f"Unsupported direction: {metric}")

        if isinstance(raw_weight, bool):
            raise ValueError(f"Invalid weight: {metric}")

        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid weight: {metric}")

        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid weight: {metric}")

        if weight > 0:
            active[metric] = weight

    if not active:
        raise ValueError("Select at least one metric.")

    # 2. Normalize weights
    total = sum(active.values())

    normalized = {
        metric: weight / total
        for metric, weight in active.items()
    }

    df = pool.copy()

    # 3. Identify schools missing selected metrics
    missing = df[list(active)].isna()

    complete_mask = ~missing.any(axis=1)

    ranked = df.loc[complete_mask].copy()
    incomplete = df.loc[~complete_mask].copy()

    if ranked.empty:
        raise ValueError(
            "No colleges have all selected metrics."
        )

    # 4. Calculate scores using the reference pool
    ranked["final_score"] = 0.0

    for metric, weight in normalized.items():

        direction = catalog[metric]["direction"]

        # Reference percentiles include every school
        # with valid data for this particular metric.
        reference_percentiles = df[metric].rank(
            pct=True,
            ascending=(direction == "higher")
        ) * 100

        ranked[f"{metric}_percentile"] = (
            reference_percentiles.loc[ranked.index]
        )

        contribution = (
            ranked[f"{metric}_percentile"] * weight
        )

        ranked[f"{metric}_contribution"] = contribution
        ranked["final_score"] += contribution

    # 5. Sort ranked colleges
    ranked = ranked.sort_values(
        ["final_score", "college_name"],
        ascending=[False, True]
    ).reset_index(drop=True)

    ranked["rank"] = range(1, len(ranked) + 1)

    # 6. Track incomplete college information
    incomplete["missing_metrics"] = missing.loc[
        ~complete_mask
    ].apply(
        lambda row: [
            metric for metric in active
            if row[metric]
        ],
        axis=1
    ) if (~complete_mask).any() else pd.Series(dtype=object)

    coverage = pd.Series(0.0, index=df.index)

    for metric, weight in normalized.items():
        coverage += (
            df[metric].notna().astype(float) * weight
        )

    incomplete["data_coverage_pct"] = (
        coverage.loc[~complete_mask] * 100
    )

    incomplete = incomplete.sort_values(
        ["data_coverage_pct", "college_name"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return ranked, incomplete


def filter_rankings_by_major(ranked, incomplete, cipcodes, reported_programs):

    # Find programs with reported bachelor's awards
    matches = reported_programs[
        reported_programs["cipcode"].isin(cipcodes)
        & (reported_programs["bachelors_awarded_2024"] > 0)
    ].copy()

    # Aggregate if one university matches multiple CIP codes
    major_counts = (
        matches.groupby("unitid", as_index=False)
        ["bachelors_awarded_2024"].sum()
        .rename(columns={
            "bachelors_awarded_2024":
            "matching_major_awards_2024"
        })
    )

    matching_ids = set(major_counts["unitid"])

    # Preserve scores calculated against all colleges
    major_ranked = ranked[
        ranked["unitid"].isin(matching_ids)
    ].copy()

    major_incomplete = incomplete[
        incomplete["unitid"].isin(matching_ids)
    ].copy()

    # Preserve the original overall rank
    major_ranked = major_ranked.rename(
        columns={"rank": "overall_rank"}
    ).reset_index(drop=True)

    # Add rank within the selected major
    major_ranked.insert(
        0, "rank", range(1, len(major_ranked) + 1)
    )

    # Attach program evidence without changing scores
    major_ranked = major_ranked.merge(
        major_counts,
        on="unitid",
        how="left",
        validate="one_to_one",
        sort=False
    )

    return major_ranked, major_incomplete


def filter_ranked_colleges(
    rankings,
    states=None,
    institution_types=None,
    min_enrollment=None,
    max_average_net_price=None
):
    """
    Filter already-scored colleges.

    Scores stay unchanged.
    Displayed ranks update within the filtered results.
    Expects an "overall_rank" column to already exist.
    """

    df = rankings.copy()

    if states:
        df = df[df["state"].isin(states)]

    if institution_types:
        df = df[
            df["institution_type"].isin(institution_types)
        ]

    if min_enrollment is not None:
        df = df[
            df["undergraduate_enrollment"] >= min_enrollment
        ]

    # Optional average net price cutoff
    if max_average_net_price is not None:
        df = df[
            df["average_net_price"] <= max_average_net_price
        ]

    df = df.reset_index(drop=True)

    if "rank" in df.columns:
        df["rank"] = range(1, len(df) + 1)
    else:
        df.insert(0, "rank", range(1, len(df) + 1))

    return df


# ==========================================
# AGENT 3 EVIDENCE
# ==========================================

def prepare_explainer_evidence_v2(
    request, original, custom,
    original_weights, custom_weights, catalog
):
    original_lookup = original.set_index("unitid")
    custom_lookup = custom.set_index("unitid")

    # Gather the top 3 from each scenario
    college_ids = list(dict.fromkeys(
        original["unitid"].head(3).tolist()
        + custom["unitid"].head(3).tolist()
    ))

    metrics = list(dict.fromkeys(
        list(original_weights) + list(custom_weights)
    ))

    def normalize(w):
        active = {k: float(v) for k, v in w.items() if v > 0}
        total = sum(active.values())
        return {k: v / total for k, v in active.items()}

    def safe_number(value):
        if value is None or pd.isna(value):
            return None
        return float(value)

    college_evidence = []

    for unitid in college_ids:
        a = (
            original_lookup.loc[unitid]
            if unitid in original_lookup.index
            else None
        )

        b = (
            custom_lookup.loc[unitid]
            if unitid in custom_lookup.index
            else None
        )

        source = a if a is not None else b

        metric_evidence = {}

        for metric in metrics:
            metric_evidence[metric] = {
                "actual_value": safe_number(source[metric]),
                "direction": catalog[metric]["direction"],
                "ai_contribution": (
                    safe_number(a.get(f"{metric}_contribution"))
                    if a is not None else None
                ),
                "custom_contribution": (
                    safe_number(b.get(f"{metric}_contribution"))
                    if b is not None else None
                )
            }

        college_evidence.append({
            "college_name": str(source["college_name"]),
            "ai_rank": int(a["rank"]) if a is not None else None,
            "custom_rank": int(b["rank"]) if b is not None else None,
            "ai_score": safe_number(a["final_score"]) if a is not None else None,
            "custom_score": safe_number(b["final_score"]) if b is not None else None,
            "metrics": metric_evidence
        })

    return {
        "student_request": request,
        "ai_weights": normalize(original_weights),
        "custom_weights": normalize(custom_weights),
        "original_ranked_count": len(original),
        "custom_ranked_count": len(custom),
        "colleges": college_evidence,
        "comparison_rule": (
            "A null rank means the college was not fully scored "
            "in that scenario. Do not invent missing ranks, "
            "scores, metric values, or comparisons."
        )
    }


# ==========================================
# GROQ CLIENT
# ==========================================

class AIServiceError(RuntimeError):
    """An AI call failed; `kind` is a short machine-readable reason."""

    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


def get_model_name():
    return os.environ.get("GROQ_MODEL") or DEFAULT_MODEL


def make_client(api_key):
    if not api_key:
        raise AIServiceError("no_key", "No Groq API key is configured.")
    from groq import Groq
    return Groq(api_key=api_key, max_retries=0)


def list_available_models(client):
    return [m.id for m in client.models.list().data]


def _classify_api_error(exc, model):
    import groq

    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return AIServiceError(
            "unavailable",
            "We couldn't reach the AI service. Check your connection and try again."
        )

    code = getattr(exc, "status_code", None)
    if code == 404 or "model_not_found" in str(exc) or "decommissioned" in str(exc):
        return AIServiceError(
            "model_not_found",
            f"The AI model '{model}' isn't available for this API key. "
            "Set GROQ_MODEL to a model listed at console.groq.com/docs/models."
        )
    if code == 429:
        return AIServiceError(
            "quota",
            "The AI service quota or rate limit was reached. Try again in a minute."
        )
    if code in (401, 403):
        return AIServiceError("bad_key", "The Groq API key was rejected.")
    if code is not None and code >= 500:
        return AIServiceError(
            "unavailable",
            "The AI service is temporarily overloaded. Try again shortly."
        )
    return AIServiceError("api_error", f"The AI service returned an error ({code}).")


def _schema_instructions(instructions, schema):
    # Groq JSON mode guarantees valid JSON but not our shape, so spell it
    # out here and validate the reply with pydantic below.
    return (
        instructions
        + "\n\nRespond with ONLY a JSON object matching this JSON Schema:\n"
        + json.dumps(schema.model_json_schema())
    )


def _generate_structured(client, model, instructions, payload, schema, retries=1):
    import groq

    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _schema_instructions(instructions, schema)},
                    {"role": "user", "content": json.dumps(payload, allow_nan=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            break
        except groq.APIError as exc:
            classified = _classify_api_error(exc, model)
            # Only retry temporary overloads, and only briefly
            if classified.kind == "unavailable" and attempt < retries:
                attempt += 1
                time.sleep(2 * attempt)
                continue
            raise classified from exc

    try:
        return schema.model_validate_json(response.choices[0].message.content)
    except (ValidationError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise AIServiceError(
            "bad_response",
            "The AI returned an answer in an unexpected format. Please try again."
        ) from exc


# ==========================================
# AGENT 1 — PREFERENCE INTERPRETER V3
# ==========================================

PREFERENCE_INSTRUCTIONS_V3 = """
    You are RankMyCollege's Preference Interpreter.

    Your audience is high school seniors.

    Your responsibilities:
    1. Understand the student's college preferences.
    2. Identify their desired college major, if mentioned.
    3. Select relevant available ranking metrics.
    4. Assign positive importance weights.
    5. Identify requirements that cannot be measured.

    MAJOR IDENTIFICATION:
    - If the student names a major, extract its name.
    - Examples: Computer Science, Nursing,
      Psychology, Data Science, Biology.
    - Return a short, standard major name.
    - Do not invent a major if none was mentioned.
    - Do not invent a university's program availability.
    - The major is a program filter, NOT a numerical
      ranking metric.

    METRIC RULES:
    - Only choose metrics from available_metrics.
    - Choose at most 6 relevant metrics.
    - Use positive importance weights.
    - Explain each selected metric.
    - Never invent statistics.
    - Do not equate acceptance rate with prestige.
    - University-wide earnings do not measure the
      quality of a specific academic department.
    - Retention rates are not direct evidence
      of student satisfaction.
    - Mark unmeasurable program quality,
      prestige, or employment reputation as
      unsupported where appropriate.
    - Average net price is not personalized aid.

    Return the MajorAwarePlan structure.
    """


def preference_agent_v3(client, student_request, rankable_metrics, model=None):
    return _generate_structured(
        client,
        model or get_model_name(),
        PREFERENCE_INSTRUCTIONS_V3,
        {
            "student_request": student_request,
            "available_metrics": rankable_metrics
        },
        MajorAwarePlan,
    )


# ==========================================
# AGENT 2 — PREFERENCE AUDITOR V3
# ==========================================

AUDITOR_INSTRUCTIONS_V3 = """
    You are RankMyCollege's Preference Auditor.

    Review Agent 1's interpretation of a high
    school student's college preferences.

    RESPONSIBILITIES:
    1. Verify the requested academic major.
    2. Correct a misidentified major when necessary.
    3. Review selected metrics and importance weights.
    4. Remove irrelevant or unsupported metrics.
    5. Identify criteria the dataset cannot evaluate.
    6. Preserve the student's actual priorities.

    MAJOR RULES:
    - Only identify majors explicitly requested.
    - Keep the major separate from numerical metrics.
    - Do not invent college program availability.
    - Do not claim degree counts measure program quality.
    - Do not substitute a loosely related major.
    - Do not invent CIP codes.

    RANKING RULES:
    - Only select metrics from available_metrics.
    - All selected weights must be positive.
    - Select at most 6 relevant metrics.
    - Institution-wide earnings do not measure
      major-specific employment outcomes.
    - Acceptance rate is not a prestige score.
    - Net price is not personalized financial aid.
    - Remove metrics that lack a clear connection
      to the student's request.

    Return a corrected MajorAwarePlan.
    """


def auditor_agent_v3(client, student_request, first_plan, rankable_metrics, model=None):
    return _generate_structured(
        client,
        model or get_model_name(),
        AUDITOR_INSTRUCTIONS_V3,
        {
            "student_request": student_request,
            "initial_plan": first_plan.model_dump(),
            "available_metrics": rankable_metrics
        },
        MajorAwarePlan,
    )


# ==========================================
# AGENT 3 — RECOMMENDATION EXPLAINER V3
# ==========================================

EXPLAINER_INSTRUCTIONS_V3 = """
    You are Agent 3, RankMyCollege's
    Recommendation Explainer.

    Your audience is high school seniors,
    typically ages 17–18.

    Your job is to explain personalized
    college rankings in a short, clear,
    accurate, and engaging way.

    =====================================
    YOUR RESPONSIBILITIES
    =====================================

    1. Explain why the original #1 college
       ranked first.

    2. Explain why the custom #1 college
       ranked first after changing weights.

    3. Explain how changing priorities
       affected the rankings.

    4. Identify meaningful tradeoffs
       between the ranked universities.

    5. Use the student's requested major
       as context when one is provided.

    =====================================
    WRITING STYLE
    =====================================

    - Write for high school seniors.
    - Sound like a helpful older student.
    - Be friendly, natural, and direct.
    - Avoid academic jargon.
    - Avoid corporate or marketing language.
    - Do not exaggerate.
    - Do not use unnecessary filler.
    - Use "you" and "your" naturally.
    - Always name the actual universities.
    - Never write generic phrases like
      "This school is a good choice."

    LENGTH REQUIREMENTS:

    - Original explanation: maximum 35 words.
    - Custom explanation: maximum 35 words.
    - Ranking changes: maximum 30 words.
    - Give up to 2 meaningful tradeoffs.
    - Each tradeoff: maximum 25 words.
    - Do not repeat the same point unnecessarily.

    AVOID PHRASES LIKE:

    - "Incredible value"
    - "Fantastic balance"
    - "Exceptional institution"
    - "Prestigious establishment"
    - "Soared past competitors"
    - "Post-entry earnings"

    =====================================
    MAJOR-AWARE EXPLANATIONS
    =====================================

    The student may have requested a
    specific academic major.

    If requested_major is provided:

    - Understand that the candidate colleges
      were filtered using reported bachelor's
      completions in matching CIP programs.

    - Reference the selected major naturally
      when useful.

    - Never claim a university has the
      best program in that major.

    - Never claim the program is excellent
      without direct supporting evidence.

    - Never treat bachelor's completion
      counts as a measure of program quality.

    - Never assume that universities with
      no reported completions lack the major.

    - Do not claim university-wide earnings
      represent earnings for that specific major.

    - Do not claim employer reputation,
      teaching quality, curriculum strength,
      or program rankings were measured
      unless explicitly supported.

    - If the student's main priority is
      unsupported, do not imply that the
      recommendation satisfies that priority.

    =====================================
    FINANCIAL AND CAREER ACCURACY
    =====================================

    - Average net price is an institutional
      average, not the student's personal cost.

    - Median student debt does not predict
      an individual student's future debt.

    - MD_EARN_WNE_P10 means reported earnings
      10 years after entering college.

    - MD_EARN_WNE_P6 means reported earnings
      6 years after entering college.

    - Never describe either earnings measure
      as graduate earnings.

    - Never promise employment, earnings,
      financial aid, or admission.

    - Do not claim that higher reported
      earnings guarantee a better career.

    =====================================
    EVIDENCE AND RANKING RULES
    =====================================

    - Use ONLY the evidence provided.

    - The original winner must have
      ai_rank equal to 1.

    - The custom winner must have
      custom_rank equal to 1.

    - Never invent colleges, scores,
      rankings, or statistics.

    - Never change ranking results.

    - Never calculate new ranking scores.

    - Only compare actual metric values
      when both values are supplied.

    - A null value means unavailable.
      Never guess the missing value.

    - A null rank means that university
      was not fully scored in that scenario.

    - Ranking scores indicate performance
      under the selected weights, not
      overall college quality.

    - If original and custom winners are
      the same, explain that they stayed
      first rather than inventing a change.

    - When explaining a ranking change,
      focus on the actual weight changes.

    - Mention unsupported requirements
      only when necessary to prevent a
      misleading recommendation.

    - Do not create a separate
      data limitations section.

    =====================================
    OUTPUT REQUIREMENTS
    =====================================

    Return exactly these four fields:

    original_winner_explanation:
    Explain the original top college.

    custom_winner_explanation:
    Explain the customized top college.

    why_rankings_changed:
    Explain the effect of adjusting weights.

    key_tradeoffs:
    List up to 2 concise, evidence-supported
    comparisons involving named colleges.

    Keep everything short, personalized,
    and easy to understand.
    """


def explainer_agent(client, evidence, model=None):
    return _generate_structured(
        client,
        model or get_model_name(),
        EXPLAINER_INSTRUCTIONS_V3,
        evidence,
        ExplanationReport,
    )


# ==========================================
# ORCHESTRATOR (end-to-end smoke test only)
# ==========================================

def run_rankmycollege_v3(ctx, client, student_request, edited_weights=None, model=None):
    """Full three-agent pipeline, as in the notebook.

    Use for smoke tests only. The Streamlit app calls the adapter
    functions below so that sliders never re-run the agents.
    """
    initial = preference_agent_v3(client, student_request, ctx.rankable_metrics, model)
    approved = auditor_agent_v3(client, student_request, initial, ctx.rankable_metrics, model)

    ai_weights = validate_weights(approved, ctx.rankable_metrics)

    selected_weights = (
        edited_weights if edited_weights is not None
        else ai_weights
    )

    # Resolve the requested major using our actual database
    major_name = approved.requested_major
    matched_programs = None
    cipcodes = None

    if major_name:
        matched_programs = resolve_major(major_name, ctx.major_catalog)
        cipcodes = set(matched_programs["cipcode"].astype(str))

    ai = calculate_rankings(ctx, ai_weights, cipcodes)
    custom = calculate_rankings(ctx, selected_weights, cipcodes)

    if ai["ranked"].empty or custom["ranked"].empty:
        raise ValueError(
            "No fully scorable colleges found for "
            f"the selected major: {major_name}"
        )

    report = explain_verified_results(
        client, ctx, student_request, ai, custom,
        ai_weights, selected_weights, major_name,
        approved.unsupported, model
    )

    return {
        "major": major_name,
        "matched_programs": matched_programs,
        "approved_plan": approved,
        "ai_weights": ai_weights,
        "ai_rankings": ai["ranked"],
        "ai_incomplete": ai["incomplete"],
        "custom_rankings": custom["ranked"],
        "custom_incomplete": custom["incomplete"],
        "explanation": report
    }


# ==========================================
# FRONTEND ADAPTER
# ==========================================

def interpret_and_audit(client, ctx, student_prompt, model=None):
    """Run Agents 1 + 2 once for a new prompt and validate the result.

    Returns a plain dict that is safe to keep in st.session_state.
    """
    student_prompt = (student_prompt or "").strip()
    if not student_prompt:
        raise ValueError("Tell us a little about what you want first.")
    if len(student_prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"Please keep your description under {MAX_PROMPT_CHARS} characters."
        )

    initial = preference_agent_v3(client, student_prompt, ctx.rankable_metrics, model)
    approved = auditor_agent_v3(client, student_prompt, initial, ctx.rankable_metrics, model)

    clean, dropped = sanitize_plan(approved, ctx.rankable_metrics)
    if not clean.choices:
        # Auditor removed everything valid; fall back to Agent 1's valid picks
        clean_initial, _ = sanitize_plan(initial, ctx.rankable_metrics)
        clean = clean.model_copy(update={"choices": clean_initial.choices})
    if not clean.choices:
        raise ValueError(
            "We couldn't connect your request to any measurable data. "
            "Try mentioning cost, debt, graduation, or earnings."
        )

    ai_weights = validate_weights(clean, ctx.rankable_metrics)

    major_name = (clean.requested_major or "").strip() or None
    major_matches = pd.DataFrame(columns=["cipcode", "major_name", "colleges_reporting"])
    major_suggestions = pd.DataFrame(columns=["cipcode", "major_name", "colleges_reporting"])
    if major_name:
        try:
            major_matches = resolve_major(major_name, ctx.major_catalog)
        except MajorNotFoundError as exc:
            major_suggestions = exc.suggestions

    return {
        "prompt": student_prompt,
        "initial_plan": initial.model_dump(),
        "approved_plan": clean.model_dump(),
        "ai_weights": ai_weights,
        "explanations": {c.metric: c.explanation for c in clean.choices},
        "unsupported": list(clean.unsupported),
        "cautions": list(clean.cautions),
        "dropped": dropped,
        "requested_major": major_name,
        "major_matches": major_matches[["cipcode", "major_name", "colleges_reporting"]].to_dict("records"),
        "major_suggestions": major_suggestions[["cipcode", "major_name", "colleges_reporting"]].to_dict("records"),
    }


def calculate_rankings(ctx, weights, cipcodes=None, filters=None):
    """Deterministic scoring. Never calls the AI.

    Scores are always computed against the full reference pool; the
    major and location filters only choose which scored colleges are
    shown, so a college's score never changes when filters change.
    """
    ranked, incomplete = rank_colleges_v2(
        ctx.reference_pool, weights, ctx.rankable_metrics
    )

    counts = {
        "reference_total": len(ctx.reference_pool),
        "reference_scored": len(ranked),
        "reference_incomplete": len(incomplete),
    }

    if cipcodes:
        ranked, incomplete = filter_rankings_by_major(
            ranked, incomplete, set(cipcodes), ctx.reported_programs
        )
    else:
        ranked = ranked.copy()
        ranked["overall_rank"] = ranked["rank"]

    counts["major_scored"] = len(ranked)
    counts["major_incomplete"] = len(incomplete)

    filters = filters or {}
    ranked = filter_ranked_colleges(ranked, **filters)
    incomplete = filter_ranked_colleges(incomplete, **filters).drop(columns="rank")

    counts["shown_scored"] = len(ranked)
    counts["shown_incomplete"] = len(incomplete)

    active = {m: float(w) for m, w in weights.items() if float(w) > 0}
    total = sum(active.values())

    return {
        "ranked": ranked,
        "incomplete": incomplete,
        "counts": counts,
        "weights": {m: w / total for m, w in active.items()},
        "cipcodes": sorted(cipcodes) if cipcodes else [],
        "filters": filters,
    }


def explain_verified_results(
    client, ctx, student_prompt, ai_result, custom_result,
    ai_weights, custom_weights, requested_major, unsupported, model=None
):
    """Agent 3 on Python-verified evidence only."""
    evidence = prepare_explainer_evidence_v2(
        student_prompt,
        ai_result["ranked"],
        custom_result["ranked"],
        ai_weights,
        custom_weights,
        ctx.rankable_metrics
    )

    evidence["requested_major"] = requested_major
    evidence["unsupported_priorities"] = list(unsupported or [])
    evidence["program_evidence_type"] = PROGRAM_EVIDENCE_NOTE

    return explainer_agent(client, evidence, model)


# ==========================================
# NON-AI EXPLANATIONS
# ==========================================

def score_breakdown(row, weights):
    """Per-metric contribution table for one scored college."""
    rows = []
    for metric, weight in weights.items():
        rows.append({
            "metric": metric,
            "label": metric_label(metric),
            "value": format_metric_value(metric, row.get(metric)),
            "weight_pct": weight * 100,
            "percentile": float(row.get(f"{metric}_percentile", np.nan)),
            "points": float(row.get(f"{metric}_contribution", 0.0)),
        })
    return pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)


def top_reasons(row, weights, n=2):
    """Plain-language reasons based only on computed contributions."""
    breakdown = score_breakdown(row, weights)
    reasons = []
    for _, item in breakdown.head(n).iterrows():
        top = max(1, round(100 - item["percentile"]))
        reasons.append(
            f"{item['label']}: {item['value']} — in the top {top}% "
            "of colleges that report it"
        )
    return reasons
