"""RankMyCollege — Streamlit frontend.

Flow: 1 Your priorities -> 2 Review your choices -> 3 Your matches -> 4 Compare.
Agents 1+2 run once per prompt, Agent 3 only when the student asks for an
explanation. Sliders, majors, and filters only re-run Python scoring.
"""

import os

import pandas as pd
import streamlit as st

import backend_v3 as backend

st.set_page_config(
    page_title="RankMyCollege",
    page_icon="🎓",
    layout="centered",
)

STEPS = ["1 · Priorities", "2 · Review", "3 · Matches", "4 · Compare"]
SLIDER_MAX = 10

EXAMPLE_PROMPTS = [
    "I want an affordable nursing program with strong job outcomes and low debt.",
    "Affordable, low student debt, and strong graduation rates. Prestige doesn't matter.",
    "I want Data Science with great financial aid and a prestigious employer reputation.",
    "Small classes where I can get to know professors, and most students graduate on time.",
]

# Deterministic presets for data-only mode (not AI suggestions)
PRESETS = {
    "💸 Keep costs low": {"average_net_price": 5, "DEBT_MDN": 4, "graduation_rate_150pct": 2},
    "⚖️ Balanced": {"average_net_price": 5, "DEBT_MDN": 4, "MD_EARN_WNE_P10": 4, "graduation_rate_150pct": 3},
    "📈 Outcomes first": {"MD_EARN_WNE_P10": 5, "graduation_rate_150pct": 4, "retention_rate": 2},
}
DEFAULT_PRESET = "⚖️ Balanced"

SCHOOL_TYPES = ["Public 4-year", "Private nonprofit 4-year", "Private for-profit 4-year"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div.stButton > button[kind="primary"] {font-size: 1.1rem; padding: 0.6rem 1.4rem;}
    .rmc-rank {font-size: 0.85rem; font-weight: 700; opacity: 0.7;}
    .rmc-score {font-size: 1.6rem; font-weight: 700; line-height: 1.1;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------

def get_setting(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        # No secrets.toml at all
        value = None
    return value or os.environ.get(name)


@st.cache_resource(show_spinner="Loading college data…")
def get_context():
    return backend.load_context()


@st.cache_data(show_spinner=False, max_entries=256)
def cached_rankings(weights_items, cipcodes, filter_items):
    # Deterministic and contains no personal text, so safe to share
    return backend.calculate_rankings(
        get_context(),
        dict(weights_items),
        set(cipcodes) if cipcodes else None,
        dict(filter_items),
    )


missing = backend.missing_data_files()
if missing:
    st.error(
        "Some data files are missing from the `data/` folder: "
        + ", ".join(f"`{m}`" for m in missing)
        + ". Copy them from `RankMyCollege_Backend.zip` (see README, section 2)."
    )
    st.stop()

ctx = get_context()
API_KEY = get_setting("GROQ_API_KEY")
if API_KEY and API_KEY.startswith("PUT_YOUR_OWN"):
    API_KEY = None  # unedited example file
MODEL = get_setting("GROQ_MODEL") or backend.DEFAULT_MODEL
AI_ON = bool(API_KEY)

METRIC_OPTIONS = [m for m in backend.METRIC_LABELS if m in ctx.rankable_metrics]
MAJOR_NAMES = dict(zip(ctx.major_catalog["cipcode"], ctx.major_catalog["major_name"]))
MAJOR_COUNTS = dict(zip(ctx.major_catalog["cipcode"], ctx.major_catalog["colleges_reporting"]))
ALL_STATES = sorted(ctx.reference_pool["state"].dropna().unique())


# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------

# Widget keys start with "k_". Re-saving them keeps their values when the
# widget isn't drawn on the current step (Streamlit forgets otherwise).
for key in list(st.session_state.keys()):
    if key.startswith("k_"):
        st.session_state[key] = st.session_state[key]

DEFAULTS = {
    "k_step": STEPS[0],
    "k_prompt": "",
    "k_metrics": [],
    "k_codes": [],
    "k_program_only": True,
    "k_states": [],
    "k_types": [],
    "k_min_enroll": 0,
    "k_price_on": False,
    "k_max_price": 25000,
    "plan": None,
    "baseline": None,
    "explanation": None,
    "ai_error": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# Navigation requested by a button in the previous run
if "_goto" in st.session_state:
    st.session_state.k_step = st.session_state.pop("_goto")


def goto(step):
    st.session_state.k_step = step


def slider_key(metric):
    return f"k_w_{metric}"


def load_weights_into_widgets(weights):
    """Turn weights into 1–10 importance sliders and remember them as the baseline."""
    top = max(weights.values())
    sliders = {m: max(1, round(w / top * SLIDER_MAX)) for m, w in weights.items()}
    st.session_state.k_metrics = list(sliders)
    for metric, value in sliders.items():
        st.session_state[slider_key(metric)] = value
    st.session_state.baseline = {"metrics": list(sliders), "sliders": sliders}
    st.session_state.explanation = None


def apply_plan(plan):
    st.session_state.plan = plan
    load_weights_into_widgets(plan["ai_weights"])
    st.session_state.k_codes = [m["cipcode"] for m in plan["major_matches"]]
    st.session_state.k_program_only = True


def reset_ai_suggestions():
    if st.session_state.plan:
        apply_plan(st.session_state.plan)


def use_preset(name):
    st.session_state.plan = None
    load_weights_into_widgets(PRESETS[name])
    st.session_state.k_codes = []
    goto(STEPS[1])


def set_example(text):
    st.session_state.k_prompt = text


def current_sliders():
    return {
        m: st.session_state.get(slider_key(m), 5)
        for m in st.session_state.k_metrics
    }


def is_unedited():
    base = st.session_state.baseline
    return bool(base) and base["sliders"] == current_sliders()


def effective_weights():
    """Weights used for scoring. Student edits always override the AI."""
    plan = st.session_state.plan
    if plan and is_unedited():
        return dict(plan["ai_weights"])
    return {m: float(v) for m, v in current_sliders().items() if v > 0}


def active_cipcodes():
    if not st.session_state.k_program_only:
        return ()
    return tuple(sorted(st.session_state.k_codes))


def active_filters():
    filters = {}
    if st.session_state.k_states:
        filters["states"] = tuple(sorted(st.session_state.k_states))
    if st.session_state.k_types:
        filters["institution_types"] = tuple(sorted(st.session_state.k_types))
    if st.session_state.k_min_enroll:
        filters["min_enrollment"] = int(st.session_state.k_min_enroll)
    if st.session_state.k_price_on:
        filters["max_average_net_price"] = int(st.session_state.k_max_price)
    return filters


def get_rankings(weights):
    return cached_rankings(
        tuple(sorted(weights.items())),
        active_cipcodes(),
        tuple(sorted(active_filters().items())),
    )


def major_label(code):
    name = MAJOR_NAMES.get(code, code)
    count = MAJOR_COUNTS.get(code)
    suffix = f" · {count:,} colleges" if count else ""
    return f"{name} ({code}){suffix}"


def selected_major_text():
    codes = active_cipcodes()
    if not codes:
        return None
    names = [MAJOR_NAMES.get(c, c) for c in codes]
    return ", ".join(names)


def show_not_measured():
    plan = st.session_state.plan
    if plan and plan["unsupported"]:
        items = "\n".join(f"- {item}" for item in plan["unsupported"])
        st.warning(
            "**Not measured yet** — you asked about things our data can't score, "
            "so your matches do **not** account for them:\n\n" + items,
            icon="⚠️",
        )


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------

st.title("🎓 RankMyCollege")
st.markdown("Find colleges that fit **your** priorities — and see exactly why they rank where they do.")

if AI_ON:
    st.caption(f"✨ AI assistant connected · model `{MODEL}`")
else:
    st.info(
        "**Data-only demo.** The AI assistant isn't connected (no Groq API key), "
        "so we can't read free-text requests. You can still pick priorities by hand — "
        "every score below is calculated from real college data.",
        icon="ℹ️",
    )

st.radio("Step", STEPS, key="k_step", horizontal=True, label_visibility="collapsed")
step = st.session_state.k_step
st.divider()


# ---------------------------------------------------------------
# Screen A — Your priorities
# ---------------------------------------------------------------

def screen_priorities():
    st.header("What matters to you in a college?")

    if AI_ON:
        st.text_area(
            "Describe what you're looking for",
            key="k_prompt",
            height=120,
            max_chars=backend.MAX_PROMPT_CHARS,
            placeholder="e.g. I want an affordable nursing program in New York with strong job outcomes",
            help="Please don't include private info like your name, address, or grades.",
        )
        go = st.button("🔎 Find my matches", type="primary", width="stretch")

        st.caption("Not sure what to write? Try one:")
        for i, example in enumerate(EXAMPLE_PROMPTS):
            st.button(example, key=f"example_{i}", on_click=set_example, args=(example,))

        if go:
            run_agents(st.session_state.k_prompt)

        if st.session_state.ai_error:
            st.error(st.session_state.ai_error, icon="🚫")
            st.button(
                "Set my priorities by hand instead",
                on_click=use_preset,
                args=(DEFAULT_PRESET,),
            )
    else:
        st.text_area(
            "Describe what you're looking for",
            placeholder="AI not connected — use a starting point below instead.",
            disabled=True,
        )

    st.subheader("Or start from a simple preset")
    st.caption("Presets are fixed starting weights (not AI). You'll be able to fine-tune them next.")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=use_preset, args=(name,), width="stretch")


def run_agents(prompt):
    st.session_state.ai_error = None
    if not (prompt or "").strip():
        st.warning("Type a sentence or two about what you want first.")
        return
    try:
        with st.spinner("Reading your priorities and double-checking them…"):
            client = backend.make_client(API_KEY)
            plan = backend.interpret_and_audit(client, ctx, prompt, MODEL)
    except backend.AIServiceError as exc:
        st.session_state.ai_error = f"The AI assistant couldn't respond: {exc}"
        return
    except ValueError as exc:
        st.session_state.ai_error = str(exc)
        return
    apply_plan(plan)
    st.session_state["_goto"] = STEPS[1]
    st.rerun()


# ---------------------------------------------------------------
# Screen B — Review your choices
# ---------------------------------------------------------------

def screen_review():
    st.header("Review your choices")
    plan = st.session_state.plan

    if plan:
        st.markdown(f"> {plan['prompt']}")
        st.caption("Here's how the AI read your request. **Anything you change here overrides the AI.**")
        show_not_measured()
        if plan["dropped"]:
            st.caption(
                "We ignored some AI suggestions that didn't match our data: "
                + "; ".join(plan["dropped"]) + "."
            )
        if plan["cautions"]:
            with st.expander("Notes from the AI double-checker"):
                for caution in plan["cautions"]:
                    st.markdown(f"- {caution}")
    elif not st.session_state.k_metrics:
        st.info("Pick a starting point first.")
        st.button("← Back to priorities", on_click=goto, args=(STEPS[0],))
        return

    # --- Major ---
    st.subheader("🎓 Major")
    codes_default = list(st.session_state.k_codes)
    suggestion_codes = []
    if plan and plan["requested_major"]:
        if plan["major_matches"]:
            names = ", ".join(m["major_name"] for m in plan["major_matches"])
            st.markdown(f"You mentioned **{plan['requested_major']}**. We matched it to: **{names}**.")
        else:
            suggestion_codes = [s["cipcode"] for s in plan["major_suggestions"]]
            if suggestion_codes:
                st.info(
                    f"We couldn't find an exact official program called **{plan['requested_major']}**, "
                    "so we didn't guess. Pick the official name(s) that fit you below "
                    "(closest matches are listed first).",
                    icon="🔎",
                )
            else:
                st.info(
                    f"We couldn't find **{plan['requested_major']}** in the official program list. "
                    "Search for a related name below, or leave it empty to see all colleges.",
                    icon="🔎",
                )

    ordered = suggestion_codes + [c for c in MAJOR_NAMES if c not in suggestion_codes]
    # Keep any previously chosen codes valid options
    ordered += [c for c in codes_default if c not in MAJOR_NAMES]
    st.multiselect(
        "Official program name(s)",
        options=ordered,
        key="k_codes",
        format_func=major_label,
        placeholder="Any major (type to search, e.g. 'nursing')",
        help="Based on colleges that reported bachelor's graduates in each program in 2024.",
    )

    # --- Metrics & weights ---
    st.subheader("📊 What counts toward your score")
    st.multiselect(
        "Things to score colleges on",
        options=METRIC_OPTIONS,
        key="k_metrics",
        format_func=backend.metric_label,
        max_selections=6,
        help="Up to 6. Fewer choices means more colleges have complete data.",
    )
    metrics = st.session_state.k_metrics
    if not metrics:
        st.warning("Choose at least one thing to score colleges on.")
        return

    for metric in metrics:
        st.session_state.setdefault(slider_key(metric), 5)

    sliders = current_sliders()
    total = sum(sliders.values())
    explanations = plan["explanations"] if plan else {}

    for metric in metrics:
        st.slider(
            f"How much does **{backend.metric_label(metric)}** matter?",
            min_value=0,
            max_value=SLIDER_MAX,
            key=slider_key(metric),
            help=backend.METRIC_LABELS[metric]["help"],
        )
        share = (st.session_state[slider_key(metric)] / total * 100) if total else 0
        why = explanations.get(metric)
        note = f"Share of your score: **{share:.0f}%**"
        if why:
            note += f" · AI: _{why}_"
        st.caption(note)

    if total == 0:
        st.warning("At least one slider needs to be above zero.")
        return

    st.caption("Weights are converted to percentages automatically, so they always add up to 100%.")

    cols = st.columns(2)
    cols[0].button(
        "See my matches →", type="primary", on_click=goto, args=(STEPS[2],),
        width="stretch",
    )
    if plan:
        cols[1].button(
            "↺ Reset AI suggestions", on_click=reset_ai_suggestions,
            width="stretch", disabled=is_unedited() and
            st.session_state.k_codes == [m["cipcode"] for m in plan["major_matches"]],
        )


# ---------------------------------------------------------------
# Screen C — Your matches
# ---------------------------------------------------------------

def render_filters():
    with st.expander("🔧 Filter results", expanded=bool(active_filters())):
        st.caption("Filters only hide colleges. They never change a college's score — just its place in your list.")
        st.multiselect("States", ALL_STATES, key="k_states", placeholder="All states")
        st.multiselect("School type", SCHOOL_TYPES, key="k_types", placeholder="All types")
        st.number_input(
            "Minimum undergraduate students", min_value=0, max_value=60000, step=500, key="k_min_enroll",
        )
        st.toggle("Set a maximum average net price", key="k_price_on")
        if st.session_state.k_price_on:
            st.slider(
                "Max average net price (per year)", 0, 60000, step=1000, key="k_max_price",
                format="$%d",
            )
        if st.session_state.k_codes:
            st.toggle(
                "Only colleges that reported graduates in my major", key="k_program_only",
            )


def render_counts(result):
    c = result["counts"]
    st.markdown(
        f"**{c['reference_scored']:,}** of {c['reference_total']:,} colleges have all the data you "
        f"selected and were scored. **{c['reference_incomplete']:,}** are missing at least one of "
        "those numbers, so they're listed separately instead of being guessed."
    )
    parts = []
    major = selected_major_text()
    if major:
        parts.append(f"**{c['major_scored']:,}** reported 2024 bachelor's graduates in {major}")
    if result["filters"]:
        parts.append(f"**{c['shown_scored']:,}** match your filters")
    if parts:
        st.markdown(" · ".join(parts))


def render_card(row, weights):
    with st.container(border=True):
        st.markdown(f"<div class='rmc-rank'>#{int(row['rank'])}</div>", unsafe_allow_html=True)
        st.markdown(f"**{row['college_name']}**")
        st.caption(f"{row['city']}, {row['state']} · {row['institution_type']}")
        st.markdown(
            f"<div class='rmc-score'>{row['final_score']:.1f}</div>"
            "<small>match score / 100</small>",
            unsafe_allow_html=True,
        )
        for reason in backend.top_reasons(row, weights):
            st.markdown(f"<small>✅ {reason}</small>", unsafe_allow_html=True)
        if pd.notna(row.get("matching_major_awards_2024")):
            st.markdown(
                f"<small>🎓 {int(row['matching_major_awards_2024']):,} bachelor's degrees "
                "in your major reported in 2024</small>",
                unsafe_allow_html=True,
            )


def top_table(ranked, weights, n=10):
    head = ranked.head(n)
    table = pd.DataFrame({
        "Rank": head["rank"],
        "College": head["college_name"],
        "State": head["state"],
        "Type": head["institution_type"],
        "Score": head["final_score"].round(1),
    })
    for metric in weights:
        table[backend.metric_label(metric, short=True)] = head[metric].map(
            lambda v, m=metric: backend.format_metric_value(m, v)
        )
    if "matching_major_awards_2024" in head:
        table["Major grads (2024)"] = head["matching_major_awards_2024"].astype("Int64")
    if active_cipcodes() or active_filters():
        table["Rank among all colleges"] = head["overall_rank"]
    return table


def screen_matches():
    st.header("Your matches")
    weights = effective_weights()
    if not weights:
        st.info("Set your priorities first.")
        st.button("← Back to review", on_click=goto, args=(STEPS[1],))
        return

    show_not_measured()
    render_filters()

    try:
        result = get_rankings(weights)
    except ValueError as exc:
        st.error(f"We couldn't score colleges with these choices: {exc}")
        return

    weights = result["weights"]
    ranked = result["ranked"]
    render_counts(result)

    if ranked.empty:
        st.info(
            "No scored colleges match these filters. Try fewer filters, more states, "
            "or choosing fewer metrics so more colleges have complete data.",
            icon="🤔",
        )
        return

    st.caption(
        "Scores are relative (0–100) and only reflect the priorities you chose. "
        "They are **not** admission chances or a measure of program quality."
    )

    st.subheader("🏆 Top 3")
    cols = st.columns(min(3, len(ranked)))
    for col, (_, row) in zip(cols, ranked.head(3).iterrows()):
        with col:
            render_card(row, weights)

    st.subheader("Top 10")
    st.dataframe(
        top_table(ranked, weights),
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        },
    )

    st.download_button(
        "⬇️ Download all scored matches (CSV)",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name="rankmycollege_matches.csv",
        mime="text/csv",
    )

    incomplete = result["incomplete"]
    if len(incomplete):
        with st.expander(f"Colleges we couldn't score ({len(incomplete):,}) — missing data"):
            st.caption("These colleges didn't report one or more of your selected numbers. That doesn't mean they're worse.")
            st.dataframe(
                pd.DataFrame({
                    "College": incomplete["college_name"],
                    "State": incomplete["state"],
                    "Missing": incomplete["missing_metrics"].map(
                        lambda ms: ", ".join(backend.metric_label(m, short=True) for m in ms)
                    ),
                    "Data available": incomplete["data_coverage_pct"].round(0).astype(int).astype(str) + "%",
                }),
                hide_index=True,
                width="stretch",
            )

    render_explanation(result)

    st.button("Compare two schools →", on_click=goto, args=(STEPS[3],), type="primary")


def explanation_signature(result):
    return (
        st.session_state.plan["prompt"] if st.session_state.plan else None,
        tuple(sorted(result["weights"].items())),
        tuple(result["cipcodes"]),
        tuple(sorted(result["filters"].items())),
    )


def render_explanation(result):
    st.subheader("💬 Why these colleges?")
    plan = st.session_state.plan

    if not (AI_ON and plan):
        st.caption(
            "The reasons on each card come straight from the numbers. "
            + ("Connect the AI assistant and describe your priorities to get a written explanation."
               if not AI_ON else
               "Describe your priorities on step 1 to get a written AI explanation.")
        )
        return

    saved = st.session_state.explanation
    signature = explanation_signature(result)

    if st.button("✨ Explain my top matches"):
        try:
            with st.spinner("Writing a short explanation…"):
                client = backend.make_client(API_KEY)
                ai_result = get_rankings(plan["ai_weights"])
                report = backend.explain_verified_results(
                    client, ctx, plan["prompt"], ai_result, result,
                    plan["ai_weights"], result["weights"],
                    selected_major_text(), plan["unsupported"], MODEL,
                )
            saved = {"report": report.model_dump(), "signature": signature}
            st.session_state.explanation = saved
        except backend.AIServiceError as exc:
            st.error(
                f"The AI explainer couldn't respond: {exc} "
                "The score breakdowns on the cards and on the Compare step still work.",
                icon="🚫",
            )
        except ValueError as exc:
            st.error(f"Couldn't build an explanation: {exc}")

    if not saved:
        return

    if saved["signature"] != signature:
        st.caption("⚠️ You changed your choices since this was written — tap the button again to refresh it.")

    report = saved["report"]
    with st.container(border=True):
        st.markdown(f"**Top match with the AI's suggested weights**  \n{report['original_winner_explanation']}")
        st.markdown(f"**Top match with your current weights**  \n{report['custom_winner_explanation']}")
        st.markdown(f"**What changed**  \n{report['why_rankings_changed']}")
        if report["key_tradeoffs"]:
            st.markdown("**Tradeoffs to think about**")
            for item in report["key_tradeoffs"]:
                st.markdown(f"- {item}")
    st.caption("AI-written from the verified numbers above. It can still make mistakes — check the table.")


# ---------------------------------------------------------------
# Screen D — Compare and understand
# ---------------------------------------------------------------

COMPARE_METRICS = ["average_net_price", "DEBT_MDN", "MD_EARN_WNE_P10", "graduation_rate_150pct"]


def screen_compare():
    st.header("Compare and understand")
    weights = effective_weights()
    if not weights:
        st.info("Set your priorities first.")
        return
    try:
        result = get_rankings(weights)
    except ValueError as exc:
        st.error(f"We couldn't score colleges with these choices: {exc}")
        return

    ranked = result["ranked"]
    weights = result["weights"]
    if len(ranked) < 1:
        st.info("No matches to compare — loosen your filters on step 3.")
        return

    lookup = ranked.set_index("unitid")
    ids = ranked["unitid"].tolist()

    def label(uid):
        row = lookup.loc[uid]
        return f"#{int(row['rank'])} {row['college_name']} ({row['state']})"

    for key, default in (("k_cmp_a", 0), ("k_cmp_b", 1), ("k_detail", 0)):
        if st.session_state.get(key) not in ids:
            st.session_state[key] = ids[min(default, len(ids) - 1)]

    if len(ids) >= 2:
        cols = st.columns(2)
        cols[0].selectbox("School A", ids, key="k_cmp_a", format_func=label)
        cols[1].selectbox("School B", ids, key="k_cmp_b", format_func=label)

        a = lookup.loc[st.session_state.k_cmp_a]
        b = lookup.loc[st.session_state.k_cmp_b]

        rows = [
            ("Rank in your matches", f"#{int(a['rank'])}", f"#{int(b['rank'])}"),
            ("Match score", f"{a['final_score']:.1f}", f"{b['final_score']:.1f}"),
        ]
        for metric in COMPARE_METRICS:
            rows.append((
                backend.metric_label(metric),
                backend.format_metric_value(metric, a[metric]),
                backend.format_metric_value(metric, b[metric]),
            ))
        if "matching_major_awards_2024" in lookup.columns:
            rows.append((
                "Bachelor's degrees in your major (2024)",
                f"{int(a['matching_major_awards_2024']):,}",
                f"{int(b['matching_major_awards_2024']):,}",
            ))
        for metric in weights:
            rows.append((
                f"Points from {backend.metric_label(metric, short=True)}",
                f"{a[f'{metric}_contribution']:.1f}",
                f"{b[f'{metric}_contribution']:.1f}",
            ))
        st.dataframe(
            pd.DataFrame(rows, columns=["", a["college_name"], b["college_name"]]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Net price, debt, and earnings are **college-wide averages/medians** from federal data — "
            "not your personal cost, aid offer, debt, or future salary. Earnings are measured "
            "10 years after students *started* college, across all majors. "
            "Degree counts show a program exists, not how good it is."
        )

    st.subheader("What drove the score?")
    st.selectbox("Pick a school", ids, key="k_detail", format_func=label)
    row = lookup.loc[st.session_state.k_detail]
    breakdown = backend.score_breakdown(row, weights)
    st.dataframe(
        pd.DataFrame({
            "What we measured": breakdown["label"],
            "Actual value": breakdown["value"],
            "Your weight": breakdown["weight_pct"].round(0).astype(int).astype(str) + "%",
            "Percentile (100 = best)": breakdown["percentile"].round(0).astype(int),
            "Points": breakdown["points"].round(1),
        }),
        hide_index=True,
        width="stretch",
        column_config={
            "Points": st.column_config.ProgressColumn("Points", min_value=0, max_value=100, format="%.1f"),
        },
    )
    st.markdown(
        f"Points add up to the match score: **{breakdown['points'].sum():.1f}** "
        f"= {row['final_score']:.1f}. Each metric gives up to (your weight × 100) points, "
        "depending on how the college compares with every college that reports that number."
    )


# ---------------------------------------------------------------
# Render
# ---------------------------------------------------------------

if step == STEPS[0]:
    screen_priorities()
elif step == STEPS[1]:
    screen_review()
elif step == STEPS[2]:
    screen_matches()
else:
    screen_compare()

st.divider()
with st.expander("ℹ️ What these scores mean (and don't)"):
    st.markdown(
        """
- **Scores are relative.** Each college is compared with ~2,500 four-year colleges on the
  things *you* chose, then weighted by how much you said they matter.
- **We don't measure:** program quality or rankings, prestige, employer reputation,
  your personal financial aid, your chances of getting in, or your future salary.
- **Average net price** is a college-wide average after grants — your price may be very different.
  Use each college's net price calculator.
- **Earnings** are median earnings of former students **10 (or 6) years after they entered**
  college, across all majors — not after graduation, and not for your major.
- **Major matching** uses colleges that reported bachelor's graduates in that program in 2024.
  A college missing from the list may still offer the major.
- **Missing data** isn't guessed. Colleges without one of your selected numbers are listed
  separately instead of being scored.

Data: U.S. Department of Education College Scorecard and IPEDS (2024 completions, CIP 2020).
"""
    )
