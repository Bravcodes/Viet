# RankMyCollege

Personalized, transparent college rankings for high school seniors. A student describes what they want; two Gemini agents turn that into measurable priorities. Python scores ~2,500 four-year colleges against real federal data, and a third agent explains the results in plain language.

The scores are relative and only reflect the priorities the student chose. They do **not** measure program quality, prestige, employer reputation, personal financial aid, admission chances, or future salary, and the app says so on screen.

## Quick start

```bash
cd RankMyCollege
python3 -m venv .venv            # Python 3.11 or 3.12
source .venv/bin/activate         # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Without an API key the app runs as a clearly labeled **data-only demo**: students start from a preset and adjust sliders by hand. Everything except free-text interpretation and the AI explanation still works.

To turn on the AI agents:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# edit .streamlit/secrets.toml and set GEMINI_API_KEY (git-ignored, never commit it)
```

`GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`, the model in the backup's `config.json`. You can override it in the secrets file or with an environment variable. If Gemini returns a 404 for the model, list the models your key can use:

```python
from google import genai
print([m.name for m in genai.Client(api_key="...").models.list()])
```

## Project layout

```text
RankMyCollege/
├── streamlit_app.py                 # UI: Priorities → Review → Matches → Compare
├── backend_v3.py                    # V3 agents + ranking engine + frontend adapter
├── RankMyCollege_AI_Agents.ipynb    # source notebook (reference only; contains no key)
├── requirements.txt
├── data/
│   ├── colleges_plus_scorecard.csv  # required
│   ├── metrics.json                 # required
│   ├── college_programs_named.csv   # required
│   ├── major_catalog.csv            # optional (rebuilt at load time)
│   └── CIPCode2020.csv              # optional reference
├── tests/
│   ├── test_backend.py              # notebook tests 1–8 + README §7 checks + adapter
│   └── test_app.py                  # Streamlit AppTest runs with Gemini faked
└── .streamlit/
    ├── config.toml
    └── secrets.example.toml         # copy to secrets.toml locally
```

## How the backend is wired

`backend_v3.py` keeps the **last tested** definition of each notebook function: `MetricChoice`, `MajorAwarePlan`, `ExplanationReport` (4-field V3), `prepare_reference_pool`, `validate_weights`, `search_majors`, `resolve_major`, `preference_agent_v3`, `auditor_agent_v3`, the V3 `explainer_agent`, `rank_colleges_v2`, `filter_rankings_by_major`, `filter_ranked_colleges`, `prepare_explainer_evidence_v2` and `run_rankmycollege_v3`. The prompts and scoring math are unchanged. Differences from the notebook:

- **No Colab code.** The notebook globals (`rankable_metrics`, `reference_pool`, `reported_programs`, `major_catalog`) live in a `BackendContext` built by `load_context()` from `data/`. The Gemini client is created from a key passed in explicitly. Importing the module makes no API calls and prints nothing.
- **Adapter**, so moving a slider never re-runs the agents:

  | Function | When it runs | Calls Gemini? |
  |---|---|---|
  | `interpret_and_audit(client, ctx, prompt)` | Once, on **Find my matches** | Agents 1 + 2 |
  | `calculate_rankings(ctx, weights, cipcodes, filters)` | Every slider, major or filter change | No |
  | `explain_verified_results(...)` | Only on **Explain my top matches** | Agent 3 |

- **Validation.** AI metric picks that are unknown, duplicated, non-positive, or over the six-metric limit are dropped and listed on screen before scoring (`sanitize_plan`). Student slider edits always override the AI.
- **Majors.** `resolve_major` accepts only an exact official name or its ", General" / ", Other" variants. Anything else (for example "Nursing") returns catalog suggestions for the student to choose from; the app never guesses a CIP code. Programs are matched by `unitid` using positive 2024 bachelor's completions.
- **Scores never move when filters change.** Every college is scored against the full 2,490-college reference pool first. The major, state, type, enrollment and price filters only decide which scored colleges are shown, and they renumber the displayed rank. `overall_rank` is kept.
- **Gemini errors** are reported as `GeminiError(kind)`: `no_key`, `model_not_found` (404), `quota` (429), `unavailable` (5xx, retried once), `bad_key` or `bad_response`. The UI shows a friendly message and the deterministic ranking keeps working.

## Tests

```bash
python -m pytest -q
```

37 tests, no network or API key needed:

- The notebook's eight backend checks.
- The historical counts: 2,796 rows, 2,490 reference pool, 1,988 scored with four metrics, 2,217 with two, and 702 for Computer Science.
- Scored plus missing-data colleges add up to the reference pool, and contributions sum to each score.
- Filters never change a college's score; changing weights changes the order; an empty filter result doesn't crash.
- "Nursing" returns suggestions; "Data Science" resolves to 30.7001 and 30.7099.
- With Gemini faked: Agents 1 and 2 run exactly once per prompt, and sliders and filters make no further calls. Agent 3 runs only on request and receives the unsupported priorities. "Not measured yet" is shown, Reset restores the AI weights, a quota error falls back to manual mode, and the key never appears in the UI.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo. `.gitignore` already excludes `.streamlit/secrets.toml` and `.venv/`. Before your first commit, check that `git status` doesn't list a secrets file.
2. At <https://share.streamlit.io/>, choose **Create app**, pick the repo and branch, and set the entrypoint to `streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.12. In **Secrets**, add `GEMINI_API_KEY = "..."` and optionally `GEMINI_MODEL`.
4. Deploy, then run the README §7 checks against the public URL in a private window on both phone and desktop.

## Status: what is live and what is demo-only

| Component | Status |
|---|---|
| Deterministic ranking, filters, major matching, compare, breakdown, CSV export | **Live**; tested |
| Agents 1 + 2 (interpret + audit) | **Wired and tested with a fake Gemini client.** Needs a real key for a live run |
| Agent 3 (explainer) | **Wired and tested with a fake Gemini client.** Needs a real key for a live run |
| Model | `gemini-3.5-flash-lite` (configurable) |

Known limitations: the major filter uses 2024 completion evidence, so a missing college may still offer the major. Earnings are institution-wide figures measured 10 or 6 years after **entry**, not graduation. Net price is a college-wide average, not a student's own aid offer.
