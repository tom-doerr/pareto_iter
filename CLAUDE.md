# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pareto DSPy is a Flask + HTMX web application that evaluates design candidates against a customizable rubric using DSPy (a programming framework for language models). Users can edit the scoring criteria directly in the UI, pick an OpenRouter model (default: Gemini 2.5 Flash), submit candidate concepts, review Pareto scores, and inspect every LLM-generated improvement produced during iterative domination attempts.

## Architecture

### Core Components

**pareto_dspy.py** – DSPy integration + Pareto logic
- Lazily configures a `dspy.LM` client for OpenRouter per evaluation request (requires `OPENROUTER_API_KEY`; imports succeed without credentials) and honors the model selected in the UI (default `google/gemini-2.5-flash`).
- Exposes rubric helpers: `DEFAULT_CRITERIA`, `get_default_criteria()`, `build_rubric_text()`, `_normalize_criteria()`.
- Defines DSPy signatures/predictors for scoring (`ScoreArtifactSig`) and improvement synthesis (`ImproveFromParetoSig`), instantiated on demand.
- Pareto utilities (`dominates`, `pareto_front`, `format_pareto_context`) operate on dynamic criteria lists.
- Public API: `evaluate_candidates(problem, candidates, criteria, *, max_iterations, stall_tolerance)` returns sanitized criteria, rubric text, per-candidate scores/justifications, Pareto front IDs, context, the improvement history, and metadata about the underlying LLM.

### Flask App (`app.py`)

- Routes:
  - `/` renders the main form with sample problem text, seed candidates, default criteria, and the model picker.
  - `/criterion-field` and `/candidate-field` are HTMX endpoints that append form controls without a full page reload.
  - `/evaluate` collects form data, normalizes candidates/criteria, invokes either the real or fake evaluator, and renders `_results.html`.
- Candidate artifacts are self-contained; helper functions (`_extract_candidate_label`, `_derive_candidate_identity`) derive a display label and a slugified identifier from the artifact content (preferring a `title:` key, falling back to the first non-empty line, then to `Candidate N`). Collisions are resolved by suffixing numbers.
- Fake responses for local demos use `_make_fake_evaluation` when `PARETO_FAKE_EVALUATION` is set.
- Jinja templates live in `templates/`, and the monochrome glassmorphism theme is in `static/style.css`; HTMX handles incremental updates.

### Data Flow

1. User edits/adds evaluation criteria, then submits a problem statement and candidate artifacts (YAML/Markdown with title, assumptions, key_decisions, risks, metrics, etc.).
2. `evaluate_candidates()` builds rubric instructions from the submitted criteria and scores each candidate through DSPy/OpenRouter.
3. Pareto front is computed over the dynamic criteria set, and iterative improvement runs for up to the requested `max_iterations`, halting early when the stall tolerance is reached or no usable improvement text is returned.
4. Every synthesized candidate is scored and appended to the running history; the last non-empty improvement is surfaced as the “Improved” concept while the full sequence shows under “LLM Improvements.”
5. Results view lists the rubric used, per-criterion scores with justifications, Pareto front members, the Pareto context string, the final improvement, and the full iteration history.

## Running the Application

### Docker Compose (preferred)
```bash
docker compose up --watch
```
The app serves at http://localhost:5050 (non-standard port to avoid conflicts). `--watch` keeps the container in sync with local edits and rebuilds when `Dockerfile` or `requirements.txt` change.

### Local virtualenv (optional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --port 5050
```

## Environment Configuration

Populate `.env` (ignored by git) with:
- `OPENROUTER_API_KEY` – required OpenRouter API key.
- Optional overrides: `OPENROUTER_MODEL` (default `google/gemini-2.5-flash`), `OPENROUTER_API_BASE` (default `https://openrouter.ai/api/v1`), `OPENROUTER_SITE_URL`, `OPENROUTER_APP_TITLE` (used for OpenRouter headers).

## Dependencies

- Flask ≥ 3.0 (web framework)
- HTMX (loaded via CDN in templates)
- gunicorn ≥ 21.2 (production WSGI server)
- DSPy ≥ 3.0 (LLM programming layer)

Install locally with `pip install -r requirements.txt` or rely on the Docker image.

## User Controls

- The main form exposes a numeric "Max improvement iterations" input (`max_iters`) to loop the Pareto improvement up to five times. Each iteration scores the latest generated candidate and feeds it back into the front before prompting for another improvement.
- A second numeric input (`stall_limit`) lets the user stop the loop after a chosen number of consecutive non-improving iterations (set `0` to disable early stopping).
- Iteration results render in the results panel with badges indicating whether each synthesized candidate improved the aggregate score.
- Pressing **Evaluate Pareto Loop** runs the configured iterations and surfaces every generated proposal (including the final dominating suggestion) directly in the web UI under “LLM Improvements.”
- The improvements section summarizes how many iterations actually ran alongside the configured maximum and stall limit for quick sanity checks.
- An expandable “How the iteration loop works” panel in the form explains how to configure `max_iters` and `stall_limit`; the results remind users they can rerun with new settings to keep refining.
- The summary highlights how many iterations produced no improvement (plus the current streak) so it’s obvious when the loop hit the stall limit.
- Evaluation criteria now sit inside a collapsible panel with the “Add Criterion” action, keeping the form concise when the list grows.
- Evaluation criteria now live inside a collapsible panel with an Add Criterion action so the form stays compact while editing long candidate lists.
- Candidate artifacts are entered as self-contained YAML or Markdown; the app will use a `title:` key if present, otherwise the first non-empty line (or a fallback label) so no separate name field is required.

## Frontend Styling

- UI lives in `templates/` with shared layout in `templates/base.html` and styling in `static/style.css`.
- The palette is monochromatic: all components reference neutral tokens defined under `:root` (e.g., `--ink`, `--surface`, `--muted-*`).
- Panels, cards, inputs, alerts, and the header use glassmorphism (semi-transparent backgrounds, `backdrop-filter` blur, and soft drop shadows) to keep legibility while contrasting against the gradient backdrop.
- The page background is a subtle grey gradient with radial highlights applied via a `body::before` overlay; this keeps depth without reintroducing color accents.
- Form inputs explicitly set text and placeholder colors to ensure readability atop translucent surfaces.

## Key Design Patterns / Notes

- DSPy LM configuration is lazy; calling `evaluate_candidates` without an OpenRouter key raises a clear error, while the chosen model (from the form or `OPENROUTER_MODEL`) is echoed back in result metadata.
- Criteria are user-editable; `_normalize_criteria` deduplicates, trims whitespace, and enforces at least one criterion (all maximize, scores clamped to [0, 1]).
- Inputs `max_iterations` and `stall_limit` (a.k.a. `stall_tolerance`) drive the improvement loop, with guards that coerce invalid values into safe ranges (1–10 iterations, stall ≤ iterations).
- Iterations stop early when the stall limit is reached or when the improver returns blank text; each attempt is recorded in `improvements` with score deltas and justifications.
- Candidate identifiers are slugified from these derived labels with collision handling, keeping the Pareto math stable while rendering human-friendly names in the UI; missing titles automatically fall back to sensible defaults.
- LLM responses are sanitized (scores clamped, justifications stringified) before use.
- Defensive exception handling surfaces LLM parsing or evaluation errors in the UI.
- HTMX powers both candidate and criterion additions for a dynamic UX without page reloads.

- **Unit / integration**: live in `tests/`.
  - `tests/test_pareto.py` stubs DSPy to exercise rubric generation, iteration limits, stall tolerance, empty-response handling, and validation guards.
  - `tests/test_app.py` drives the Flask routes via the test client and validates the rendered HTML.
- **Playwright E2E**: `tests/e2e/test_ui_playwright.py` spins up the Flask app (with `PARETO_FAKE_EVALUATION=1`) and automates the web UI via Playwright’s Python bindings. It launches `/snap/bin/chromium` by default; override with `PLAYWRIGHT_CHROMIUM` if needed.
  - These tests require permission to bind a local TCP socket. In sandboxes where socket creation is forbidden they auto-skip; otherwise ensure the system Chromium is available and run `pytest tests/e2e/test_ui_playwright.py`.
- Global test command: `pytest` (outputs the unit/integration suite and the Playwright E2E, skipping the latter automatically if sockets are disallowed). A benign DeprecationWarning from litellm cache setup may appear.
