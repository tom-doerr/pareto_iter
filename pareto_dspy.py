"""Pareto DSPy web integration utilities."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence

import dspy

# ---------------------------------------------------------------------------
# Rubric seed data and helpers
# ---------------------------------------------------------------------------
DEFAULT_CRITERIA: List[Dict[str, str]] = [
    {"name": "Impact", "desc": "Expected value toward core objective."},
    {"name": "CostEfficiency", "desc": "Value per unit of resources."},
    {"name": "Speed", "desc": "Time-to-value and iteration velocity."},
    {"name": "Simplicity", "desc": "Low complexity; easy to operate/maintain."},
    {"name": "Robustness", "desc": "Stable under uncertainty and perturbations."},
    {"name": "RiskControl", "desc": "Safety/compliance; low failure/severity."},
    {"name": "Scalability", "desc": "Improves with scale; low marginal cost."},
    {"name": "Reversibility", "desc": "Easy to roll back/pivot with low sunk cost."},
]


def get_default_criteria() -> List[Dict[str, str]]:
    """Return a shallow copy of the default criteria list."""

    return [dict(item) for item in DEFAULT_CRITERIA]


def build_rubric_text(criteria: Sequence[Dict[str, str]]) -> str:
    """Create rubric instructions from the provided criteria."""

    lines: List[str] = []
    for entry in criteria:
        name = entry.get("name", "").strip()
        if not name:
            continue
        desc = entry.get("desc", "").strip() or "No description provided."
        lines.append(f"- {name}: {desc} (score 0.0..1.0; 1.0 is best)")

    lines.extend(
        [
            "- Calibrate tightly. Use increments of ~0.05. Do not change definitions mid-run.",
            "- Prefer evidence and explicit assumptions. If unknown, estimate and lower confidence.",
        ]
    )
    return "\n".join(lines)


def _normalize_criteria(raw_criteria: Sequence[Dict[str, str]] | None) -> List[Dict[str, str]]:
    """Deduplicate and sanitize criteria input."""

    source = raw_criteria or DEFAULT_CRITERIA
    normalized: List[Dict[str, str]] = []
    seen = set()
    for entry in source:
        name = (entry.get("name") or "").strip()
        if not name or name in seen:
            continue
        desc = (entry.get("desc") or "").strip()
        normalized.append({"name": name, "desc": desc})
        seen.add(name)

    if not normalized:
        raise ValueError("At least one evaluation criterion is required.")
    return normalized


# ---------------------------------------------------------------------------
# DSPy signatures and LM configuration
# ---------------------------------------------------------------------------
class ScoreArtifactSig(dspy.Signature):
    """Score an artifact against a rubric.
    Return JSON only: {"scores": {criterion: float in [0,1]}, "justifications": {criterion: "1-2 sentences"}}
    """

    rubric = dspy.InputField(desc="Universal criteria and scoring instructions.")
    problem = dspy.InputField(desc="Short problem statement/context.")
    artifact = dspy.InputField(desc="The candidate design as text (YAML/Markdown ok).")
    scores_json = dspy.OutputField(desc="Strict JSON with keys: scores, justifications.")


class ImproveFromParetoSig(dspy.Signature):
    """Synthesize a strictly-better candidate from the Pareto front.
    Output YAML with keys: title, assumptions, key_decisions, risks, metrics (estimates), predicted_scores, why_it_should_dominate.
    """

    rubric = dspy.InputField()
    problem = dspy.InputField()
    pareto_context = dspy.InputField(desc="Table of front candidates with their scores and brief notes.")
    improved_yaml = dspy.OutputField(desc="YAML of the improved candidate (no extra prose).")


def _build_lm(model_name: str | None) -> tuple[dspy.LM, str, str]:
    """Construct an LM client configured for OpenRouter.

    Returns a tuple of (lm_instance, requested_model_name, provider_prefixed_model).
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required to call OpenRouter."
        )

    model = (model_name or os.getenv("OPENROUTER_MODEL") or "google/gemini-2.5-flash").strip()
    api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

    extra_headers = {}
    site = os.getenv("OPENROUTER_SITE_URL")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if site:
        extra_headers["HTTP-Referer"] = site
    if title:
        extra_headers["X-Title"] = title

    model_identifier = model if "/" in model else f"openrouter/{model}"
    lm_instance = dspy.LM(
        model=model_identifier,
        api_key=api_key,
        base_url=api_base,
        extra_headers=extra_headers or None,
    )
    return lm_instance, model, model_identifier


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------
def dominates(a: Dict[str, float], b: Dict[str, float], dims: Iterable[str]) -> bool:
    """Return True if candidate `a` Pareto-dominates candidate `b` on `dims`."""

    ge_all = all(a.get(k, 0.0) >= b.get(k, 0.0) for k in dims)
    gt_any = any(a.get(k, 0.0) > b.get(k, 0.0) for k in dims)
    return ge_all and gt_any


def pareto_front(
    score_table: Dict[str, Dict[str, float]],
    dims: Sequence[str],
) -> List[str]:
    """Return candidate IDs on the Pareto front for the provided dimensions."""

    dims = list(dims)
    ids = list(score_table.keys())
    front: List[str] = []
    for cand_id in ids:
        candidate_scores = score_table[cand_id]
        dominated = any(
            dominates(score_table[other_id], candidate_scores, dims)
            for other_id in ids
            if other_id != cand_id
        )
        if not dominated:
            front.append(cand_id)
    return front


def format_pareto_context(
    candidates: Iterable[str],
    scores: Dict[str, Dict[str, Dict[str, float]]],
    criteria_names: Sequence[str],
) -> str:
    """Build a compact context string for the LLM using the front candidates."""

    criteria_names = list(criteria_names)
    lines: List[str] = ["Pareto Front:"]
    for cand_id in candidates:
        summary = scores[cand_id]["scores"]
        score_line = ", ".join([f"{crit}={summary.get(crit, 0.0):.2f}" for crit in criteria_names])
        lines.append(f"- {cand_id}: {score_line}")
        strengths = sorted(criteria_names, key=lambda key: summary.get(key, 0.0), reverse=True)[:3]
        if strengths:
            lines.append(f"  strengths: {', '.join(strengths)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def score_candidate(
    predictor: dspy.Predict,
    problem: str,
    artifact_text: str,
    rubric_text: str,
    criteria_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Call the scorer and return parsed JSON with clamped floats."""

    out = predictor(rubric=rubric_text, problem=problem, artifact=artifact_text)
    try:
        payload = json.loads(out.scores_json)
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        raise RuntimeError(f"Scoring JSON parse failed: {exc}\nRaw: {out.scores_json}") from exc

    raw_scores = payload.get("scores") or {}
    raw_justifications = payload.get("justifications") or {}
    normalized_scores: Dict[str, float] = {}
    normalized_justifications: Dict[str, str] = {}

    for name in criteria_names:
        raw_value = raw_scores.get(name, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        normalized_scores[name] = max(0.0, min(1.0, value))
        normalized_justifications[name] = str(raw_justifications.get(name, "")).strip()

    payload["scores"] = normalized_scores
    payload["justifications"] = normalized_justifications
    return payload


def synthesize_improved(
    predictor: dspy.Predict,
    problem: str,
    pareto_ctx: str,
    rubric_text: str,
) -> str:
    """Call the improver and return YAML string."""

    out = predictor(rubric=rubric_text, problem=problem, pareto_context=pareto_ctx)
    return out.improved_yaml


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------
def evaluate_candidates(
    problem: str,
    candidates: Dict[str, str],
    criteria: Sequence[Dict[str, str]] | None = None,
    model: str | None = None,
    max_iterations: int | None = None,
    stall_tolerance: int | None = None,
) -> Dict[str, object]:
    """Evaluate candidates and generate Pareto-improved suggestion."""

    if not problem.strip():
        raise ValueError("Problem statement cannot be empty.")
    if not candidates:
        raise ValueError("At least one candidate is required.")

    try:
        max_iterations_val = int(max_iterations) if max_iterations is not None else 1
    except (TypeError, ValueError):
        max_iterations_val = 1

    if max_iterations_val < 1:
        max_iterations_val = 1
    elif max_iterations_val > 10:
        max_iterations_val = 10

    try:
        stall_limit_val = int(stall_tolerance) if stall_tolerance is not None else 1
    except (TypeError, ValueError):
        stall_limit_val = 1

    if stall_limit_val < 0:
        stall_limit_val = 0
    elif stall_limit_val > max_iterations_val:
        stall_limit_val = max_iterations_val

    normalized_criteria = _normalize_criteria(criteria)
    criteria_names = [item["name"] for item in normalized_criteria]
    rubric_text = build_rubric_text(normalized_criteria)

    lm, model_requested, model_identifier = _build_lm(model)

    scored: Dict[str, Dict[str, Dict[str, float]]] = {}
    improvements: List[Dict[str, object]] = []
    with dspy.context(lm=lm):
        score_predictor = dspy.Predict(ScoreArtifactSig)
        improve_predictor = dspy.Predict(ImproveFromParetoSig)

        for cand_id, artifact in candidates.items():
            if not artifact.strip():
                continue
            scored[cand_id] = score_candidate(
                score_predictor,
                problem,
                artifact,
                rubric_text,
                criteria_names,
            )

        if not scored:
            raise ValueError("No non-empty candidates provided.")

        numeric_scores = {cid: data["scores"] for cid, data in scored.items()}
        best_total = max((sum(scores.values()) for scores in numeric_scores.values()), default=0.0)
        stalls = 0

        for iteration in range(max_iterations_val):
            front_ids = pareto_front(numeric_scores, criteria_names)
            pareto_ctx_iter = format_pareto_context(front_ids, scored, criteria_names)

            improved_text = synthesize_improved(
                improve_predictor,
                problem,
                pareto_ctx_iter,
                rubric_text,
            )
            improved_text = improved_text.strip()

            base_id = f"improved_iter_{iteration + 1}"
            new_id = base_id
            suffix = 2
            while new_id in scored:
                new_id = f"{base_id}_{suffix}"
                suffix += 1

            improvement_payload: Dict[str, object] = {
                "id": new_id,
                "text": improved_text,
            }
            improvements.append(improvement_payload)

            if not improved_text:
                improvement_payload["improved"] = False
                break

            scored[new_id] = score_candidate(
                score_predictor,
                problem,
                improved_text,
                rubric_text,
                criteria_names,
            )

            improvement_payload["scores"] = scored[new_id]["scores"]
            improvement_payload["justifications"] = scored[new_id]["justifications"]
            total_score = sum(improvement_payload["scores"].values())

            if total_score > best_total + 1e-6:
                best_total = total_score
                stalls = 0
                improvement_payload["improved"] = True
            else:
                stalls += 1
                improvement_payload["improved"] = False

            numeric_scores = {cid: data["scores"] for cid, data in scored.items()}

            if stall_limit_val and stalls >= stall_limit_val:
                break

        front = pareto_front(numeric_scores, criteria_names)
        pareto_ctx = format_pareto_context(front, scored, criteria_names)

    improved = ""
    for item in reversed(improvements):
        text = str(item.get("text") or "").strip()
        if text:
            improved = text
            break

    return {
        "scores": scored,
        "front": front,
        "pareto_context": pareto_ctx,
        "improved": improved,
        "improvements": improvements,
        "max_iterations": max_iterations_val,
        "stall_tolerance": stall_limit_val,
        "criteria_names": criteria_names,
        "criteria": normalized_criteria,
        "rubric_text": rubric_text,
        "model_used": model_requested,
        "model_identifier": model_identifier,
    }
