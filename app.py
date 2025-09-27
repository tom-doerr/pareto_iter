from __future__ import annotations

import os
import re
import textwrap
from typing import Dict, List, Tuple

from flask import Flask, render_template, request

from pareto_dspy import evaluate_candidates, get_default_criteria

app = Flask(__name__)

DEFAULT_MODEL = "google/gemini-2.5-flash"
SUGGESTED_MODELS = [
    DEFAULT_MODEL,
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "mistral/mistral-large",
]


def _make_fake_evaluation(
    problem: str,
    candidates: Dict[str, str],
    criteria: List[Dict[str, str]],
    model: str | None,
    max_iterations: int,
    stall_limit: int,
) -> Dict[str, object]:
    """Return deterministic evaluation output for Playwright e2e tests."""

    criteria_names = [item["name"] for item in criteria]
    scores = {}
    for idx, (cid, artifact) in enumerate(candidates.items(), start=1):
        artifact_label = artifact.split("\n", 1)[0] or cid
        scores[cid] = {
            "scores": {name: min(1.0, 0.65 + 0.03 * idx) for name in criteria_names},
            "justifications": {
                name: f"Stub justification for {artifact_label}"
                for name in criteria_names
            },
        }

    improvement_text = "title: Synthetic improvement\nmetrics: {}"
    improvements = [
        {
            "id": "improved_iter_1",
            "text": improvement_text,
            "scores": {name: 0.82 for name in criteria_names},
            "justifications": {name: "Stub improvement" for name in criteria_names},
            "improved": True,
        }
    ]

    front_ids = list(candidates.keys())
    pareto_ctx = "Pareto Front:\n" + "\n".join(
        f"- {cid}: " + ", ".join(f"{name}=0.75" for name in criteria_names)
        for cid in front_ids
    )

    return {
        "scores": scores,
        "front": front_ids,
        "pareto_context": pareto_ctx,
        "improved": improvement_text,
        "improvements": improvements,
        "max_iterations": max_iterations,
        "stall_tolerance": stall_limit,
        "criteria_names": criteria_names,
        "criteria": criteria,
        "rubric_text": "",
        "model_used": model,
        "model_identifier": model,
    }

SAMPLE_PROBLEM = (
    "Design a small, rugged outdoor sensor puck that must be sealed (IP67), "
    "rechargeable, and inexpensive."
)

SAMPLE_CANDIDATES = [
    {
        "text": textwrap.dedent(
            """
            title: Sealed puck with USB-C port
            assumptions:
              - enclosure: sealed with a gasketed door for the port
            key_decisions:
              - charging: USB-C 5V/3A behind covered flap
              - materials: ABS + silicone seal
            risks:
              - ingress via port flap wear
              - user friction: open flap to charge
            metrics (estimates):
              - bom_usd: 16.5
              - charge_time_min: 120
              - failure_rate_%: 2.0
              - assembly_steps: 6
            """
        ).strip(),
    },
    {
        "text": textwrap.dedent(
            """
            title: Sealed puck with Qi2 wireless charging
            assumptions:
              - enclosure: fully sealed, no external connectors
            key_decisions:
              - charging: Qi2 magnetic 15W
              - materials: PC + TPU overmold, thin base under coil
            risks:
              - coil heating at 40C ambient
              - pad alignment on thick walls
            metrics (estimates):
              - bom_usd: 18.9
              - charge_time_min: 170
              - failure_rate_%: 1.2
              - assembly_steps: 5
            """
        ).strip(),
    },
]


def _extract_candidate_label(artifact_text: str) -> str:
    """Return a label based on the artifact content."""

    for raw_line in artifact_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("title:"):
            candidate_title = stripped.split(":", 1)[1].strip()
            if candidate_title:
                return candidate_title
            continue
        return stripped
    return ""


_slug_pattern = re.compile(r"[^a-z0-9]+")


def _slugify_label(label: str) -> str:
    slug = _slug_pattern.sub("-", label.lower()).strip("-")
    return slug


def _derive_candidate_identity(
    index: int, artifact_text: str, used_ids: set[str]
) -> Tuple[str, str]:
    label = _extract_candidate_label(artifact_text) or f"Candidate {index}"
    slug = _slugify_label(label)
    if not slug:
        slug = f"candidate-{index}"
    base = slug
    counter = 2
    while slug in used_ids:
        slug = f"{base}-{counter}"
        counter += 1
    used_ids.add(slug)
    return slug, label


@app.route("/", methods=["GET"])
def index() -> str:
    default_model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    return render_template(
        "index.html",
        problem=SAMPLE_PROBLEM,
        candidates=SAMPLE_CANDIDATES,
        criteria=get_default_criteria(),
        model_value=default_model,
        model_options=SUGGESTED_MODELS,
        max_iters_value=1,
        stall_limit_value=1,
    )


@app.route("/criterion-field", methods=["POST"])
def criterion_field() -> str:
    current_count = int(request.form.get("criterion_count", 0))
    next_index = current_count + 1
    return render_template("_criterion_field_response.html", index=next_index)


@app.route("/candidate-field", methods=["POST"])
def candidate_field() -> str:
    current_count = int(request.form.get("candidate_count", 0))
    next_index = current_count + 1
    return render_template("_candidate_field_response.html", index=next_index)


@app.route("/evaluate", methods=["POST"])
def evaluate() -> str:
    problem = request.form.get("problem", "").strip()
    artifacts: List[str] = request.form.getlist("candidate_texts")
    criterion_names: List[str] = request.form.getlist("criterion_names")
    criterion_descs: List[str] = request.form.getlist("criterion_descs")
    model_choice = request.form.get("model", "").strip()
    max_iters_raw = request.form.get("max_iters", "").strip()
    stall_limit_raw = request.form.get("stall_limit", "").strip()

    try:
        max_iterations = int(max_iters_raw)
    except ValueError:
        max_iterations = 1

    if max_iterations < 1:
        max_iterations = 1
    elif max_iterations > 5:
        max_iterations = 5

    try:
        stall_limit = int(stall_limit_raw)
    except ValueError:
        stall_limit = 1

    if stall_limit < 0:
        stall_limit = 0
    elif stall_limit > max_iterations:
        stall_limit = max_iterations

    collected: Dict[str, str] = {}
    display_names: Dict[str, str] = {}
    used_ids: set[str] = set()
    for idx, artifact in enumerate(artifacts, start=1):
        artifact_text = artifact.strip()
        if not artifact_text:
            continue
        candidate_id, label = _derive_candidate_identity(idx, artifact_text, used_ids)
        collected[candidate_id] = artifact_text
        display_names[candidate_id] = label

    criteria_payload = []
    for idx, crit_name in enumerate(criterion_names):
        desc = criterion_descs[idx] if idx < len(criterion_descs) else ""
        criteria_payload.append({"name": crit_name.strip(), "desc": desc.strip()})

    if not problem or not collected:
        return render_template(
            "_results.html",
            error="Provide a problem statement and at least one complete candidate.",
        )

    try:
        if os.getenv("PARETO_FAKE_EVALUATION"):
            evaluation = _make_fake_evaluation(
                problem,
                collected,
                criteria_payload,
                model_choice or None,
                max_iterations,
                stall_limit,
            )
        else:
            evaluation = evaluate_candidates(
                problem,
                collected,
                criteria_payload,
                model_choice or None,
                max_iterations=max_iterations,
                stall_tolerance=stall_limit,
            )
    except Exception as exc:  # pragma: no cover - defensive surface for LLM errors
        return render_template(
            "_results.html",
            error=f"Evaluation failed: {exc}",
        )

    scored = evaluation["scores"]
    ordered_results = []
    for cid, artifact_text in collected.items():
        payload = scored.get(cid)
        if not payload:
            continue
        ordered_results.append(
            {
                "id": cid,
                "label": display_names.get(cid, cid),
                "artifact": artifact_text,
                "scores": payload.get("scores", {}),
                "justifications": payload.get("justifications", {}),
            }
        )

    improvements = evaluation.get("improvements", [])
    iteration_count = len(improvements)
    max_iterations_used = evaluation.get("max_iterations")
    stall_tolerance_used = evaluation.get("stall_tolerance")
    non_improved_total = sum(1 for item in improvements if not item.get("improved"))
    trailing_no_gain = 0
    for item in reversed(improvements):
        if item.get("improved"):
            break
        trailing_no_gain += 1

    return render_template(
        "_results.html",
        results=ordered_results,
        criteria_names=evaluation["criteria_names"],
        criteria_info=evaluation["criteria"],
        front_ids=evaluation["front"],
        pareto_context=evaluation["pareto_context"],
        improved=evaluation["improved"],
        improvements=improvements,
        iteration_count=iteration_count,
        max_iterations=max_iterations_used,
        stall_tolerance=stall_tolerance_used,
        non_improved_total=non_improved_total,
        trailing_no_gain=trailing_no_gain,
        model_used=evaluation.get("model_used"),
        model_identifier=evaluation.get("model_identifier"),
    )


if __name__ == "__main__":
    app.run(debug=True)
