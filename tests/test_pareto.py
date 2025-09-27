import json
import sys
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pareto_dspy  # noqa: E402


def stub_dspy(monkeypatch, score_map, improvement_texts):
    """Patch DSPy helpers so evaluation runs deterministically."""

    def fake_build_lm(model_name):
        requested = model_name or "google/gemini-2.5-flash"
        identifier = requested if "/" in requested else f"openrouter/{requested}"
        return SimpleNamespace(model=identifier), requested, identifier

    monkeypatch.setattr(pareto_dspy, "_build_lm", fake_build_lm)

    class DummyContext:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pareto_dspy.dspy, "context", lambda **kwargs: DummyContext(**kwargs))

    improvement_iter = iter(improvement_texts)

    def fake_predict(signature_cls):
        if signature_cls is pareto_dspy.ScoreArtifactSig:
            def scorer(**kwargs):
                artifact = kwargs["artifact"].strip()
                raw_scores = score_map.get(artifact, {})
                payload_scores = {k: float(v) for k, v in raw_scores.items()}
                payload_justifications = {k: f"{artifact} -> {k}" for k in payload_scores}
                return SimpleNamespace(scores_json=json.dumps({
                    "scores": payload_scores,
                    "justifications": payload_justifications,
                }))

            return scorer

        if signature_cls is pareto_dspy.ImproveFromParetoSig:
            counter = count(1)

            def improver(**kwargs):
                try:
                    text = next(improvement_iter)
                except StopIteration:
                    text = ""
                # Ensure different text instances don't break strip() handling.
                return SimpleNamespace(improved_yaml=str(text))

            return improver

        raise AssertionError("Unexpected signature")

    monkeypatch.setattr(pareto_dspy.dspy, "Predict", fake_predict)


def test_normalize_criteria_deduplicates_and_trims():
    criteria = [
        {"name": " Impact ", "desc": "Primary"},
        {"name": "Impact", "desc": "Duplicate"},
        {"name": "", "desc": "Ignored"},
    ]

    normalized = pareto_dspy._normalize_criteria(criteria)

    assert normalized == [{"name": "Impact", "desc": "Primary"}]


def test_build_rubric_text_appends_guidance():
    rubric = pareto_dspy.build_rubric_text([
        {"name": "Impact", "desc": "Value created."},
    ])

    assert "Impact: Value created." in rubric
    assert "Calibrate tightly" in rubric
    assert rubric.endswith("lower confidence.")


def test_evaluate_candidates_with_stubbed_lm(monkeypatch):
    score_map = {
        "title: Candidate A": {"Impact": 0.8, "Simplicity": 0.4},
        "title: Candidate B": {"Impact": 0.6, "Simplicity": 0.7},
        "title: Improved\nmetrics: {}": {"Impact": 0.7, "Simplicity": 0.75},
    }
    stub_dspy(monkeypatch, score_map, ["title: Improved\nmetrics: {}"])

    candidates = {
        "cand_a": "title: Candidate A",
        "cand_b": "title: Candidate B",
    }
    criteria = [
        {"name": "Impact", "desc": "How much value is delivered."},
        {"name": "Simplicity", "desc": "Operational ease."},
    ]

    evaluation = pareto_dspy.evaluate_candidates(
        problem="Test problem",
        candidates=candidates,
        criteria=criteria,
        model="custom/provider-model",
        max_iterations=2,
    )

    assert {"cand_a", "cand_b"}.issubset(evaluation["scores"].keys())
    assert evaluation["improved"] == "title: Improved\nmetrics: {}"
    assert evaluation["model_used"] == "custom/provider-model"
    assert evaluation["model_identifier"] == "custom/provider-model"
    assert evaluation["criteria_names"] == ["Impact", "Simplicity"]
    assert evaluation["improvements"][0]["improved"] is True


def test_evaluate_candidates_respects_stall_tolerance(monkeypatch):
    score_map = {
        "title: Candidate A": {"Impact": 0.5, "Speed": 0.5},
        "title: Candidate B": {"Impact": 0.4, "Speed": 0.4},
        "title: better 1": {"Impact": 0.6, "Speed": 0.6},
        "title: better 2": {"Impact": 0.6, "Speed": 0.6},
    }
    stub_dspy(monkeypatch, score_map, ["title: better 1", "title: better 2", "title: better 3"])

    evaluation = pareto_dspy.evaluate_candidates(
        problem="Problem",
        candidates={"cand_a": "title: Candidate A", "cand_b": "title: Candidate B"},
        criteria=[
            {"name": "Impact", "desc": ""},
            {"name": "Speed", "desc": ""},
        ],
        max_iterations=5,
        stall_tolerance=1,
    )

    improvements = evaluation["improvements"]
    assert len(improvements) == 2  # Stops after one stall.
    assert improvements[0]["improved"] is True
    assert improvements[1]["improved"] is False
    assert evaluation["improved"] == "title: better 2"
    assert evaluation["max_iterations"] == 5
    assert evaluation["stall_tolerance"] == 1


def test_evaluate_candidates_stops_on_empty_improvement(monkeypatch):
    score_map = {
        "title: Candidate A": {"Impact": 0.6, "Speed": 0.6},
    }
    stub_dspy(monkeypatch, score_map, ["", "title: ignored"])

    evaluation = pareto_dspy.evaluate_candidates(
        problem="Problem",
        candidates={"cand_a": "title: Candidate A"},
        criteria=[{"name": "Impact", "desc": ""}, {"name": "Speed", "desc": ""}],
        max_iterations=3,
        stall_tolerance=2,
    )

    improvements = evaluation["improvements"]
    assert len(improvements) == 1
    assert improvements[0]["text"] == ""
    assert improvements[0]["improved"] is False
    assert "scores" not in improvements[0]
    assert evaluation["improved"] == ""
    assert set(evaluation["front"]) == {"cand_a"}


def test_evaluate_candidates_validates_inputs():
    with pytest.raises(ValueError):
        pareto_dspy.evaluate_candidates(" ", {"a": "artifact"})

    with pytest.raises(ValueError):
        pareto_dspy.evaluate_candidates("Problem", {})
