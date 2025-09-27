import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import app as flask_app  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

    with flask_app.app.test_client() as client:
        yield client


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Pareto DSPy Loop" in response.data
    assert b"Evaluation Criteria" in response.data


def test_evaluate_flow(monkeypatch, client):
    # Stub evaluation to avoid real LLM calls.
    def fake_evaluate(problem, candidates, criteria, model=None, **_kwargs):
        scores = {}
        for cid, artifact in candidates.items():
            scores[cid] = {
                "scores": {"Impact": 0.8, "Speed": 0.5},
                "justifications": {
                    "Impact": f"High impact for {cid}",
                    "Speed": f"Moderate speed for {cid}",
                },
            }

        return {
            "scores": scores,
            "front": list(candidates.keys()),
            "pareto_context": "Pareto Front:\n- " + "\n- ".join(candidates.keys()),
            "improved": "title: Better cand\nmetrics: {}",
            "criteria_names": ["Impact", "Speed"],
            "criteria": criteria,
            "model_used": model,
            "model_identifier": model,
        }

    monkeypatch.setattr(flask_app, "evaluate_candidates", fake_evaluate)

    response = client.post(
        "/evaluate",
        data={
            "problem": "Test problem",
            "model": "openai/gpt-4o-mini",
            "candidate_texts": ["title: Candidate A"],
            "criterion_names": ["Impact", "Speed"],
            "criterion_descs": ["Value", "Time"],
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    body = response.data.decode()
    assert "Candidate Scores" in body
    assert "Pareto Front" in body
    assert "LLM Improvements" in body
    assert "Model" in body
