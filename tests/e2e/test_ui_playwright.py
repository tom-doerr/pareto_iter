import os
import socket
import subprocess
import time
import urllib.request

import pytest
from playwright.sync_api import expect, sync_playwright

CHROMIUM_EXECUTABLE = os.environ.get("PLAYWRIGHT_CHROMIUM", "/snap/bin/chromium")

BASE_PORT = int(os.environ.get("PLAYWRIGHT_TEST_PORT", "5051"))


@pytest.fixture(scope="session")
def start_server():
    # Verify the environment permits binding to a TCP socket; otherwise skip.
    try:
        probe = socket.socket()
    except PermissionError:
        pytest.skip("Environment forbids opening sockets; skipping Playwright e2e test.")
    else:
        probe.close()

    env = os.environ.copy()
    env.setdefault("FLASK_APP", "app.py")
    env.setdefault("OPENROUTER_API_KEY", "dummy-key")
    env.setdefault("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    env.setdefault("PARETO_FAKE_EVALUATION", "1")

    command = [
        "flask",
        "run",
        "--port",
        str(BASE_PORT),
        "--no-debugger",
        "--no-reload",
    ]

    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    url = f"http://127.0.0.1:{BASE_PORT}/"
    for _ in range(120):
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Flask server failed to start. stdout={stdout.decode()} stderr={stderr.decode()}")
        try:
            with urllib.request.urlopen(url, timeout=1):
                break
        except Exception:
            time.sleep(0.5)
    else:
        process.terminate()
        process.wait(timeout=10)
        raise TimeoutError("Flask server did not become ready in time")

    yield url

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def test_user_can_submit_evaluation(start_server):
    if not os.path.exists(CHROMIUM_EXECUTABLE):
        pytest.skip(f"Chromium executable not found at {CHROMIUM_EXECUTABLE}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=CHROMIUM_EXECUTABLE,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page()
        page.set_default_timeout(10_000)

        page.goto(start_server)
        expect(page.get_by_role("heading", name="Pareto DSPy Loop")).to_be_visible()

        page.fill("#problem", "Playwright end-to-end scenario")
        page.fill("#model", "openai/gpt-4o-mini")

        page.get_by_role("button", name="+ Add Criterion").click()
        new_name = page.locator('input[name="criterion_names"]').last()
        new_desc = page.locator('textarea[name="criterion_descs"]').last()
        new_name.fill("Delight")
        new_desc.fill("Measures user wow factor.")

        first_artifact = page.locator('#candidate-fields textarea').first()
        first_artifact.fill(
            "title: Candidate Alpha\n"
            "assumptions:\n  - lab-test\n"
            "metrics:\n  - bom_usd: 10\n  - cycle_time_min: 30"
        )

        page.get_by_role("button", name="Evaluate Pareto Loop").click()

        results_panel = page.locator("#results")
        expect(results_panel.get_by_text("Candidate Scores")).to_be_visible()
        expect(results_panel.get_by_text("Pareto Front")).to_be_visible()
        expect(results_panel.get_by_text("LLM Improvements")).to_be_visible()
        expect(results_panel.get_by_text("Model:")).to_be_visible()
        expect(results_panel.get_by_text("Synthetic improvement")).to_be_visible()

        browser.close()
