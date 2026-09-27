import json
import subprocess
from types import SimpleNamespace

from src.nodes import workers


def _fake_completed_process(stdout):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_pylint_worker_no_repo_path():
    assert workers.pylint_worker({}) == {"issues": []}


def test_pylint_worker_parses_output(monkeypatch):
    fake_output = json.dumps([
        {"path": "app.py", "line": 5, "message": "unused variable", "symbol": "unused-variable"}
    ])

    def fake_run(*args, **kwargs):
        return _fake_completed_process(fake_output)

    monkeypatch.setattr(workers.subprocess, "run", fake_run)

    result = workers.pylint_worker({"repo_path": "./fake"})
    issues = result["issues"]
    assert len(issues) == 1
    assert issues[0].file == "app.py"
    assert issues[0].line == 5
    assert issues[0].source.value == "style"


def test_pylint_worker_handles_malformed_json(monkeypatch):
    def fake_run(*args, **kwargs):
        return _fake_completed_process("not valid json {{{")

    monkeypatch.setattr(workers.subprocess, "run", fake_run)

    result = workers.pylint_worker({"repo_path": "./fake"})
    assert result == {"issues": []}


def test_bandit_worker_parses_output(monkeypatch):
    fake_output = json.dumps({
        "results": [
            {"filename": "app.py", "line_number": 12, "issue_text": "hardcoded password", "test_id": "B105"}
        ]
    })

    def fake_run(*args, **kwargs):
        return _fake_completed_process(fake_output)

    monkeypatch.setattr(workers.subprocess, "run", fake_run)

    result = workers.bandit_worker({"repo_path": "./fake"})
    issues = result["issues"]
    assert len(issues) == 1
    assert issues[0].source.value == "security"
    assert issues[0].severity.value == "high"


def test_radon_worker_parses_output(monkeypatch):
    fake_output = json.dumps({
        "app.py": [
            {"lineno": 20, "complexity": 15}
        ]
    })

    def fake_run(*args, **kwargs):
        return _fake_completed_process(fake_output)

    monkeypatch.setattr(workers.subprocess, "run", fake_run)

    result = workers.radon_worker({"repo_path": "./fake"})
    issues = result["issues"]
    assert len(issues) == 1
    assert issues[0].file == "app.py"
    assert issues[0].source.value == "logic"


def test_radon_worker_handles_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="radon", timeout=120)

    monkeypatch.setattr(workers.subprocess, "run", fake_run)

    result = workers.radon_worker({"repo_path": "./fake"})
    assert result == {"issues": []}