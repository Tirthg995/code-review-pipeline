from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from src.nodes import synthesizer as synthesizer_module
from src.nodes.synthesizer import synthesizer_node
from src.schemas import Issue, IssueSource, Severity


def _make_issue():
    return Issue(
        source=IssueSource.SECURITY,
        severity=Severity.HIGH,
        file="app.py",
        line=1,
        message="hardcoded secret",
        suggestion="use env vars",
    )


def test_synthesizer_short_circuits_on_no_issues(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("LLM should not be called when there are no issues")

    monkeypatch.setattr(synthesizer_module, "get_llm", explode)

    result = synthesizer_node({"issues": []})
    assert "no" in result["final_report"].lower()


def test_synthesizer_generates_report_from_issues(monkeypatch):
    captured = {}

    def fake_llm_factory(temperature=0.0):
        def _invoke(prompt_value):
            captured["prompt_text"] = prompt_value.to_string()
            return SimpleNamespace(content="## Fake Report\n- HIGH: hardcoded secret")
        return RunnableLambda(_invoke)

    monkeypatch.setattr(synthesizer_module, "get_llm", fake_llm_factory)

    result = synthesizer_node({"issues": [_make_issue()]})
    assert "Fake Report" in result["final_report"]
    assert "hardcoded secret" in captured["prompt_text"]


def test_synthesizer_includes_reviewer_feedback_on_retry(monkeypatch):
    """Regression test for the bug where a rejected report was
    regenerated without ever seeing the evaluator's feedback."""
    captured = {}

    def fake_llm_factory(temperature=0.0):
        def _invoke(prompt_value):
            captured["prompt_text"] = prompt_value.to_string()
            return SimpleNamespace(content="revised report")
        return RunnableLambda(_invoke)

    monkeypatch.setattr(synthesizer_module, "get_llm", fake_llm_factory)

    state = {
        "issues": [_make_issue()],
        "reviewer_feedback": "You forgot to categorize by severity.",
    }
    synthesizer_node(state)

    assert "You forgot to categorize by severity." in captured["prompt_text"]