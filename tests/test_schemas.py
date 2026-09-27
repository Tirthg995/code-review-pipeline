import pytest
from pydantic import ValidationError
from src.schemas import Issue, IssueSource, Severity, WorkerDecision


def test_issue_confidence_within_bounds():
    issue = Issue(
        source=IssueSource.SECURITY,
        severity=Severity.HIGH,
        file="app.py",
        line=10,
        message="hardcoded secret",
        suggestion="use env vars",
        confidence=0.9,
    )
    assert issue.confidence == 0.9


def test_issue_confidence_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        Issue(
            source=IssueSource.SECURITY,
            severity=Severity.HIGH,
            file="app.py",
            message="bad confidence",
            suggestion="n/a",
            confidence=1.5,  # invalid: must be <= 1.0
        )


def test_issue_confidence_optional():
    issue = Issue(
        source=IssueSource.STYLE,
        severity=Severity.LOW,
        file="app.py",
        message="line too long",
        suggestion="wrap it",
    )
    assert issue.confidence is None


def test_worker_decision_roundtrip():
    decision = WorkerDecision(worker_name="pylint", reason="style check", should_run=True)
    assert decision.worker_name == "pylint"
    assert decision.should_run is True