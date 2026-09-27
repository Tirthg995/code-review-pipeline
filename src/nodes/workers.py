import os
import subprocess
import json
from src.schemas import Issue, IssueSource, Severity

PYLINT_TIMEOUT_SECONDS = 300
BANDIT_TIMEOUT_SECONDS = 120
RADON_TIMEOUT_SECONDS = 120

# Bandit reports its own severity per finding ("LOW"/"MEDIUM"/"HIGH") —
# map it directly instead of hardcoding every finding to the same level.
BANDIT_SEVERITY_MAP = {
    "LOW": Severity.LOW,
    "MEDIUM": Severity.MEDIUM,
    "HIGH": Severity.HIGH,
}


def _relative_path(repo_path: str, reported_path: str) -> str:
    """
    Tool output reports paths as given on the command line. Since we pass an
    absolute repo_path (the cloned temp dir), strip that prefix so reports
    show clean paths like 'src/metrics.py' instead of leaking the server's
    temp directory layout.
    """
    try:
        return os.path.relpath(reported_path, repo_path)
    except ValueError:
        # Different drives on Windows, or some other non-relative case —
        # fall back to whatever the tool reported rather than crashing.
        return reported_path


def _radon_severity(complexity: int) -> Severity:
    """
    Maps radon's numeric cyclomatic complexity score onto our Severity scale,
    following radon's own rank thresholds (A/B = simple, C/D = moderate,
    E/F = high complexity) collapsed into our four-level enum.
    """
    if complexity <= 10:
        return Severity.LOW
    elif complexity <= 20:
        return Severity.MEDIUM
    elif complexity <= 30:
        return Severity.HIGH
    else:
        return Severity.CRITICAL


def pylint_worker(payload: dict) -> dict:
    """
    Runs pylint on the provided repository path and returns issues.
    """
    repo_path = payload.get("repo_path")
    issues = []

    if not repo_path:
        return {"issues": issues}

    try:
        result = subprocess.run(
            [
                 "pylint",
                 repo_path,
                 "--output-format=json",
                 "--jobs=0",
                 "--ignore=.venv,venv,.git,.pytest_cache,__pycache__,.vscode",
                 "--ignore-patterns=^\\.venv$,^venv$,^\\.git$,^__pycache__$",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=PYLINT_TIMEOUT_SECONDS,
    )
    except FileNotFoundError:
        print("pylint executable not found. Is it installed?")
        return {"issues": issues}
    except subprocess.TimeoutExpired:
        print(f"Pylint timed out after {PYLINT_TIMEOUT_SECONDS}s.")
        return {"issues": issues}

    if result.stdout:
        try:
            parsed_data = json.loads(result.stdout)
            for item in parsed_data:
                issue = Issue(
                    source=IssueSource.STYLE,
                    severity=Severity.LOW,
                    file=_relative_path(repo_path, item.get("path", "unknown")),
                    line=item.get("line"),
                    message=item.get("message", "No message provided"),
                    suggestion=f"Review pylint rule: {item.get('symbol', 'unknown')}"
                )
                issues.append(issue)
        except json.JSONDecodeError:
            print("Failed to parse Pylint output.")

    return {"issues": issues}


def bandit_worker(payload: dict) -> dict:
    """
    Runs bandit on the provided repository path and returns security issues.
    """
    repo_path = payload.get("repo_path")
    issues = []

    if not repo_path:
        return {"issues": issues}

    try:
        result = subprocess.run(
            ["bandit", "-r", repo_path, "-f", "json", "-x", "venv,.venv,__pycache__"],
            capture_output=True,
            text=True,
            check=False,
            timeout=BANDIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        print("bandit executable not found. Is it installed?")
        return {"issues": issues}
    except subprocess.TimeoutExpired:
        print(f"Bandit timed out after {BANDIT_TIMEOUT_SECONDS}s.")
        return {"issues": issues}

    if result.stdout:
        try:
            parsed_data = json.loads(result.stdout)
            for item in parsed_data.get("results", []):
                severity = BANDIT_SEVERITY_MAP.get(
                    str(item.get("issue_severity", "")).upper(),
                    Severity.MEDIUM,
                )
                issue = Issue(
                    source=IssueSource.SECURITY,
                    severity=severity,
                    file=_relative_path(repo_path, item.get("filename", "unknown")),
                    line=item.get("line_number"),
                    message=item.get("issue_text"),
                    suggestion=f"Review security findings: {item.get('test_id', 'unknown')}"
                )
                issues.append(issue)
        except json.JSONDecodeError:
            print("Failed to parse Bandit output.")

    return {"issues": issues}


def radon_worker(payload: dict) -> dict:
    """
    Runs radon on the provided repository path to check for code complexity.
    """
    repo_path = payload.get("repo_path")
    issues = []

    if not repo_path:
        return {"issues": issues}

    try:
        result = subprocess.run(
            ["radon", "cc", repo_path, "-j"],
            capture_output=True,
            text=True,
            check=False,
            timeout=RADON_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        print("radon executable not found. Is it installed?")
        return {"issues": issues}
    except subprocess.TimeoutExpired:
        print(f"Radon timed out after {RADON_TIMEOUT_SECONDS}s.")
        return {"issues": issues}

    if result.stdout:
        try:
            parsed_data = json.loads(result.stdout)
            for filepath, blocks in parsed_data.items():
                for block in blocks:
                    complexity = block.get("complexity", 0)
                    issue = Issue(
                        source=IssueSource.LOGIC,
                        severity=_radon_severity(complexity),
                        file=_relative_path(repo_path, filepath),
                        line=block.get("lineno"),
                        message=f"Complexity score: {complexity}",
                        suggestion="Consider refactoring to reduce complexity."
                    )
                    issues.append(issue)
        except json.JSONDecodeError:
            print("Failed to parse Radon output.")

    return {"issues": issues}