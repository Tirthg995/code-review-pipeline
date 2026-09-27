from langgraph.graph import END
from src.nodes.router import route_to_workers
from src.schemas import WorkerDecision


def test_route_to_workers_all_selected():
    state = {
        "repo_path": "./",
        "worker_decisions": [
            WorkerDecision(worker_name="pylint", reason="check style", should_run=True),
            WorkerDecision(worker_name="bandit", reason="check security", should_run=True),
            WorkerDecision(worker_name="radon", reason="check complexity", should_run=True),
        ],
    }
    sends = route_to_workers(state)
    node_names = {s.node for s in sends}
    assert node_names == {"pylint_worker", "bandit_worker", "radon_worker"}


def test_route_to_workers_partial_selection():
    state = {
        "repo_path": "./",
        "worker_decisions": [
            WorkerDecision(worker_name="pylint", reason="skip", should_run=False),
            WorkerDecision(worker_name="bandit", reason="run it", should_run=True),
            WorkerDecision(worker_name="radon", reason="skip", should_run=False),
        ],
    }
    sends = route_to_workers(state)
    assert len(sends) == 1
    assert sends[0].node == "bandit_worker"


def test_route_to_workers_none_selected_ends_gracefully():
    """Regression test for the bug where an empty fan-out silently
    skipped synthesizer/evaluator with no explanation."""
    state = {
        "repo_path": "./",
        "worker_decisions": [
            WorkerDecision(worker_name="pylint", reason="skip", should_run=False),
            WorkerDecision(worker_name="bandit", reason="skip", should_run=False),
            WorkerDecision(worker_name="radon", reason="skip", should_run=False),
        ],
    }
    result = route_to_workers(state)
    assert result == END


def test_route_to_workers_empty_decisions_list():
    state = {"repo_path": "./", "worker_decisions": []}
    result = route_to_workers(state)
    assert result == END