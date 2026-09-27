from typing import List, Union
from langgraph.graph import END
from src.state import State
from langgraph.types import Send

def route_to_workers(state: State) -> Union[List[Send], str]:
    """
    Reads the worker_decisions from the state and dynamically 
    dispatches the approved workers in parallel using the Send API.
    """
    decisions = state.get("worker_decisions", [])
    sends = []

    for decision in decisions:
        if decision.should_run:
            if decision.worker_name == "pylint":
                sends.append(Send("pylint_worker", {"repo_path": state.get("repo_path")}))
            elif decision.worker_name == "bandit":
                sends.append(Send("bandit_worker", {"repo_path": state.get("repo_path")}))
            elif decision.worker_name == "radon":
                sends.append(Send("radon_worker", {"repo_path": state.get("repo_path")}))

    if not sends:
        # No workers selected — nothing to fan out to.
        # Returning END here short-circuits the graph cleanly instead of
        # silently skipping synthesizer/evaluator.
        return END

    return sends

