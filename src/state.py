import operator
from typing import Annotated, List, TypedDict

from src.schemas import Issue, WorkerDecision

class State(TypedDict):
    """
    Represents the state of the code review graph.
    """
    repo_path: str
    worker_decisions: List[WorkerDecision]
    issues: Annotated[List[Issue], operator.add]
    final_report: str
    force_all_workers: bool


# for evaluator node 
    reviewer_feedback: str # hold critique from evaluator 
    is_report_approved: bool # tells whether to finsh or loop back 
    revision_count: int # helps avoid infinite loop 





