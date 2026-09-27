from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.state import State
from src.schemas import WorkerDecision
from src.llm_provider import get_llm, invoke_with_retry


# Helper schema to force the LLM to return a list of decisions
class OrchestratorOutput(BaseModel):
    decisions: List[WorkerDecision] = Field(
        description="List of decisions for each available worker (pylint, bandit, radon)"
    )

def orchestrator_node(state: State) -> dict:
    """
    Analyzes the repository context and decides which specialist workers to run.
    """
    # 1. Check for the override flag
    if state.get("force_all_workers"):
        # we return all 3 tools set to TRUE, bypassing the LLM
        return { 
            "worker_decisions":[
                WorkerDecision(worker_name="pylint", reason="Forced via --all flag", should_run=True),
                WorkerDecision(worker_name="bandit", reason="Forced via --all flag", should_run=True),
                WorkerDecision(worker_name="radon", reason="Forced via --all flag", should_run=True),
            ]
        }

    # 2. Dynamic routing via LLM
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(OrchestratorOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Code Review Orchestrator. 
        Your job is to decide which static analysis tools to run based on the 
        repository path provided.
        
        The available tools are:
        - pylint: Python style/PEP8
        - bandit: Security vulnerabilities
        - radon: Code complexity/maintainability
        You must evaluate the context and return a decision for ALL THREE tools.
        """),
        ("human", "Evaluate the tools needed for this repository path: {repo_path}")
    ])
    chain = prompt | structured_llm
    
    # 3. Invocation and State Update
    # We extract the repo_path from the state and pass it into the prompt variables
    result = invoke_with_retry(chain, {"repo_path": state.get("repo_path")})
    
    # 4. Validate the LLM returned a decision for all three tools; backfill any it missed.
    expected_workers = {"pylint", "bandit", "radon"}
    decisions = list(result.decisions)
    seen = {d.worker_name for d in decisions}
    for missing in expected_workers - seen:
        decisions.append(
            WorkerDecision(
                worker_name=missing,
                should_run=True,
                reason="Backfilled: orchestrator LLM did not return a decision for this tool.",
            )
        )

    # We return a dictionary mapping to our State's 'worker_decisions' key.
    # result.decisions contains the strictly typed List[WorkerDecision] from the LLM.
    return {"worker_decisions": result.decisions}


