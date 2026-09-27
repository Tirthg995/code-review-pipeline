from langchain_core.runnables import RunnableLambda

from src.nodes import orchestrator as orchestrator_module
from src.nodes.orchestrator import orchestrator_node, OrchestratorOutput
from src.schemas import WorkerDecision


def test_orchestrator_force_all_bypasses_llm(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("LLM should not be called when force_all_workers is True")

    monkeypatch.setattr(orchestrator_module, "get_llm", explode)

    state = {"repo_path": "./", "force_all_workers": True}
    result = orchestrator_node(state)

    decisions = result["worker_decisions"]
    assert len(decisions) == 3
    assert all(d.should_run for d in decisions)
    assert {d.worker_name for d in decisions} == {"pylint", "bandit", "radon"}


def test_orchestrator_dynamic_routing_uses_llm_decision(monkeypatch):
    fake_decisions = OrchestratorOutput(
        decisions=[
            WorkerDecision(worker_name="pylint", reason="python repo", should_run=True),
            WorkerDecision(worker_name="bandit", reason="no security risk detected", should_run=False),
            WorkerDecision(worker_name="radon", reason="check complexity", should_run=True),
        ]
    )

    class FakeLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: fake_decisions)

    monkeypatch.setattr(orchestrator_module, "get_llm", lambda temperature=0.0: FakeLLM())

    state = {"repo_path": "./my-repo", "force_all_workers": False}
    result = orchestrator_node(state)

    decisions = result["worker_decisions"]
    assert decisions == fake_decisions.decisions