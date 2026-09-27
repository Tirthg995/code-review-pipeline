from langchain_core.runnables import RunnableLambda

from src.nodes import evaluator as evaluator_module
from src.nodes.evaluator import evaluator_node, EvaluatorOutput


def test_evaluator_approves_report(monkeypatch):
    fake_result = EvaluatorOutput(is_report_approved=True, reviewer_feedback="Looks good.")

    class FakeLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: fake_result)

    monkeypatch.setattr(evaluator_module, "get_llm", lambda temperature=0.0: FakeLLM())

    result = evaluator_node({"final_report": "## Report\n- HIGH: issue", "revision_count": 0})

    assert result["is_report_approved"] is True
    assert result["reviewer_feedback"] == "Looks good."
    assert result["revision_count"] == 1


def test_evaluator_increments_revision_count_across_calls(monkeypatch):
    fake_result = EvaluatorOutput(is_report_approved=False, reviewer_feedback="Needs more detail.")

    class FakeLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: fake_result)

    monkeypatch.setattr(evaluator_module, "get_llm", lambda temperature=0.0: FakeLLM())

    result = evaluator_node({"final_report": "draft", "revision_count": 2})

    assert result["is_report_approved"] is False
    assert result["revision_count"] == 3