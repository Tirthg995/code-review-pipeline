# src/nodes/evaluator.py
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.state import State
from src.llm_provider import get_llm, invoke_with_retry

# 1. Define the Expected Output Schema
class EvaluatorOutput(BaseModel):

    is_report_approved: bool = Field(
        description="True if the report is perfectly structured and hallucination-free. False otherwise."
    )
    reviewer_feedback: str = Field(
        description="Specific feedback on what needs to change, or positive praise if approved."
    )

def evaluator_node(state: State) -> dict:
    """
    Evaluates the generated markdown report for quality and hallucinations.
    """
    report = state.get("final_report", "")

    # We grab the current count (defaulting to 0 if it's the first run)
    current_revision = state.get("revision_count", 0)

    # 2. Set up the LLM with our strictly typed schema
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(EvaluatorOutput)

    # 3. Create the evaluation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict Quality Assurance AI.
        Review the provided Markdown report for code analysis.
        Ensure it is professional, categorized by severity, and contains no hallucinations.
        If it passes all criteria, set is_report_approved to true and leave positive feedback.
        If it fails, set is_report_approved to false and provide specific feedback on what needs to change."""),
        ("human", "Review this report:\n\n{report}")
    ])

    # 4. Invocation
    chain = prompt | structured_llm
    result = invoke_with_retry(chain, {"report": report})

    # 5. Return the state update
    return {
        "is_report_approved": result.is_report_approved,
        "reviewer_feedback": result.reviewer_feedback,
        "revision_count": current_revision + 1 
    }