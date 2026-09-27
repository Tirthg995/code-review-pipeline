# src/nodes/synthesizer.py
from langchain_core.prompts import ChatPromptTemplate
from src.state import State
from src.llm_provider import get_llm, invoke_with_retry

def synthesizer_node(state: State) -> dict:
    issues = state.get("issues", [])
    
    if not issues:
        return {"final_report": "🎉 Great news! No style, logic, or security issues were found in the codebase."}

    issues_text = "\n".join(
        [f"- [{i.severity.value.upper()}] {i.source.value} in {i.file} (Line {i.line}): {i.message}. Suggestion: {i.suggestion}" for i in issues]
    )

    feedback = state.get("reviewer_feedback")

    llm = get_llm(temperature=0.0)

    system_prompt = """You are a Senior QA Engineer. 
        Your task is to review the provided list of static analysis findings and generate a clean, structured Markdown report.
        
        CRITICAL RULES:
        - You must ONLY report on the exact issues provided in the input list. 
        - Do not invent, hallucinate, or assume any other bugs exist. 
        - Categorize the findings by severity.
        - If the list is empty, simply state that no issues were found."""

    if feedback:
        system_prompt += f"""

        IMPORTANT: A previous version of this report was reviewed and REJECTED.
        Reviewer feedback to address in this revision:
        {feedback}
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here are the raw findings to synthesize:\n\n{issues_list}")
    ])
    
    chain = prompt | llm
    response = invoke_with_retry(chain, {"issues_list": issues_text})
    
    return {"final_report": response.content}
