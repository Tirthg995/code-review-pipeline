# src/graph.py
from langgraph.graph import StateGraph, START, END
from src.nodes.evaluator import evaluator_node
from src.state import State
from src.nodes.orchestrator import orchestrator_node
from src.nodes.workers import pylint_worker, bandit_worker, radon_worker
from src.nodes.synthesizer import synthesizer_node
from src.nodes.router import route_to_workers
from langgraph.checkpoint.memory import MemorySaver

def route_after_evaluation(state: State) -> str:
    """
    Decides whether to finish the graph or loop back to the synthesizer.
    """
    if state.get("is_report_approved"):
        return END
    
    # The Circuit Breaker you asked about!
    if state.get("revision_count", 0) < 3:
        return "synthesizer"
    
    return END



# Initialize the graph with our shared state blueprint
workflow = StateGraph(State)

# 1. Add all our nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("pylint_worker", pylint_worker)
workflow.add_node("bandit_worker", bandit_worker)
workflow.add_node("radon_worker", radon_worker)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("evaluator", evaluator_node)

# 2. Define the exact flow (Edges)
workflow.add_edge(START, "orchestrator")

# 3. The Dynamic Fan-Out
# This conditional edge uses our router to fire off the Send() objects to the workers
workflow.add_conditional_edges(
    "orchestrator", 
    route_to_workers,
    ["pylint_worker", "bandit_worker", "radon_worker"]
)

# 4. The Fan-In
workflow.add_edge("pylint_worker", "synthesizer")
workflow.add_edge("bandit_worker", "synthesizer")
workflow.add_edge("radon_worker", "synthesizer")

# 5. The Finish Line
workflow.add_edge("synthesizer", "evaluator")

# Evaluator uses the router to decide if it loops back (< 3) or ends
workflow.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {"synthesizer": "synthesizer", END: END}
)
memory = MemorySaver() # MemorySaver is a checkpoint saver that saves the state of the graph to memory

# Finally, compile the graph into a runnable app
app = workflow.compile(
    checkpointer = memory,
    interrupt_after = ["orchestrator"]
)

