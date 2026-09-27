# review.py
import os
import sys
from dotenv import load_dotenv
import uuid
from src.llm_provider import LLMInvocationError

# 1. Environment Setup (Must be at the very top!)
load_dotenv()
from src.llm_provider import validate_provider_config
from src.graph import app

def main():

    try:
        validate_provider_config()
    except (EnvironmentError, ValueError) as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
        
    # 1. Check if the user typed '--all' in the terminal
    force_all = "--all" in sys.argv
    
    # 2. Clean the target_repo string by removing '--all' from the arguments list
    args = [arg for arg in sys.argv[1:] if arg != "--all"]
    target_repo = args[0] if args else "./"
    
    print(f"🚀 Starting Code Review Pipeline for: {target_repo}\n")

    initial_state = {
        "repo_path": target_repo,
        "force_all_workers": force_all  # Uses our new CLI flag!
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if not force_all:
        print("🤖 Orchestrator is analyzing the repository...")
    
    # Run up to the breakpoint
    try:
        app.invoke(initial_state, config=config)
    except LLMInvocationError as e:
        print(f"\n❌ Pipeline failed while contacting the LLM provider: {e}")
        sys.exit(1)
    

    # 3. NEW LOGIC: Only show the prompt if we didn't use the --all flag!
    if force_all:
        print("⚡ '--all' flag detected: Bypassing AI plan and human pause. Running all workers!")
    else:
        current_state = app.get_state(config).values
        proposed_decisions = current_state.get("worker_decisions", [])

        print("\n🛑 PAUSED: Orchestrator's Proposed Plan:")
        for d in proposed_decisions:
            status = "✅ RUN" if d.should_run else "❌ SKIP"
            print(f" - {d.worker_name}: {status} (Reason: {d.reason})")

        # The Human-in-the-Loop Override
        user_input = input("\nPress Enter to approve, or type a comma-separated list of tools to run instead (e.g., pylint, bandit): ").strip()

        if user_input:
            chosen_tools = [t.strip().lower() for t in user_input.split(",")]
            for d in proposed_decisions:
                d.should_run = d.worker_name in chosen_tools
            app.update_state(config, {"worker_decisions": proposed_decisions})
            print(f"\n🔄 State updated! Running custom tools: {chosen_tools}")
        else:
            print("\n✅ Plan approved! Resuming execution...")

    print("⚙️ Workers and Synthesizer are running...\n")
    
    # Resume the graph
    try:
        final_state = app.invoke(None, config=config)
    except LLMInvocationError as e:
        print(f"\n❌ Pipeline failed while contacting the LLM provider: {e}")
        sys.exit(1)

    print("✅ Review Complete!\n")
    report = final_state.get("final_report")
    if report:
        print(report)
    else:
        print("⚠️ No workers were run, so no findings were collected and no report was generated.")

if __name__ == "__main__":
    main()





   