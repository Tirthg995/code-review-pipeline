# Code Review Pipeline

An agentic code review tool built with LangGraph and OpenAI. It runs static
analysis tools (pylint, bandit, radon) against a repository, has an AI
orchestrator decide which tools are worth running, lets a human review and
override that plan before it executes, then synthesizes the raw findings
into a structured Markdown report — which a second AI evaluator critiques
and can send back for revision before approving.

🔗 Live demo:https://code-review-pipeline-5h9epfotc-triff1.vercel.app (access code required — DM me for one)

## What makes this interesting

- **Dynamic multi-agent fan-out** — LangGraph's `Send` API dispatches
  pylint/bandit/radon workers in parallel based on the orchestrator's
  decision, not a fixed pipeline.
- **Human-in-the-loop approval** — the graph pauses after the orchestrator
  proposes a plan (`interrupt_after`) so a human can approve or override it
  before any worker runs, then resumes from that exact checkpoint.
- **Generate → critique → revise loop** — a synthesizer drafts the report,
  an evaluator checks it for structure and hallucination-free accuracy, and
  rejected reports get regenerated with the evaluator's specific feedback
  (with a circuit breaker capping retries at 3).
- **Provider-agnostic LLM layer** — swappable between OpenAI, WatsonX, and
  Groq via one environment variable, with centralized retry-with-backoff so
  a transient API failure doesn't crash the whole pipeline.
- **Two interfaces to the same graph** — a CLI (`review.py`) and a FastAPI
  backend (`api.py`) that exposes the same human-in-the-loop pause/resume
  as two HTTP endpoints, so the interactive terminal `input()` prompt
  becomes a real two-request web flow for a browser UI.

## Architecture

orchestrator → [pylint_worker, bandit_worker, radon_worker] (parallel)
→ synthesizer → evaluator ─┐
↑____________________┘ (loop up to 3x if rejected)


The graph pauses right after the orchestrator (`interrupt_after`), so a
human can inspect and edit `worker_decisions` in the checkpointed state
before the fan-out to workers happens.

## Running it

**CLI:**
```bash
pip install -r requirements.txt
python review.py ./path/to/repo          # interactive: prompts for plan approval
python review.py ./path/to/repo --all    # skips the AI plan, runs all three tools
```

**Web API + UI:**
```bash
uvicorn api:api --reload      # backend, http://127.0.0.1:8000
# in another terminal:
cd frontend && python3 -m http.server 5500   # frontend, http://localhost:5500
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:
- `LLM_PROVIDER` — `openai`, `watsonx`, or `groq`
- Provider-specific API key/model settings
- `ACCESS_CODE` — required by the API; every request must include a matching
  `X-Access-Code` header, so a public deployment can't have its LLM credits
  spent by strangers

## Testing

```bash
pytest -v
```
26 tests covering router logic, schema validation, worker output parsing
(subprocess calls mocked), and all three LLM-calling nodes (orchestrator,
synthesizer, evaluator — LLM calls mocked via LangChain's `RunnableLambda`,
so the suite runs in under a second with zero API cost and no dependency on
pylint/bandit/radon being installed).

## Notable bugs found and fixed during development

Several of these were caught by writing tests, not by manual testing —
included here because that's arguably the more interesting story than the
bugs themselves.

- **Synthesizer ignored evaluator feedback.** The generate→critique→revise
  loop looked correct structurally, but the synthesizer never actually read
  the evaluator's `reviewer_feedback` on a retry — it just regenerated the
  same report from the same input, up to 3 times, for no benefit. Fixed by
  injecting the feedback into the retry prompt.
- **Silent failure on empty tool selection.** If every worker was
  deselected (by the AI or a human override), LangGraph's `Send`-based
  fan-out received an empty list, which skipped the rest of the graph
  entirely with no report and no explanation. Fixed by routing to `END`
  explicitly and surfacing a clear message instead of a blank result.
- **Dead timeout-handling code.** `radon_worker` caught
  `subprocess.TimeoutExpired`, but the `subprocess.run()` call was never
  given a `timeout=` argument, so that handler could never trigger.
  `pylint`/`bandit` had no timeout protection at all. Fixed with per-tool
  timeouts (pylint needs more headroom than bandit/radon).
- **`--all` flag was broken on every run.** `WorkerDecision` requires a
  `reason` field with no default, but the `force_all_workers` code path
  constructed instances without one — a `ValidationError` on every
  `--all` invocation. Caught by a test, not by manual use.
- **A silently bloated pylint scan.** A `.venv` folder sitting in the
  project root was being recursively linted, turning a ~5 second scan into
  one that exceeded a 300-second timeout. Bandit's `-x` exclude flag fixed
  it immediately; pylint's `--ignore-patterns` (regex) did *not* reliably
  exclude the folder despite matching — switching to `--ignore` (exact
  basename match) plus `--jobs=0` for parallelism fixed it for real.

## Deployment notes

The backend needs a persistent, long-running process — it keeps graph
state in memory (`MemorySaver`) between the two-request human-in-the-loop
flow, can block for up to a few minutes on `pylint`/OpenAI calls, and shells
out to external CLI tools. That ruled out serverless platforms like Vercel
for the backend. Current split:
- **Frontend** (`frontend/index.html`) → Vercel (static hosting)
- **Backend** (`api.py`) → Render (or any host supporting long-running
  Python processes with shell access)

## What this doesn't do (yet)

The pipeline detects and categorizes issues but doesn't propose actual code
fixes — pylint/bandit's `suggestion` fields are rule-ID references (e.g.
"Review pylint rule: unused-import"), and the synthesizer LLM is
deliberately restricted to only report on the exact findings it's given,
never to invent or propose remediations. A natural next step would be a
`fixer` node that proposes a concrete diff per issue for human review,
without auto-applying it.
