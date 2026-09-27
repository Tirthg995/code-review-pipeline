"""
api.py — HTTP interface for the code review pipeline, for the web UI.

This mirrors what review.py does interactively in a terminal, but split into
two HTTP requests instead of one blocking input() call:
  1. POST /reviews             -> clones the given GitHub repo into a private
                                   temp directory, runs the orchestrator,
                                   pauses, and returns the proposed worker
                                   plan (same pause point as review.py's
                                   interrupt_after=["orchestrator"])
  2. POST /reviews/{id}/resume -> applies any human override to the plan,
                                   resumes the graph to completion, and
                                   cleans up the cloned repo

The review_id doubles as the LangGraph thread_id — the checkpointer
(MemorySaver, in src/graph.py) uses it to find exactly where that specific
review was paused, in this server process's memory. It's also used to name
that review's temp clone directory, so concurrent reviews never collide.

Both endpoints require an X-Access-Code header matching ACCESS_CODE in .env,
so a public deployment can't have its OpenAI credits burned by strangers.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.graph import app as graph_app
from src.llm_provider import validate_provider_config, LLMInvocationError
from src.schemas import WorkerDecision


api = FastAPI(title="Code Review Pipeline API")

# Allow the Vercel-hosted frontend to call this API from the browser.
# Tighten allow_origins to your actual Vercel domain once you have it,
# instead of "*", before sharing this publicly.
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only allow cloning from github.com over https. This blocks someone from
# passing file:///etc, ssh://, or other schemes/hosts that `git clone` would
# otherwise happily try to read from the server's own environment.
GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+(?:\.git)?/?$"
)
CLONE_TIMEOUT_SECONDS = 60

# review_id -> path of that review's cloned temp directory, so resume_review
# can clean it up once the report is generated (or start_review can clean up
# after itself if the graph fails before getting that far).
_review_temp_dirs: dict[str, str] = {}


@api.on_event("startup")
def on_startup():
    # Same idea as review.py's startup check: fail immediately and clearly
    # if the LLM provider isn't configured, instead of failing deep inside
    # the first request that happens to hit it.
    validate_provider_config()
    if not os.environ.get("ACCESS_CODE"):
        raise EnvironmentError(
            "ACCESS_CODE is not set in .env — required to protect the API from public use."
        )


def verify_access_code(x_access_code: str = Header(None)):
    expected = os.environ.get("ACCESS_CODE")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: ACCESS_CODE is not set.",
        )
    if x_access_code != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing access code.")


def clone_repo(repo_url: str, review_id: str) -> str:
    """
    Clones repo_url into a fresh temp directory unique to this review_id and
    returns that directory's path. Raises HTTPException on any failure so
    callers can just let it propagate.
    """
    dest_dir = os.path.join(tempfile.gettempdir(), f"review-{review_id}")
    os.makedirs(dest_dir, exist_ok=True)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, dest_dir],
            capture_output=True,
            text=True,
            check=False,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(
            status_code=502,
            detail=f"Cloning the repository timed out after {CLONE_TIMEOUT_SECONDS}s.",
        )
    except FileNotFoundError:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="git is not installed on the server.",
        )

    if result.returncode != 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository. It may not exist or may be private. ({result.stderr.strip()})",
        )

    return dest_dir


def cleanup_review(review_id: str) -> None:
    temp_dir = _review_temp_dirs.pop(review_id, None)
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


class StartReviewRequest(BaseModel):
    repo_url: str
    force_all: bool = False

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        if not GITHUB_URL_PATTERN.match(v.strip()):
            raise ValueError(
                "repo_url must be a public GitHub URL, e.g. https://github.com/owner/repo"
            )
        return v.strip()


class WorkerDecisionOut(BaseModel):
    worker_name: str
    reason: str
    should_run: bool


class StartReviewResponse(BaseModel):
    review_id: str
    force_all: bool
    proposed_decisions: List[WorkerDecisionOut]


class ResumeReviewRequest(BaseModel):
    # None = accept the proposed plan as-is.
    # A list = override should_run: only the named tools run.
    selected_tools: Optional[List[str]] = None


class ResumeReviewResponse(BaseModel):
    review_id: str
    final_report: str


@api.post(
    "/reviews",
    response_model=StartReviewResponse,
    dependencies=[Depends(verify_access_code)],
)
def start_review(payload: StartReviewRequest):
    review_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": review_id}}

    # Clone the requester's repo into an isolated temp directory before the
    # graph ever runs. This is what replaces the old local repo_path.
    temp_dir = clone_repo(payload.repo_url, review_id)
    _review_temp_dirs[review_id] = temp_dir

    initial_state = {
        "repo_path": temp_dir,
        "force_all_workers": payload.force_all,
    }

    try:
        graph_app.invoke(initial_state, config=config)
    except LLMInvocationError as e:
        cleanup_review(review_id)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        cleanup_review(review_id)
        raise

    state = graph_app.get_state(config).values
    decisions: List[WorkerDecision] = state.get("worker_decisions", [])

    return StartReviewResponse(
        review_id=review_id,
        force_all=payload.force_all,
        proposed_decisions=[
            WorkerDecisionOut(
                worker_name=d.worker_name, reason=d.reason, should_run=d.should_run
            )
            for d in decisions
        ],
    )


@api.post(
    "/reviews/{review_id}/resume",
    response_model=ResumeReviewResponse,
    dependencies=[Depends(verify_access_code)],
)
def resume_review(review_id: str, payload: ResumeReviewRequest):
    config = {"configurable": {"thread_id": review_id}}

    existing_state = graph_app.get_state(config).values
    if not existing_state:
        raise HTTPException(status_code=404, detail="No review found with that ID.")

    if payload.selected_tools is not None:
        chosen = {t.strip().lower() for t in payload.selected_tools}
        decisions: List[WorkerDecision] = existing_state.get("worker_decisions", [])
        for d in decisions:
            d.should_run = d.worker_name in chosen
        graph_app.update_state(config, {"worker_decisions": decisions})

    try:
        final_state = graph_app.invoke(None, config=config)
    except LLMInvocationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        # The cloned repo is only needed while the graph runs. Whether this
        # succeeds or fails, the review is done with it either way.
        cleanup_review(review_id)

    report = final_state.get("final_report")
    if not report:
        report = "No workers were run, so no findings were collected and no report was generated."

    return ResumeReviewResponse(review_id=review_id, final_report=report)