from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IssueSource(str, Enum):
    STYLE = "style"       # pylint
    SECURITY = "security" # bandit
    LOGIC = "logic"       # radon


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Issue(BaseModel):
    source: IssueSource
    severity: Severity
    file: str
    line: Optional[int] = None
    message: str
    suggestion: str
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model's confidence that this finding is a true positive, 0.0-1.0",
    )

class WorkerDecision(BaseModel):
    worker_name: str
    reason: str
    should_run: bool  
