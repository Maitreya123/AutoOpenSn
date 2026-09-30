"""HTTP access to the workflow engine.

``create_app`` builds the FastAPI application; ``JobRegistry`` is the in-process
record of studies running in the background. Both are importable without
FastAPI being configured for anything, so the CLI can stay thin.
"""

from autoopensn.api.app import create_app
from autoopensn.api.jobs import Job, JobRegistry

__all__ = ["Job", "JobRegistry", "create_app"]
