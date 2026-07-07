"""
MLflow helpers shared across all three notebooks/scripts.
Provides experiment setup for the ./mlruns file store.

Note: the per-experiment run loggers (log_vision_run/log_text_run/log_clip_run)
were removed — they had no call sites (each trainer/pipeline logs to MLflow
inline) and had drifted out of sync with the metrics actually produced.
"""

from __future__ import annotations

import os

# MLflow >=3 moved the local filesystem tracking store into "maintenance mode"
# and raises unless the caller opts in. This project intentionally uses the
# simple ./mlruns file store (browsable via `mlflow ui`, gitignored) rather than
# a database backend, so opt in before importing mlflow. Set the env var to
# "false" to override and force the database-backend guidance instead.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402


def setup_mlflow(experiment_name: str, tracking_uri: str = "mlruns") -> None:
    """Initialise MLflow with the given experiment. Creates it if it doesn't exist."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
