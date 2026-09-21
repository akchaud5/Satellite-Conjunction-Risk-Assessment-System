"""Background dispatch for ML training jobs.

Training used to run inline in the POST handler for /api/ml/training/ --
including the GridSearchCV path, which fits hundreds of models. That held a
gunicorn/runserver worker for the whole fit and returned a gateway timeout to
the client long before the job finished.

Training now runs on a worker thread and the endpoint returns 202 immediately
with a job id to poll at /api/ml/training/<id>/. The TrainingJob row is the
source of truth for progress: the training functions themselves move it through
running -> completed/failed.

A thread is the smallest change that gets the work off the request path with no
new infrastructure. For a real deployment -- multiple workers, retries,
surviving a restart -- this should become a Celery (or RQ) task; `run_training`
is deliberately shaped so that swapping the executor is a one-function change.
"""

import logging
import threading

from django.db import close_old_connections
from django.utils import timezone

logger = logging.getLogger(__name__)


def _run_and_record(train_callable, training_job_id, **kwargs):
    """Invoke a training function, making sure failures land on the job row."""
    # Imported here to keep this module importable from anywhere.
    from ..models.ml_model import TrainingJob

    try:
        train_callable(training_job_id=training_job_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - must not escape the thread
        logger.exception("Training job %s failed", training_job_id)
        # The training functions record their own failures, but a crash before
        # or after that handler would otherwise leave the job stuck at
        # 'running' with nothing to explain it.
        try:
            job = TrainingJob.objects.get(id=training_job_id)
            if job.status not in {"completed", "failed"}:
                job.status = "failed"
                job.error_message = str(exc)
                job.completed_at = timezone.now()
                job.save()
                job.ml_model.status = "failed"
                job.ml_model.save()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not mark training job %s as failed", training_job_id
            )
    finally:
        # Worker threads get their own DB connection; drop it so it is not
        # left open for the life of the process.
        close_old_connections()


def run_training(train_callable, training_job_id, **kwargs):
    """Start `train_callable` on a background thread and return immediately."""
    thread = threading.Thread(
        target=_run_and_record,
        args=(train_callable, training_job_id),
        kwargs=kwargs,
        name=f"ml-training-{training_job_id}",
        daemon=True,
    )
    thread.start()
    return thread
