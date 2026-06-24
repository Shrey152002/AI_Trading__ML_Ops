"""In-memory ring buffer of backend log records, polled by the dashboard to show
live pipeline progress. Single-process, last-N-lines only — there is no persistence
or multi-worker fan-in here, which is fine for the local/single-operator deployment
this dashboard targets.
"""
import logging
from collections import deque
from datetime import datetime, timezone
from itertools import count
from threading import Lock

_BUFFER_MAXLEN = 2000
_buffer: deque = deque(maxlen=_BUFFER_MAXLEN)
_lock = Lock()
_seq_counter = count(1)


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "seq": next(_seq_counter),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        with _lock:
            _buffer.append(entry)


# Only the loggers actually involved in a pipeline run — deliberately excludes
# api.main's per-request access log, so routine dashboard polling doesn't drown
# out training progress in the buffer.
PIPELINE_LOGGER_NAMES = [
    "data.ingestion",
    "data.validation",
    "feature_store.store",
    "drift.detector",
    "scheduler.pipeline",
    "training.trainer",
    "training.hyperopt",
    "registry.model_registry",
    "reports.generator",
]

_installed = False


def install_log_capture(level: int = logging.INFO) -> None:
    """Attaches the buffering handler to the pipeline-relevant loggers. Safe to call more than once."""
    global _installed
    if _installed:
        return
    handler = _BufferHandler()
    handler.setLevel(level)
    for name in PIPELINE_LOGGER_NAMES:
        pipeline_logger = logging.getLogger(name)
        # Set explicitly rather than relying on root's level: logging.basicConfig()
        # is a no-op once root already has a handler (e.g. attached by pytest's
        # logging plugin, or by uvicorn's own setup), which would otherwise leave
        # these loggers at the default WARNING level and silently drop .info() calls.
        pipeline_logger.setLevel(level)
        pipeline_logger.addHandler(handler)
    _installed = True


def get_recent_logs(since: int = 0, limit: int = 500) -> list[dict]:
    """Returns buffered entries with seq > `since`, oldest first, capped at `limit`."""
    with _lock:
        matching = [entry for entry in _buffer if entry["seq"] > since]
    return matching[-limit:]
