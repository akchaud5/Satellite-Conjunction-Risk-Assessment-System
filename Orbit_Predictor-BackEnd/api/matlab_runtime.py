"""Lazy, process-wide access to the MATLAB Engine for Python.

The MATLAB Engine ships with a MATLAB installation rather than from PyPI, so it
is not available on every machine that runs this project. Importing
``matlab.engine`` at module scope therefore made the entire Django app
unimportable without MATLAB -- including the CDM, user and ML endpoints, which
do not need it at all.

Everything here is deferred to first use instead:

* ``matlab_available()`` reports whether the engine can be imported, without
  starting it.
* ``get_matlab()`` imports and starts the engine on first call and reuses it
  afterwards. The previous code called ``start_matlab()`` on every request and
  never called ``quit()``, leaking a MATLAB process per conjunction evaluated.

Callers that need the engine should catch :class:`MatlabUnavailable` and turn it
into a 503, so an install without MATLAB degrades to "this endpoint is off"
rather than failing to boot.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory holding the NASA CARA .m files (Pc2D_Foster, Pc3D_Hall, ...).
# Those scripts add their own ``Utils`` subdirectory to the MATLAB path at run
# time, so only this directory is registered here.
MATLAB_SCRIPT_PATH = Path(__file__).resolve().parent / "matlab"

_engine = None
_engine_lock = threading.Lock()


class MatlabUnavailable(RuntimeError):
    """Raised when an analytic endpoint is hit on an install without MATLAB."""


def matlab_available():
    """Return True if the MATLAB Engine can be imported. Does not start it."""
    try:
        import matlab.engine  # noqa: F401
    except Exception:  # pragma: no cover - depends on local MATLAB install
        return False
    return True


def get_matlab():
    """Return ``(matlab_module, engine)``, starting the engine on first use.

    The engine is cached for the lifetime of the process and shared by all
    callers. Starting MATLAB takes several seconds, so the first call is slow
    and subsequent ones are not.

    Raises:
        MatlabUnavailable: if the engine is not installed or will not start.
    """
    global _engine

    try:
        import matlab
        import matlab.engine
    except Exception as exc:  # pragma: no cover - depends on local MATLAB
        raise MatlabUnavailable(
            "MATLAB Engine for Python is not installed, so analytic collision "
            "probability is unavailable. See requirements-matlab.txt."
        ) from exc

    if _engine is None:
        with _engine_lock:
            if _engine is None:
                logger.info("Starting MATLAB engine (first use)...")
                try:
                    engine = matlab.engine.start_matlab()
                    engine.addpath(str(MATLAB_SCRIPT_PATH))
                except Exception as exc:  # pragma: no cover
                    raise MatlabUnavailable(
                        f"MATLAB Engine failed to start: {exc}"
                    ) from exc
                _engine = engine
                logger.info("MATLAB engine ready.")

    return matlab, _engine


def shutdown_matlab():
    """Stop the cached engine, if one is running. Safe to call repeatedly."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.quit()
            except Exception:  # pragma: no cover
                logger.warning("MATLAB engine did not shut down cleanly.")
            _engine = None
