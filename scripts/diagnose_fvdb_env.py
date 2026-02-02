import json
import os
import sys
import time
import traceback
from importlib import metadata
from importlib.util import find_spec

LOG_PATH = "/Users/tbm/software/FRAME/.cursor/debug.log"
SESSION_ID = "debug-session"
RUN_ID = "pre-fix"


def _log_event(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> int:
    # region agent log
    _log_event(
        "H1",
        "scripts/diagnose_fvdb_env.py:29",
        "Python runtime info",
        {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": sys.platform,
        },
    )
    # endregion

    # region agent log
    _log_event(
        "H2",
        "scripts/diagnose_fvdb_env.py:41",
        "Env markers",
        {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
            "UV_PROJECT_ENV": os.environ.get("UV_PROJECT_ENV"),
        },
    )
    # endregion

    torch_spec = find_spec("torch")
    # region agent log
    _log_event(
        "H2",
        "scripts/diagnose_fvdb_env.py:56",
        "Torch module spec",
        {"found": torch_spec is not None, "origin": getattr(torch_spec, "origin", None)},
    )
    # endregion

    torch_version = None
    torch_import_error = None
    try:
        import torch  # noqa: F401

        torch_version = metadata.version("torch")
    except Exception:  # noqa: BLE001
        torch_import_error = traceback.format_exc(limit=2)

    # region agent log
    _log_event(
        "H3",
        "scripts/diagnose_fvdb_env.py:74",
        "Torch import result",
        {"version": torch_version, "error": torch_import_error},
    )
    # endregion

    numpy_version = None
    numpy_import_error = None
    try:
        import numpy  # noqa: F401

        numpy_version = metadata.version("numpy")
    except Exception:  # noqa: BLE001
        numpy_import_error = traceback.format_exc(limit=2)

    # region agent log
    _log_event(
        "H5",
        "scripts/diagnose_fvdb_env.py:94",
        "Numpy import result",
        {"version": numpy_version, "error": numpy_import_error},
    )
    # endregion

    torch_lib_dir = None
    torch_python_libs = []
    if torch_spec is not None and torch_spec.origin is not None:
        torch_lib_dir = os.path.join(os.path.dirname(torch_spec.origin), "lib")
        if os.path.isdir(torch_lib_dir):
            torch_python_libs = [
                name
                for name in os.listdir(torch_lib_dir)
                if name.startswith("libtorch_python.")
            ]

    # region agent log
    _log_event(
        "H6",
        "scripts/diagnose_fvdb_env.py:117",
        "Torch libtorch_python files",
        {"lib_dir": torch_lib_dir, "files": torch_python_libs},
    )
    # endregion

    # region agent log
    _log_event(
        "H1",
        "scripts/diagnose_fvdb_env.py:128",
        "Importlib torch version",
        {"metadata_version": torch_version},
    )
    # endregion

    fvdb_spec = find_spec("fvdb")
    # region agent log
    _log_event(
        "H4",
        "scripts/diagnose_fvdb_env.py:139",
        "fvdb module spec",
        {"found": fvdb_spec is not None, "origin": getattr(fvdb_spec, "origin", None)},
    )
    # endregion

    fvdb_version = None
    fvdb_import_error = None
    try:
        import fvdb  # noqa: F401

        fvdb_version = metadata.version("fvdb-core")
    except Exception:  # noqa: BLE001
        fvdb_import_error = traceback.format_exc(limit=2)

    # region agent log
    _log_event(
        "H4",
        "scripts/diagnose_fvdb_env.py:157",
        "fvdb import result",
        {"version": fvdb_version, "error": fvdb_import_error},
    )
    # endregion

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
