import sys
from pathlib import Path

# On Windows, importing pyarrow AFTER other native libs (onnxruntime, gradio's
# stack) into the same process can trigger a DLL access violation during
# pytest's single-process collection. Importing it first here lets it
# initialise cleanly before anything else loads. No-op if pyarrow is absent
# (e.g. lean CI runners). Harmless on Linux/macOS.
try:  # noqa: SIM105
    import pyarrow  # noqa: F401
except Exception:
    pass

# Ensure repo root is importable so `import configs...` / `import src...` work.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
