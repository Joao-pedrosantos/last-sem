import sys
from pathlib import Path

# Put the project root (parent of backend/) on sys.path so tests can do
# `from backend.cxr_pipeline import ...` regardless of where pytest is invoked.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
