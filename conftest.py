"""Make the repo root importable for pytest (so `import qwip_atlas` works)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
