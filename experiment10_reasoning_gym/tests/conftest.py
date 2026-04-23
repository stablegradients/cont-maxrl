"""pytest configuration: put the repo root on sys.path so the
``experiment10_reasoning_gym`` package can be imported as
``experiment10_reasoning_gym.src.*`` regardless of cwd.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
