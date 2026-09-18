"""Test documentation freshness against committed JSON receipts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_documentation_matches_receipts() -> None:
    """Ensure README.md and WALKTHROUGH.md match reports/v2 receipts exactly."""
    script_path = REPO_ROOT / "scripts" / "render_receipts.py"
    assert script_path.exists(), "scripts/render_receipts.py must exist"

    cmd = [sys.executable, str(script_path), "--check"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Documentation drift detected! Run 'python scripts/render_receipts.py' to update.\n"
        f"Output:\n{result.stdout}\n{result.stderr}"
    )
