"""The committed notebook must be exactly what notebook_src.txt builds.

The source file is the single place to edit. This test rebuilds the notebook
in a temporary folder and compares every cell, so a hand edit to the notebook
or a forgotten rebuild fails here instead of reaching a reader.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "cosmos3_edge_first_timers_guide.ipynb"


def _cells(path: Path) -> list[tuple[str, str]]:
    """Cell type and source for each cell, ignoring ids and metadata."""
    notebook = nbformat.read(path, as_version=4)
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def test_notebook_matches_source(tmp_path: Path) -> None:
    """Rebuild from the source and compare with the committed notebook."""
    built = tmp_path / NOTEBOOK
    subprocess.run(
        [sys.executable, "build_notebook.py", str(built)], cwd=ROOT, check=True
    )
    assert _cells(built) == _cells(ROOT / NOTEBOOK)


def test_committed_notebook_has_no_outputs() -> None:
    """Outputs belong in records/, never in the notebook readers open."""
    notebook = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(not cell.outputs for cell in code_cells)
