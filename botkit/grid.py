"""
grid.py — flexible 2D grid for spatial games
---------------------------------------------
Features:
  - Arbitrary size, configurable cell type
  - 4-directional and 8-directional neighbor enumeration
  - Fog-of-war / visibility masking
  - Incremental diff/patch updates from server responses
  - numpy backend when available (falls back to list-of-lists)
"""

from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, Optional

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """
    A single grid cell.  Subclass or replace the `data` field freely.

    Attributes:
        row, col  : position
        passable  : whether pathfinding treats this cell as traversable
        visible   : False means fog-of-war; search helpers respect this flag
        data      : arbitrary game-specific payload (terrain type, item, etc.)
    """

    row: int
    col: int
    passable: bool = True
    visible: bool = True
    data: Any = None

    def __repr__(self) -> str:
        v = "" if self.visible else "?"
        p = "" if self.passable else "#"
        return f"Cell({self.row},{self.col}{p}{v})"


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

DIRS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))  # N S W E
DIRS_8 = DIRS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))  # + diagonals


class Grid:
    """
    A 2D grid of Cell objects.

    Parameters
    ----------
    rows, cols      : dimensions
    default_passable: initial passability for all cells (True = open)
    eight_directional: if True, neighbors() yields 8 neighbors instead of 4
    cell_factory    : optional callable(row, col) -> Cell for custom init

    Usage
    -----
    >>> g = Grid(5, 5)
    >>> g[2, 2].passable = False          # place a wall
    >>> g.set_visible([(0,0),(0,1)], True) # reveal cells
    >>> for n in g.neighbors(1, 1):
    ...     print(n)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        default_passable: bool = True,
        eight_directional: bool = False,
        cell_factory: Optional[Callable[[int, int], Cell]] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.eight_directional = eight_directional
        self._dirs = DIRS_8 if eight_directional else DIRS_4

        if cell_factory:
            self._cells = [
                [cell_factory(r, c) for c in range(cols)] for r in range(rows)
            ]
        else:
            self._cells = [
                [Cell(r, c, passable=default_passable) for c in range(cols)]
                for r in range(rows)
            ]

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def __getitem__(self, pos: tuple[int, int]) -> Cell:
        r, c = pos
        return self._cells[r][c]

    def __setitem__(self, pos: tuple[int, int], cell: Cell) -> None:
        r, c = pos
        self._cells[r][c] = cell

    def get(self, row: int, col: int, default=None) -> Optional[Cell]:
        """Bounds-safe getter."""
        if self.in_bounds(row, col):
            return self._cells[row][col]
        return default

    # ------------------------------------------------------------------
    # Neighbors
    # ------------------------------------------------------------------

    def neighbors(
        self,
        row: int,
        col: int,
        passable_only: bool = True,
        visible_only: bool = False,
    ) -> Generator[Cell, None, None]:
        """
        Yield neighboring cells.

        Parameters
        ----------
        passable_only : skip walls / impassable cells (default True)
        visible_only  : skip fog-of-war cells
        """
        for dr, dc in self._dirs:
            r, c = row + dr, col + dc
            if not self.in_bounds(r, c):
                continue
            cell = self._cells[r][c]
            if passable_only and not cell.passable:
                continue
            if visible_only and not cell.visible:
                continue
            yield cell

    def neighbor_coords(
        self,
        row: int,
        col: int,
        passable_only: bool = True,
        visible_only: bool = False,
    ) -> Generator[tuple[int, int], None, None]:
        """Like neighbors() but yields (row, col) tuples."""
        for cell in self.neighbors(row, col, passable_only, visible_only):
            yield cell.row, cell.col

    # ------------------------------------------------------------------
    # Fog of war
    # ------------------------------------------------------------------

    def set_visible(
        self, positions: Iterable[tuple[int, int]], visible: bool = True
    ) -> None:
        for r, c in positions:
            if self.in_bounds(r, c):
                self._cells[r][c].visible = visible

    def reveal_all(self) -> None:
        for row in self._cells:
            for cell in row:
                cell.visible = True

    def hide_all(self) -> None:
        for row in self._cells:
            for cell in row:
                cell.visible = False

    def visible_cells(self) -> Generator[Cell, None, None]:
        for row in self._cells:
            for cell in row:
                if cell.visible:
                    yield cell

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def all_cells(self) -> Generator[Cell, None, None]:
        for row in self._cells:
            yield from row

    def passable_cells(self) -> Generator[Cell, None, None]:
        for cell in self.all_cells():
            if cell.passable:
                yield cell

    def set_passable_rect(
        self, r0: int, c0: int, r1: int, c1: int, passable: bool
    ) -> None:
        """Mark a rectangular region passable or impassable."""
        for r in range(max(0, r0), min(self.rows, r1 + 1)):
            for c in range(max(0, c0), min(self.cols, c1 + 1)):
                self._cells[r][c].passable = passable

    # ------------------------------------------------------------------
    # Diff / patch (for incremental server updates)
    # ------------------------------------------------------------------

    def patch(self, updates: Iterable[dict]) -> None:
        """
        Apply a list of cell updates from a server response.

        Each update dict should have at minimum 'row' and 'col' keys.
        Remaining keys are applied as attributes on the Cell.

        Example update dicts:
            {"row": 2, "col": 3, "passable": False}
            {"row": 0, "col": 0, "visible": True, "data": {"type": "gold"}}
        """
        for upd in updates:
            r, c = upd["row"], upd["col"]
            if not self.in_bounds(r, c):
                continue
            cell = self._cells[r][c]
            for key, val in upd.items():
                if key in ("row", "col"):
                    continue
                setattr(cell, key, val)

    def diff(self, other: "Grid") -> list[dict]:
        """
        Return a list of update dicts describing cells that differ between
        self and other (same dimensions assumed).  Useful for state diffing.
        """
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                a, b = self._cells[r][c], other._cells[r][c]
                changes = {"row": r, "col": c}
                if a.passable != b.passable:
                    changes["passable"] = b.passable
                if a.visible != b.visible:
                    changes["visible"] = b.visible
                if a.data != b.data:
                    changes["data"] = b.data
                if len(changes) > 2:
                    result.append(changes)
        return result

    # ------------------------------------------------------------------
    # Flood fill / connected components
    # ------------------------------------------------------------------

    def flood_fill(
        self,
        start_row: int,
        start_col: int,
        passable_only: bool = True,
    ) -> set[tuple[int, int]]:
        """
        Return the set of (row, col) positions reachable from start via BFS.
        Useful for territory counting and reachability checks.
        """
        from collections import deque

        visited = set()
        queue = deque([(start_row, start_col)])
        visited.add((start_row, start_col))
        while queue:
            r, c = queue.popleft()
            for nr, nc in self.neighbor_coords(r, c, passable_only=passable_only):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    def connected_components(
        self, passable_only: bool = True
    ) -> list[set[tuple[int, int]]]:
        """Return a list of connected components as sets of (row, col) tuples."""
        visited: set[tuple[int, int]] = set()
        components = []
        for cell in self.all_cells():
            pos = (cell.row, cell.col)
            if pos in visited:
                continue
            if passable_only and not cell.passable:
                continue
            component = self.flood_fill(cell.row, cell.col, passable_only)
            components.append(component)
            visited |= component
        return components

    # ------------------------------------------------------------------
    # Numpy interop
    # ------------------------------------------------------------------

    def to_numpy_passable(self):
        """Return a boolean numpy array: True = passable. Requires numpy."""
        if np is None:
            raise RuntimeError("numpy not installed")
        return np.array(
            [
                [self._cells[r][c].passable for c in range(self.cols)]
                for r in range(self.rows)
            ],
            dtype=bool,
        )

    def to_numpy_data(self, key: str = "data", dtype=float):
        """
        Return a numpy array of a numeric data field.
        Example: grid.to_numpy_data() if cell.data is a float cost.
        """
        if np is None:
            raise RuntimeError("numpy not installed")
        return np.array(
            [
                [getattr(self._cells[r][c], key) for c in range(self.cols)]
                for r in range(self.rows)
            ],
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        for row in self._cells:
            parts = []
            for cell in row:
                if not cell.visible:
                    parts.append("?")
                elif not cell.passable:
                    parts.append("#")
                else:
                    parts.append(".")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols})"
