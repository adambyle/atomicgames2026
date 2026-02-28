"""
graph.py — lightweight graph + search algorithms
-------------------------------------------------
Provides:
  Graph         — adjacency-list digraph with optional edge weights
  bfs           — breadth-first search; shortest path in unweighted graphs
  dijkstra      — shortest paths with non-negative weights
  astar         — A* search with pluggable heuristic
  greedy_best   — greedy best-first (heuristic only, no cost accumulation)

All search functions accept either a Graph OR a neighbor_fn callable so they
work equally well on Grid objects or any implicit graph.

neighbor_fn signature:  node -> Iterable[(neighbor, cost)]
"""

import heapq
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

Node = TypeVar("Node")
NeighborFn = Callable[[Any], Iterable[Tuple[Any, float]]]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class Graph:
    """
    Directed adjacency-list graph with optional edge weights.

    Undirected usage: call add_edge(u, v) and add_edge(v, u), or use
    add_undirected_edge(u, v).

    Nodes can be anything hashable.

    Example
    -------
    >>> g = Graph()
    >>> g.add_undirected_edge("A", "B", weight=1.5)
    >>> g.add_undirected_edge("B", "C", weight=2.0)
    >>> path, cost = astar(g, "A", "C", heuristic=lambda n: 0)
    """

    def __init__(self):
        # {node: [(neighbor, weight), ...]}
        self._adj: Dict[Any, list] = defaultdict(list)

    def add_node(self, node: Any) -> None:
        if node not in self._adj:
            self._adj[node] = []

    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        self._adj[u].append((v, weight))
        # Ensure v exists
        if v not in self._adj:
            self._adj[v] = []

    def add_undirected_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        self.add_edge(u, v, weight)
        self.add_edge(v, u, weight)

    def remove_edge(self, u: Any, v: Any) -> None:
        self._adj[u] = [(n, w) for n, w in self._adj[u] if n != v]

    def neighbors(self, node: Any) -> list[Tuple[Any, float]]:
        """Return list of (neighbor, weight) for the given node."""
        return self._adj.get(node, [])

    def neighbor_fn(self) -> NeighborFn:
        """Return a neighbor function compatible with the search algorithms."""
        return lambda node: self._adj.get(node, [])

    def nodes(self) -> list:
        return list(self._adj.keys())

    def __contains__(self, node: Any) -> bool:
        return node in self._adj

    def __repr__(self) -> str:
        return f"Graph(nodes={len(self._adj)})"

    # ------------------------------------------------------------------
    # Convenience: build from Grid
    # ------------------------------------------------------------------

    @classmethod
    def from_grid(cls, grid, cost_fn=None) -> "Graph":
        """
        Build a Graph from a botkit Grid.

        cost_fn(cell_from, cell_to) -> float, defaults to 1.0 per step.
        Only passable cells are included.
        """
        g = cls()
        for cell in grid.passable_cells():
            for nb in grid.neighbors(cell.row, cell.col, passable_only=True):
                cost = cost_fn(cell, nb) if cost_fn else 1.0
                g.add_edge((cell.row, cell.col), (nb.row, nb.col), weight=cost)
        return g


# ---------------------------------------------------------------------------
# Path reconstruction helper
# ---------------------------------------------------------------------------


def _reconstruct(came_from: dict, start: Any, goal: Any) -> list:
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


def _get_neighbor_fn(graph_or_fn) -> NeighborFn:
    """Accept a Graph instance or a raw callable."""
    if callable(graph_or_fn):
        return cast(NeighborFn, graph_or_fn)
    return graph_or_fn.neighbor_fn()


# ---------------------------------------------------------------------------
# BFS — unweighted shortest path
# ---------------------------------------------------------------------------


def bfs(
    graph_or_fn,
    start: Any,
    goal: Any,
) -> Tuple[Optional[list], int]:
    """
    Breadth-first search for an unweighted shortest path.

    Returns
    -------
    (path, steps)  where path is a list of nodes from start to goal,
                   or (None, -1) if no path exists.
    """
    neighbor_fn = _get_neighbor_fn(graph_or_fn)
    came_from: dict = {start: None}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == goal:
            # reconstruct
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path, len(path) - 1
        for nb, _ in neighbor_fn(node):
            if nb not in came_from:
                came_from[nb] = node
                queue.append(nb)

    return None, -1


def bfs_all(
    graph_or_fn,
    start: Any,
) -> Dict[Any, int]:
    """
    BFS from start to all reachable nodes.

    Returns dict of {node: distance_from_start}.
    """
    neighbor_fn = _get_neighbor_fn(graph_or_fn)
    dist: Dict[Any, int] = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb, _ in neighbor_fn(node):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


# ---------------------------------------------------------------------------
# Dijkstra — weighted shortest path
# ---------------------------------------------------------------------------


def dijkstra(
    graph_or_fn,
    start: Any,
    goal: Any = None,
) -> Tuple[Optional[list], float]:
    """
    Dijkstra's algorithm.

    If goal is None, runs to exhaustion and returns (None, dict_of_costs).
    Otherwise returns (path, total_cost) or (None, inf) if unreachable.

    Backed by heapq (C extension).
    """
    neighbor_fn = _get_neighbor_fn(graph_or_fn)
    dist: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Any] = {}
    heap = [(0.0, start)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        if goal is not None and node == goal:
            return _reconstruct(came_from, start, goal), cost
        for nb, w in neighbor_fn(node):
            new_cost = cost + w
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                came_from[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    if goal is None:
        return None, dist  # type: ignore
    return None, float("inf")


# ---------------------------------------------------------------------------
# A* — heuristic-guided shortest path
# ---------------------------------------------------------------------------


def astar(
    graph_or_fn,
    start: Any,
    goal: Any,
    heuristic: Callable[[Any], float] = lambda n: 0.0,
) -> Tuple[Optional[list], float]:
    """
    A* search.

    Parameters
    ----------
    graph_or_fn : Graph or neighbor_fn(node) -> [(neighbor, cost), ...]
    start       : start node
    goal        : goal node
    heuristic   : admissible heuristic h(node) -> estimated cost to goal
                  Defaults to 0 (equivalent to Dijkstra).

    Returns
    -------
    (path, cost) or (None, inf) if unreachable.

    Tips
    ----
    - For grid Manhattan distance: heuristic=lambda n: abs(n[0]-goal[0]) + abs(n[1]-goal[1])
    - For 8-directional Chebyshev: heuristic=lambda n: max(abs(n[0]-goal[0]), abs(n[1]-goal[1]))
    - For weighted goals, pass a custom cost function in your neighbor_fn.
    """
    neighbor_fn = _get_neighbor_fn(graph_or_fn)
    g_score: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Any] = {}
    # heap entries: (f_score, tie_break_counter, node)
    counter = 0
    heap = [(heuristic(start), 0, start)]

    while heap:
        f, _, node = heapq.heappop(heap)
        if node == goal:
            return _reconstruct(came_from, start, goal), g_score[goal]
        if f > g_score.get(node, float("inf")) + heuristic(node):
            continue
        current_g = g_score[node]
        for nb, w in neighbor_fn(node):
            tentative_g = current_g + w
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = node
                counter += 1
                heapq.heappush(heap, (tentative_g + heuristic(nb), counter, nb))

    return None, float("inf")


# ---------------------------------------------------------------------------
# Greedy best-first — fast but not optimal
# ---------------------------------------------------------------------------


def greedy_best_first(
    graph_or_fn,
    start: Any,
    goal: Any,
    heuristic: Callable[[Any], float],
) -> Tuple[Optional[list], int]:
    """
    Greedy best-first search.  Fast, not guaranteed optimal.
    Good for large spaces when you need an answer quickly.
    """
    neighbor_fn = _get_neighbor_fn(graph_or_fn)
    came_from: Dict[Any, Any] = {start: None}
    counter = 0
    heap = [(heuristic(start), 0, start)]

    while heap:
        _, _, node = heapq.heappop(heap)
        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path, len(path) - 1
        for nb, _ in neighbor_fn(node):
            if nb not in came_from:
                came_from[nb] = node
                counter += 1
                heapq.heappush(heap, (heuristic(nb), counter, nb))

    return None, -1


# ---------------------------------------------------------------------------
# Common grid heuristics (convenience)
# ---------------------------------------------------------------------------


def manhattan(a: tuple, b: tuple) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: tuple, b: tuple) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def euclidean(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def grid_neighbor_fn(grid, cost_fn=None) -> NeighborFn:
    """
    Build a neighbor function from a botkit Grid for use with search functions.

    cost_fn(from_cell, to_cell) -> float, defaults to 1.0.

    Example
    -------
    >>> nfn = grid_neighbor_fn(my_grid)
    >>> path, cost = astar(nfn, (0,0), (4,4), heuristic=lambda n: manhattan(n, (4,4)))
    """

    def _fn(pos):
        r, c = pos
        results = []
        for nb in grid.neighbors(r, c, passable_only=True):
            cost = cost_fn(grid[r, c], nb) if cost_fn else 1.0
            results.append(((nb.row, nb.col), cost))
        return results

    return _fn
