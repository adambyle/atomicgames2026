"""
search.py — minimax with alpha-beta pruning + iterative deepening
-----------------------------------------------------------------
Provides:
  minimax        — single-depth minimax with alpha-beta pruning
  minimax_timed  — iterative-deepening minimax with a wall-clock time budget
                   (returns the best action found before time runs out)

Both functions work with any GameState subclass.

Design notes
------------
- Perspective is always player 0 (you).  evaluate(0) must return positive
  values for states good for you.
- Transposition table (optional): pass tt={} to cache evaluated positions.
  Uses GameState.zobrist_hash() as the key.
- Move ordering: the search calls state.action_order_hint() rather than
  state.get_actions(), so you can bias toward promising moves for better
  pruning — this can make alpha-beta several times faster in practice.
"""

import math
import time
from typing import Any, Dict, Optional, Tuple

from .gamestate import GameState

_INF = math.inf

# Transposition table entry flags
_EXACT = 0
_LOWER = 1  # alpha cutoff (we had a beta-cut, stored a lower bound)
_UPPER = 2  # beta  cutoff (stored an upper bound)


# ---------------------------------------------------------------------------
# Core alpha-beta minimax
# ---------------------------------------------------------------------------


def _alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    perspective: int,
    tt: Optional[Dict],
) -> float:
    """
    Recursive alpha-beta minimax.  Returns the heuristic value of `state`
    from the point of view of `perspective`.
    """
    # --- Transposition table lookup ---
    zh = state.zobrist_hash() if tt is not None else None
    if tt is not None and zh in tt:
        entry = tt[zh]
        if entry["depth"] >= depth:
            flag, val = entry["flag"], entry["value"]
            if flag == _EXACT:
                return val
            elif flag == _LOWER:
                alpha = max(alpha, val)
            elif flag == _UPPER:
                beta = min(beta, val)
            if alpha >= beta:
                return val

    # --- Base cases ---
    if depth == 0 or state.is_terminal():
        val = state.evaluate(perspective)
        if tt is not None:
            tt[zh] = {"depth": depth, "flag": _EXACT, "value": val}
        return val

    actions = state.action_order_hint()
    if not actions:
        val = state.evaluate(perspective)
        if tt is not None:
            tt[zh] = {"depth": depth, "flag": _EXACT, "value": val}
        return val

    maximising = state.is_maximising(perspective)
    orig_alpha = alpha
    best_val = -_INF if maximising else _INF

    for action in actions:
        child = state.apply_action(action)
        val = _alphabeta(child, depth - 1, alpha, beta, perspective, tt)
        if maximising:
            best_val = max(best_val, val)
            alpha = max(alpha, best_val)
        else:
            best_val = min(best_val, val)
            beta = min(beta, best_val)
        if alpha >= beta:
            break  # prune

    # --- Transposition table store ---
    if tt is not None:
        if best_val <= orig_alpha:
            flag = _UPPER
        elif best_val >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        tt[zh] = {"depth": depth, "flag": flag, "value": best_val}

    return best_val


# ---------------------------------------------------------------------------
# Public: minimax (fixed depth)
# ---------------------------------------------------------------------------


def minimax(
    state: GameState,
    depth: int,
    perspective: int = 0,
    tt: Optional[Dict] = None,
) -> Tuple[Any, float]:
    """
    Alpha-beta minimax to a fixed depth.

    Parameters
    ----------
    state       : current game state
    depth       : search depth (plies)
    perspective : player index to maximise for (default 0 = you)
    tt          : optional transposition table dict (pass {} to enable,
                  reuse across calls to the same game for best effect)

    Returns
    -------
    (best_action, score)
      best_action : the action to take from the current state
      score       : evaluated score for that action
    """
    actions = state.action_order_hint()
    if not actions:
        return None, state.evaluate(perspective)

    best_action = None
    best_score = -_INF
    alpha = -_INF
    beta = _INF

    for action in actions:
        child = state.apply_action(action)
        score = _alphabeta(child, depth - 1, alpha, beta, perspective, tt)
        if score > best_score:
            best_score = score
            best_action = action
        alpha = max(alpha, best_score)
        # Note: no beta cutoff at root — we need to evaluate all root moves
        # to return the best action (not just a bound).

    return best_action, best_score


# ---------------------------------------------------------------------------
# Public: minimax with iterative deepening + time budget
# ---------------------------------------------------------------------------


def minimax_timed(
    state: GameState,
    time_limit: float = 0.5,
    max_depth: int = 50,
    perspective: int = 0,
    tt: Optional[Dict] = None,
) -> Tuple[Any, float, int]:
    """
    Iterative-deepening minimax with a wall-clock time budget.

    Searches depth 1, 2, 3, ... until either `time_limit` seconds have
    elapsed or `max_depth` is reached.  Always returns the best result
    found so far, so it's safe to interrupt at any depth.

    Parameters
    ----------
    state       : current game state
    time_limit  : wall-clock seconds budget (default 0.5s)
    max_depth   : hard cap on search depth (default 50)
    perspective : player to maximise for (default 0)
    tt          : transposition table dict; highly recommended here since
                  shallower searches prime the table for deeper ones

    Returns
    -------
    (best_action, best_score, depth_reached)
      best_action   : action to play
      best_score    : score at the deepest completed search
      depth_reached : deepest fully-completed depth

    Notes
    -----
    Pass a shared tt={} across turns of the same game for significant gains.
    The TT is keyed by Zobrist hash, so implement zobrist_hash() well.
    """
    if tt is None:
        tt = {}

    deadline = time.monotonic() + time_limit
    best_action = None
    best_score = -_INF
    depth_reached = 0

    # Always have at least a depth-1 result
    actions = state.action_order_hint()
    if not actions:
        return None, state.evaluate(perspective), 0

    for depth in range(1, max_depth + 1):
        if time.monotonic() >= deadline:
            break

        action, score = _minimax_root_with_deadline(
            state, depth, perspective, tt, deadline
        )
        if action is None:
            # Ran out of time mid-search; keep previous result
            break

        best_action = action
        best_score = score
        depth_reached = depth

        # Early exit on proven win/loss
        if abs(best_score) == _INF:
            break

    return best_action, best_score, depth_reached


def _minimax_root_with_deadline(
    state: GameState,
    depth: int,
    perspective: int,
    tt: Dict,
    deadline: float,
) -> Tuple[Optional[Any], float]:
    """Root call that bails out if the deadline passes mid-search."""
    actions = state.action_order_hint()
    best_action = None
    best_score = -_INF
    alpha = -_INF
    beta = _INF

    for action in actions:
        if time.monotonic() >= deadline:
            return None, -_INF  # signal: incomplete search
        child = state.apply_action(action)
        score = _alphabeta_deadline(
            child, depth - 1, alpha, beta, perspective, tt, deadline
        )
        if score is None:
            return None, -_INF  # propagate timeout
        if score > best_score:
            best_score = score
            best_action = action
        alpha = max(alpha, best_score)

    return best_action, best_score


def _alphabeta_deadline(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    perspective: int,
    tt: Dict,
    deadline: float,
) -> Optional[float]:
    """Alpha-beta that returns None if the deadline is exceeded."""
    if time.monotonic() >= deadline:
        return None

    zh = state.zobrist_hash()
    if zh in tt:
        entry = tt[zh]
        if entry["depth"] >= depth:
            flag, val = entry["flag"], entry["value"]
            if flag == _EXACT:
                return val
            elif flag == _LOWER:
                alpha = max(alpha, val)
            elif flag == _UPPER:
                beta = min(beta, val)
            if alpha >= beta:
                return val

    if depth == 0 or state.is_terminal():
        val = state.evaluate(perspective)
        tt[zh] = {"depth": depth, "flag": _EXACT, "value": val}
        return val

    actions = state.action_order_hint()
    if not actions:
        val = state.evaluate(perspective)
        tt[zh] = {"depth": depth, "flag": _EXACT, "value": val}
        return val

    maximising = state.is_maximising(perspective)
    orig_alpha = alpha
    best_val = -_INF if maximising else _INF

    for action in actions:
        child = state.apply_action(action)
        val = _alphabeta_deadline(
            child, depth - 1, alpha, beta, perspective, tt, deadline
        )
        if val is None:
            return None  # propagate timeout
        if maximising:
            best_val = max(best_val, val)
            alpha = max(alpha, best_val)
        else:
            best_val = min(best_val, val)
            beta = min(beta, best_val)
        if alpha >= beta:
            break

    flag = (
        _UPPER if best_val <= orig_alpha else (_LOWER if best_val >= beta else _EXACT)
    )
    tt[zh] = {"depth": depth, "flag": flag, "value": best_val}
    return best_val
