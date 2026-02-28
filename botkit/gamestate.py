"""
gamestate.py — abstract base class for two-player (or N-player) game states
----------------------------------------------------------------------------
Design goals:
  - Immutable-style: apply_action() returns a NEW state, never mutates self.
    This makes minimax and MCTS both work correctly without deep-copy bugs.
  - Player-agnostic: current_player() returns whichever player acts next,
    so minimax handles pass turns, multi-action turns, and >2 players.
  - Minimal surface area: only the methods you MUST implement are abstract.
    Everything else has sensible defaults you can override.

Quick-start
-----------
class MyGame(GameState):
    def __init__(self, board, player):
        self.board = board
        self.player = player

    def get_actions(self):
        return [all legal moves for self.player]

    def apply_action(self, action):
        new_board = ... # derive new board
        return MyGame(new_board, 1 - self.player)

    def is_terminal(self):
        return someone_won(self.board) or no_moves_left(self.board)

    def evaluate(self, perspective: int) -> float:
        # positive = good for `perspective`, negative = bad
        return score_for(perspective, self.board)

    def current_player(self):
        return self.player
"""

from abc import ABC, abstractmethod
from typing import Any, List


class GameState(ABC):
    """
    Abstract base class for game states used with minimax / MCTS.

    All state should be stored in instance attributes.
    apply_action() MUST return a new instance (do not mutate self).
    """

    # ------------------------------------------------------------------
    # Abstract interface — you must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_actions(self) -> List[Any]:
        """
        Return a list of legal actions available to the current player.
        Actions can be any hashable object (tuples, strings, ints, etc.).
        Return an empty list if the position is terminal.
        """
        ...

    @abstractmethod
    def apply_action(self, action: Any) -> "GameState":
        """
        Return a NEW GameState resulting from applying `action`.
        Do NOT mutate self.  The returned state should reflect the next
        player's turn (or the same player's if they get another move).
        """
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over (win/loss/draw/no moves)."""
        ...

    @abstractmethod
    def evaluate(self, perspective: int) -> float:
        """
        Return a heuristic score from the perspective of player `perspective`.

        Convention:
          +large  →  very good for `perspective`
          -large  →  very bad  for `perspective`
          0       →  neutral / even

        For terminal states this should return exact values (e.g. +inf / -inf
        for win/loss, 0 for draw) so minimax plays optimally when it reaches
        a terminal node.
        """
        ...

    @abstractmethod
    def current_player(self) -> int:
        """
        Return the index of the player who acts next.
        Convention: 0 = maximising player (you), 1 = minimising player.
        Extend to more players by returning 0..N-1.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides with sensible defaults
    # ------------------------------------------------------------------

    def is_maximising(self, perspective: int) -> bool:
        """
        Return True if current_player() is the maximising player from
        the given perspective.  Override for N-player games.
        """
        return self.current_player() == perspective

    def action_order_hint(self) -> List[Any]:
        """
        Return actions in a preferred search order (best first).
        Override to improve alpha-beta pruning efficiency dramatically.
        Default: same as get_actions().
        """
        return self.get_actions()

    def zobrist_hash(self) -> Any:
        """
        Return a hashable key representing this state for transposition tables.
        Default uses Python's built-in hash on the object — subclasses should
        override this with a proper Zobrist hash or tuple hash for correctness.
        """
        return hash(self)

    def clone(self) -> "GameState":
        """
        Return a deep copy of this state.
        Default uses Python's copy.deepcopy; override for efficiency.
        """
        import copy

        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def action_sequence(self, actions: List[Any]) -> "GameState":
        """Apply a sequence of actions and return the resulting state."""
        state = self
        for action in actions:
            state = state.apply_action(action)
        return state

    def children(self) -> List[tuple["GameState", Any]]:
        """Return [(next_state, action), ...] for all legal actions."""
        return [(self.apply_action(a), a) for a in self.get_actions()]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"player={self.current_player()}, "
            f"terminal={self.is_terminal()}, "
            f"actions={len(self.get_actions())})"
        )
