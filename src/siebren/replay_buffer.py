import importlib
from typing import Literal, Tuple, Union

import numpy as np
import numpy.typing as npt

_native = importlib.import_module("siebren.siebren")

Game = Literal["tictactoe", "connect4", "bytefight"]


def _normalize_game(game: str) -> Game:
    normalized = game.strip().lower()
    if normalized not in {"tictactoe", "connect4", "bytefight"}:
        raise ValueError(
            f"Unknown game: {game}. Expected one of: tictactoe, connect4, bytefight"
        )
    return normalized  # type: ignore[return-value]


class EphemeralReplayBuffer:
    """In-memory replay buffer that stores observations directly."""

    def __init__(self, capacity: int, game: str) -> None:
        self.game = _normalize_game(game)
        cls = {
            "tictactoe": _native.TicTacToeEphemeralReplayBuffer,
            "connect4": _native.Connect4EphemeralReplayBuffer,
            "bytefight": _native.ByteFightEphemeralReplayBuffer,
        }[self.game]
        self._inner = cls(capacity)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def capacity(self) -> int:
        return self._inner.capacity

    def sample(
        self, n: int, seed: int
    ) -> Tuple[
        npt.NDArray[np.generic], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        return self._inner.sample(n, seed)


class SavableReplayBuffer:
    """Disk-persisted replay buffer placeholder.

    TODO: Reintroduce notation-backed persistence on top of the new API.
    """

    def __init__(self, capacity: int, game: str) -> None:
        self.capacity = capacity
        self.game = _normalize_game(game)
        raise NotImplementedError(
            "SavableReplayBuffer is temporarily disabled after replay refactor. "
            "Use EphemeralReplayBuffer for now."
        )


ReplayBuffer = Union[EphemeralReplayBuffer, SavableReplayBuffer]
