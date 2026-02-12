import importlib
from typing import Any, Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .replay_buffer import EphemeralReplayBuffer, _normalize_game

_native = importlib.import_module("siebren.siebren")


class SelfPlay:
    """Persistent self-play session for all supported games.

    Wraps a native Rust session that preserves in-progress games across
    pause/resume boundaries.

    IMPORTANT: num_threads * workers_per_thread must be >= 256 (the batch size).
    """

    def __init__(
        self,
        game: str,
        replay_buffer: EphemeralReplayBuffer,
        *,
        num_threads: int = 32,
        workers_per_thread: int = 256,
        mcts_num_simulations: int = 20,
        mcts_c_puct: float = 1.5,
        mcts_dirichlet_alpha: float = 0.3,
        mcts_dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        exploration_moves: int = 30,
        seed: int = 42,
        execute_model: Optional[
            Callable[
                [npt.NDArray[np.generic]],
                Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
            ]
        ] = None,
        model: Optional[Any] = None,
        selfplay_precision: str = "fp32",
    ) -> None:
        self.game = _normalize_game(game)
        total_workers = num_threads * workers_per_thread
        assert total_workers >= 256, (
            f"num_threads * workers_per_thread must be >= 256 (the batch size), "
            f"got {num_threads} * {workers_per_thread} = {total_workers}"
        )
        assert mcts_num_simulations > 0, (
            f"mcts_num_simulations must be >= 1, got {mcts_num_simulations}"
        )
        assert mcts_c_puct > 0.0, f"mcts_c_puct must be > 0, got {mcts_c_puct}"
        assert mcts_dirichlet_alpha > 0.0, (
            f"mcts_dirichlet_alpha must be > 0, got {mcts_dirichlet_alpha}"
        )
        assert 0.0 <= mcts_dirichlet_epsilon <= 1.0, (
            f"mcts_dirichlet_epsilon must be in [0, 1], got {mcts_dirichlet_epsilon}"
        )
        assert temperature >= 0.0, f"temperature must be >= 0, got {temperature}"
        assert exploration_moves >= 0, (
            f"exploration_moves must be >= 0, got {exploration_moves}"
        )
        assert replay_buffer.game == self.game

        if self.game == "bytefight":
            if model is None:
                raise ValueError(
                    "bytefight self-play requires a CUDA model for rust-cudagraph backend"
                )
            self._session = _native.ByteFightSelfPlay(
                replay_buffer._inner,
                num_threads,
                workers_per_thread,
                seed,
                mcts_num_simulations=mcts_num_simulations,
                mcts_c_puct=mcts_c_puct,
                mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                mcts_dirichlet_epsilon=mcts_dirichlet_epsilon,
                temperature=temperature,
                exploration_moves=exploration_moves,
                model=model,
                selfplay_precision=selfplay_precision,
            )
        else:
            if execute_model is None:
                raise ValueError("execute_model callback is required for this game")
            cls = {
                "tictactoe": _native.TicTacToeSelfPlay,
                "connect4": _native.Connect4SelfPlay,
            }[self.game]
            self._session = cls(
                replay_buffer._inner,
                num_threads,
                workers_per_thread,
                seed,
                execute_model,
                mcts_num_simulations=mcts_num_simulations,
                mcts_c_puct=mcts_c_puct,
                mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                mcts_dirichlet_epsilon=mcts_dirichlet_epsilon,
                temperature=temperature,
                exploration_moves=exploration_moves,
            )

    def start(self) -> None:
        """Start self-play with no sample limit."""
        self._session.start()

    def wait_for(self, target_samples: int) -> int:
        """Block until absolute target_samples is reached, then pause and quiesce.

        Returns the actual number of samples collected (may exceed target).
        After this returns, it is safe to read the replay buffer.
        """
        return self._session.wait_for(target_samples)

    def samples(self) -> int:
        """Return the current absolute sample count."""
        return self._session.samples()

    def drop(self) -> None:
        """Shut down the session. Idempotent."""
        self._session.drop()

    def __del__(self) -> None:
        """Best-effort cleanup."""
        try:
            self._session.drop()
        except Exception:
            pass
