import importlib
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .replay_buffer import EphemeralReplayBuffer, ReplayBuffer, _normalize_game

_native = importlib.import_module("siebren.siebren")


class SelfPlay:
    """Unified self-play runner for all supported games.

    IMPORTANT: num_threads * workers_per_thread must be >= 256 (the batch size).
    """

    def __init__(
        self,
        game: str,
        num_threads: int = 32,
        workers_per_thread: int = 256,
        mcts_num_simulations: int = 20,
        seed: int = 42,
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
        self.num_threads = num_threads
        self.workers_per_thread = workers_per_thread
        self.mcts_num_simulations = mcts_num_simulations
        self.seed = seed

    def play_games(
        self,
        replay_buffer: ReplayBuffer,
        num_samples: int,
        execute_model: Optional[
            Callable[
                [npt.NDArray[np.generic]],
                Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
            ]
        ] = None,
        *,
        use_rust_cudagraph: bool = False,
        model: Optional[Any] = None,
        selfplay_precision: str = "fp32",
    ) -> Tuple[int, int, Dict[str, int]]:
        """Run self-play games and return (games_completed, samples_collected, executor_counters)."""
        assert replay_buffer.game == self.game
        assert isinstance(replay_buffer, EphemeralReplayBuffer)

        if self.game == "bytefight":
            if model is None:
                raise ValueError(
                    "bytefight self-play requires a CUDA model for rust-cudagraph backend"
                )
            return _native.selfplay_bytefight_ephemeral(
                replay_buffer._inner,
                self.num_threads,
                self.workers_per_thread,
                num_samples,
                self.seed,
                mcts_num_simulations=self.mcts_num_simulations,
                model=model,
                selfplay_precision=selfplay_precision,
            )

        if execute_model is None:
            raise ValueError("execute_model callback is required for this game")

        fn = {
            "tictactoe": _native.selfplay_tictactoe_ephemeral,
            "connect4": _native.selfplay_connect4_ephemeral,
        }[self.game]

        return fn(
            replay_buffer._inner,
            self.num_threads,
            self.workers_per_thread,
            num_samples,
            self.seed,
            execute_model,
            mcts_num_simulations=self.mcts_num_simulations,
        )
