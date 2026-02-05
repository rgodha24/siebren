from typing import TYPE_CHECKING, Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt

from . import siebren

if TYPE_CHECKING:
    from .siebren import TicTacToeReplayBuffer


class TicTacToeSelfPlay:
    """Self-play runner for TicTacToe.

    IMPORTANT: num_threads * workers_per_thread must be >= 256 (the batch size).
    Workers submit observations to a shared queue that dispatches when full.
    With fewer workers than batch size, workers will deadlock waiting for a
    batch that can never fill. Default config (32 * 16 = 512) is safe.

    The execute_model callback:
    - Input: (256, 9) int8 array - board states (0=empty, 1=X, -1=O)
    - Output: tuple of (policy, value)
        - policy: (256, 9) float32 - action probabilities
        - value: (256,) float32 - position evaluations in [-1, 1]
    """

    NUM_ACTIONS = 9

    def __init__(
        self,
        num_threads: int = 32,
        workers_per_thread: int = 16,
        seed: int = 42,
    ) -> None:
        total_workers = num_threads * workers_per_thread
        assert total_workers >= 256, (
            f"num_threads * workers_per_thread must be >= 256 (the batch size), "
            f"got {num_threads} * {workers_per_thread} = {total_workers}"
        )
        self.num_threads = num_threads
        self.workers_per_thread = workers_per_thread
        self.seed = seed

    def play_games(
        self,
        replay_buffer: "TicTacToeReplayBuffer",
        num_samples: int,
        execute_model: Callable[
            [npt.NDArray[np.int8]],
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
        ],
    ) -> Tuple[int, int, Dict[str, int]]:
        """Run self-play games and return (games_completed, samples_collected, executor_counters)."""
        return siebren.selfplay_tictactoe(
            replay_buffer,
            self.num_threads,
            self.workers_per_thread,
            num_samples,
            self.seed,
            execute_model,
        )
