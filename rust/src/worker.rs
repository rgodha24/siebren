//! Worker loop and training data collection.
//!
//! Each worker runs MCTS searches, plays games, and collects training samples.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rand::Rng;

use crate::eval::Evaluator;
use crate::mcts::{best_action_index, sample_action_index, visits_to_policy, MCTSConfig, MCTS};
use crate::replay_buffer::{ReplayBuffer, Sample};
use crate::{Action, Environment, Player, TerminalState};

/// A training sample from a single game step.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    /// Serialized game state at this step.
    pub notation: String,
    /// The policy from MCTS (normalized visit counts).
    pub policy: Vec<f32>,
    /// The value from MCTS search.
    pub value: f32,
    /// Player to move at this step.
    ///
    /// Notation often encodes current player, but keeping it here avoids
    /// reparsing notation while backfilling terminal outcomes.
    pub player: Player,
}

/// Configuration for the worker.
#[derive(Clone)]
pub struct WorkerConfig {
    /// MCTS configuration.
    pub mcts: MCTSConfig,
    /// Temperature for action selection (1.0 = proportional to visits, 0.0 = argmax).
    pub temperature: f32,
    /// Number of moves at the start of the game to use exploration temperature.
    /// After this many moves, use temperature 0 (argmax).
    pub exploration_moves: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            mcts: MCTSConfig::default(),
            temperature: 1.0,
            exploration_moves: 30,
        }
    }
}

/// Run a single self-play game, collecting training samples.
///
/// Returns the collected samples. The game continues until terminal.
pub async fn play_game<E, V, R>(
    evaluator: &V,
    config: &WorkerConfig,
    rng: &mut R,
) -> Vec<TrainingSample>
where
    E: Environment,
    V: Evaluator<E>,
    R: Rng,
{
    let mut env = E::new();
    let mut samples = Vec::new();
    let mut move_count = 0;

    let mcts = MCTS::new(evaluator, &config.mcts);

    loop {
        if env.is_terminal().is_some() {
            break;
        }

        let visits = mcts.search(&mut env, rng).await;
        // Convert visits to policy
        let temp = if move_count < config.exploration_moves {
            config.temperature
        } else {
            0.0
        };
        let policy = visits_to_policy(&visits, temp);
        let player = env.current_player();
        let notation = env.to_notation();

        // Value is set to 0.0 here and backfilled with game outcome after the game ends.
        // This is standard AlphaZero practice - we use the actual game result rather than
        // the search value estimate for training.
        let value = 0.0;

        // Select action
        let action_idx = if temp > 0.0 {
            sample_action_index(&policy, rng)
        } else {
            best_action_index(&visits)
        };

        let action_idx = action_idx.expect("no valid actions but game not terminal");
        let action = E::Action::from_index(action_idx).expect("invalid action index");

        // Record sample
        samples.push(TrainingSample {
            player,
            notation,
            policy,
            value,
        });

        // Apply action
        env.apply_action(action);
        move_count += 1;
    }

    // Backfill values with game outcome
    let outcome = env.is_terminal().expect("game should be terminal");
    backfill_values(&mut samples, outcome);

    samples
}

/// Backfill sample values with the game outcome.
///
/// For wins, the winner's moves get +1, loser's get -1.
/// For draws, all moves get 0.
fn backfill_values(samples: &mut [TrainingSample], outcome: TerminalState) {
    for sample in samples.iter_mut() {
        sample.value = match outcome {
            TerminalState::Win(winner) => {
                if sample.player == winner {
                    1.0
                } else {
                    -1.0
                }
            }
            TerminalState::Draw => 0.0,
        };
    }
}

/// Run a worker loop that plays games until the target sample count is reached.
///
/// Workers play games and increment `samples_collected` after each game.
/// When the counter reaches `target_samples`, workers stop. The executor's
/// cancel callback should check this condition to terminate remaining workers.
///
/// Samples are pushed directly to the shared `replay_buffer` after each completed game.
pub async fn worker_loop<E, V, R>(
    evaluator: &V,
    config: &WorkerConfig,
    rng: &mut R,
    samples_collected: Arc<AtomicUsize>,
    games_completed: Arc<AtomicUsize>,
    target_samples: usize,
    replay_buffer: &ReplayBuffer,
) where
    E: Environment + Clone,
    V: Evaluator<E>,
    R: Rng,
{
    loop {
        if samples_collected.load(Ordering::Acquire) >= target_samples {
            break;
        }

        let game_samples = play_game::<E, V, R>(evaluator, config, rng).await;
        let num_samples = game_samples.len();

        // Push samples to replay buffer
        let mut guard = replay_buffer.reserve(num_samples);
        guard.extend(game_samples.into_iter().map(|s| Sample {
            notation: s.notation,
            policy: s.policy,
            value: s.value,
        }));

        samples_collected.fetch_add(num_samples, Ordering::AcqRel);
        games_completed.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::TicTacToe;
    use crate::eval::UniformEvaluator;
    use crate::executor::Executor;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_play_game_collects_samples() {
        let evaluator = UniformEvaluator;
        let config = WorkerConfig {
            mcts: MCTSConfig {
                num_simulations: 50,
                ..Default::default()
            },
            ..Default::default()
        };

        let rng = Rc::new(RefCell::new(ChaCha8Rng::seed_from_u64(42)));
        let result: Rc<RefCell<Option<Vec<TrainingSample>>>> = Rc::new(RefCell::new(None));

        let rng_clone = rng.clone();
        let result_clone = result.clone();

        let fut = async move {
            let samples =
                play_game::<TicTacToe, _, _>(&evaluator, &config, &mut *rng_clone.borrow_mut())
                    .await;
            *result_clone.borrow_mut() = Some(samples);
        };

        let event = event_listener::Event::new();
        let executor = Executor::new(|| event.listen());
        executor.run(vec![Box::pin(fut)], || false);

        let samples = result.borrow().clone().unwrap();

        // TicTacToe games are 5-9 moves
        assert!(samples.len() >= 5);
        assert!(samples.len() <= 9);

        // Each sample should have correct policy size
        for sample in &samples {
            assert_eq!(sample.policy.len(), 9);
            // Policy should sum to ~1
            let sum: f32 = sample.policy.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "policy sum: {}", sum);
        }

        // Values should be set (all -1, 0, or 1)
        for sample in &samples {
            assert!(sample.value == -1.0 || sample.value == 0.0 || sample.value == 1.0);
        }
    }

    #[test]
    fn test_backfill_values_win() {
        let mut samples = vec![TrainingSample {
            player: crate::Player::PlayerA,
            notation: String::new(),
            policy: vec![],
            value: 0.0,
        }];

        backfill_values(&mut samples, TerminalState::Win(crate::Player::PlayerA));
        assert_eq!(samples[0].value, 1.0);

        backfill_values(&mut samples, TerminalState::Win(crate::Player::PlayerB));
        assert_eq!(samples[0].value, -1.0);
    }

    #[test]
    fn test_backfill_values_draw() {
        let mut samples = vec![TrainingSample {
            player: crate::Player::PlayerA,
            notation: String::new(),
            policy: vec![],
            value: 0.5, // Should be overwritten
        }];

        backfill_values(&mut samples, TerminalState::Draw);
        assert_eq!(samples[0].value, 0.0);
    }
}
