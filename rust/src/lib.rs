use pyo3::prelude::*;
use std::hash::Hash;

pub mod mcts;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn siebren(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    PlayerA,
    PlayerB,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminalState {
    Win(Player),
    Draw,
}

/// An action that can be taken in an environment.
///
/// Actions must be convertible to/from a unique index in the range `0..NUM_ACTIONS`.
/// This is used for indexing into policy vectors from the neural network.
pub trait Action: Copy + Eq + Hash {
    /// Total number of possible actions (for policy vector sizing)
    const NUM_ACTIONS: usize;

    /// Convert this action to its unique index in `0..NUM_ACTIONS`
    fn to_index(self) -> usize;

    /// Convert an index back to an action. Returns None if index >= NUM_ACTIONS.
    fn from_index(index: usize) -> Option<Self>;
}

/// An environment implements a game, like Connect4, that we want to train a model to play.
///
/// These environments should be able to rollback state updates to step in and out of states
/// efficiently, with low copying costs.
pub trait Environment: Clone + Hash {
    /// Sent to the neural network evaluator
    type Observation;
    /// The action type for this environment
    type Action: Action;
    /// Used to rollback a move
    type RollbackState;

    /// Creates an environment. This should be randomly generated if possible on every call,
    /// to avoid the network overfitting on a single map.
    fn new() -> Self;

    /// Returns None if the game is still going, and Some(Win/Draw) if it's over.
    fn is_terminal(&self) -> Option<TerminalState>;

    /// Returns an iterator over the valid actions. Playing an invalid action means
    /// the current player loses.
    fn valid_actions(&self) -> impl Iterator<Item = Self::Action>;

    /// Returns the current player
    fn current_player(&self) -> Player;

    /// Generates the observation that is fed to the neural network
    fn observation(&self) -> Self::Observation;

    /// Modifies the environment to apply this action. Returns a rollback state which can be
    /// passed to the rollback function such that:
    ///
    /// `env == rollback(apply_action(env, a))`
    ///
    /// Implementors may assume that the action is valid as defined by `valid_actions`.
    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState;

    /// Rollback function (see docs for `apply_action`)
    fn rollback(&mut self, rollback: Self::RollbackState);
}
