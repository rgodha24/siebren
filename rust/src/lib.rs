use pyo3::prelude::*;
use std::hash::Hash;

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

#[derive(Clone, Copy, PartialEq)]
enum Player {
    PlayerA,
    PlayerB,
}
#[derive(Clone, Copy, PartialEq)]
enum TerminalState {
    Win(Player),
    Draw,
}

/// an environment implements a game, like Connect4, that we want to train a model to play.
///
/// these environments should be able to rollback state updates to step in and out of states
/// efficiently, with low copying costs.
trait Environment: Clone + Hash {
    /// sent to the neural network evaluator
    type Observation;
    /// this should be a cheap type, like an enum/usize. It should impl AsRef<usize>,
    /// where the usize should be unique and in the range 0<=n<=NUM_ACTIONS;
    type Action: Copy + Eq + Hash + AsRef<usize>;
    /// used to rollback a move
    type RollbackState;
    const NUM_ACTIONS: usize;

    /// creates an environment. this should be ~randomly generated if possible on every call, to
    /// avoid the network overfitting on a single map.
    fn new() -> Self;
    /// returns None if the game is still going, and Some(PlayerAWin/PlayerBWin/Draw) if its over.
    fn is_terminal(&self) -> Option<TerminalState>;

    /// returns an iterator over the valid actions. playing an
    /// invalid action means the current player loses.
    fn valid_actions(&self) -> impl Iterator<Item = Self::Action>;
    /// current_player
    fn current_player(&self) -> Player;
    /// generates the observation that is fed to the neural network
    fn observation(&self) -> Self::Observation;

    /// modifies the environment to accept this action. this returns a rollback state, which can be
    /// passed to the rollback function, such that:
    ///
    /// E == rollback(apply_action(E, a))
    ///
    /// implementors of this function may assume that the action is a valid action, as defined by
    /// the valid_actions function.
    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState;

    /// rollback function (look at docs for apply_action)
    fn rollback(&mut self, rollback: Self::RollbackState);
}
