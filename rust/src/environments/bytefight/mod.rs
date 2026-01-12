use serde::{Deserialize, Serialize};

use crate::{Environment, Player, TerminalState};

pub mod game;
pub mod map;
pub mod pen;
pub mod snake;
pub mod types;

pub use pen::ByteFightPen;
pub use types::{ByteFightAction, Point};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "ByteFightPen", into = "ByteFightPen")]
pub struct ByteFight {
    board: game::Board,
}

impl From<ByteFightPen> for ByteFight {
    fn from(pen: ByteFightPen) -> Self {
        ByteFight {
            board: pen.into_board().expect("invalid ByteFight PEN"),
        }
    }
}

impl From<ByteFight> for ByteFightPen {
    fn from(board: ByteFight) -> Self {
        ByteFightPen::from(&board.board)
    }
}

impl From<&ByteFight> for ByteFightPen {
    fn from(board: &ByteFight) -> Self {
        ByteFightPen::from(&board.board)
    }
}

impl crate::Action for ByteFightAction {
    fn to_index(self) -> usize {
        self as usize
    }

    fn from_index(index: usize) -> Option<Self> {
        ByteFightAction::new(index as u8)
    }
}

impl Environment for ByteFight {
    type Observation = types::BoardHeuristics;
    type Action = ByteFightAction;
    type RollbackState = game::RollbackState;
    const NUM_ACTIONS: usize = 11;

    fn new() -> Self {
        let mut rng = rand::rng();
        let board = game::Board::new_random(&mut rng);
        ByteFight::from(ByteFightPen::from(&board))
    }

    fn is_terminal(&self) -> Option<TerminalState> {
        match self.board.terminal_state()? {
            types::TerminalState::PlayerAWin => Some(TerminalState::Win(Player::PlayerA)),
            types::TerminalState::PlayerBWin => Some(TerminalState::Win(Player::PlayerB)),
            types::TerminalState::Draw => Some(TerminalState::Draw),
        }
    }

    fn valid_actions(&self) -> impl Iterator<Item = Self::Action> {
        self.board.get_valid_moves().actions()
    }

    fn current_player(&self) -> Player {
        if self.board.is_player_a {
            Player::PlayerA
        } else {
            Player::PlayerB
        }
    }

    fn observation(&self) -> Self::Observation {
        self.board.heuristics()
    }

    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState {
        self.board
            .apply_move(action)
            .expect("apply_action called with invalid action")
    }

    fn rollback(&mut self, rollback: Self::RollbackState) {
        self.board.rollback(rollback);
    }
}

#[cfg(test)]
mod tests {
    use crate::Action;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_action_trait() {
        assert_eq!(ByteFightAction::North.to_index(), 0);
        assert_eq!(ByteFightAction::EndTurn.to_index(), 10);
        assert_eq!(ByteFightAction::from_index(0), Some(ByteFightAction::North));
        assert_eq!(
            ByteFightAction::from_index(10),
            Some(ByteFightAction::EndTurn)
        );
        assert_eq!(ByteFightAction::from_index(11), None);
    }

    #[test]
    fn test_pen_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let board = game::Board::new_random(&mut rng);
        let pen = ByteFightPen::from(&board);
        let rebuilt = pen.clone().into_board().expect("valid pen");
        assert_eq!(pen.0, ByteFightPen::from(&rebuilt).0);
    }
}
