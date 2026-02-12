use ndarray::{ArrayViewMut, Ix2};
use serde::{Deserialize, Serialize};

use crate::{Environment, GameNotation, Player, TerminalState};

pub mod game;
pub mod map;
pub mod pen;
pub mod snake;
pub mod types;

pub use pen::{ByteFightPen, PenError};
pub use types::{ByteFightAction, ByteFightPolicyAction, Point};

#[derive(Debug, Clone, Copy, Default)]
struct PolicyValidMoves(u8);

#[derive(Debug, Clone, Copy)]
struct PolicyValidMovesIter {
    mask: u8,
    index: u8,
}

impl Iterator for PolicyValidMovesIter {
    type Item = ByteFightPolicyAction;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < ByteFight::NUM_ACTIONS as u8 {
            let idx = self.index;
            self.index += 1;
            if self.mask & (1 << idx) != 0 {
                return ByteFightPolicyAction::new(idx);
            }
        }
        None
    }
}

impl PolicyValidMoves {
    #[inline]
    fn add(&mut self, action: ByteFightPolicyAction) {
        self.0 |= 1 << (action as u8);
    }

    fn into_iter(self) -> PolicyValidMovesIter {
        PolicyValidMovesIter {
            mask: self.0,
            index: 0,
        }
    }
}

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

impl crate::Action for ByteFightPolicyAction {
    fn to_index(self) -> usize {
        self as usize
    }

    fn from_index(index: usize) -> Option<Self> {
        ByteFightPolicyAction::new(index as u8)
    }
}

impl Environment for ByteFight {
    type ObsElem = u8;
    type ObsDim = Ix2;
    type Action = ByteFightPolicyAction;
    type RollbackState = game::RollbackState;
    const NUM_ACTIONS: usize = 7;
    const OBS_SHAPE: Ix2 = Ix2(types::OBS_SERIALIZED_SIDE, types::OBS_SERIALIZED_WIDTH);

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
        let valid = self.board.get_valid_moves();
        let mut policy_moves = PolicyValidMoves::default();

        for action in [
            ByteFightPolicyAction::Forward,
            ByteFightPolicyAction::Left,
            ByteFightPolicyAction::LeftForward,
            ByteFightPolicyAction::Right,
            ByteFightPolicyAction::RightForward,
        ] {
            let absolute = self.policy_to_absolute(action);
            if valid.contains(absolute) {
                policy_moves.add(action);
            }
        }

        if valid.contains(ByteFightAction::Trap) {
            policy_moves.add(ByteFightPolicyAction::Trap);
        }
        if valid.contains(ByteFightAction::EndTurn) {
            policy_moves.add(ByteFightPolicyAction::EndTurn);
        }

        policy_moves.into_iter()
    }

    fn current_player(&self) -> Player {
        if self.board.is_player_a {
            Player::PlayerA
        } else {
            Player::PlayerB
        }
    }

    fn observation(&self, mut out: ArrayViewMut<u8, Ix2>) {
        let out_slice = out
            .as_slice_mut()
            .expect("bytefight observation output must be contiguous");

        let bitpacked = self.board.bitpacked_observation_16x16();
        out_slice[..types::OBS_CELLS].copy_from_slice(&bitpacked);

        let mut offset = types::OBS_CELLS;
        let direction = self.active_direction() as usize;
        for idx in 0..types::OBS_DIRECTIONS {
            out_slice[offset + idx] = if idx == direction { 1 } else { 0 };
        }
        offset += types::OBS_DIRECTIONS;

        let heuristics = self.board.heuristics();
        out_slice[offset..offset + types::OBS_HEURISTICS].copy_from_slice(&heuristics);
        offset += types::OBS_HEURISTICS;

        for byte in &mut out_slice[offset..] {
            *byte = 0;
        }
    }

    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState {
        let absolute = self.policy_to_absolute(action);
        self.board
            .apply_move(absolute)
            .expect("apply_action called with invalid action")
    }

    fn rollback(&mut self, rollback: Self::RollbackState) {
        self.board.rollback(rollback);
    }
}

impl ByteFight {
    fn active_direction(&self) -> ByteFightAction {
        let snake = if self.board.is_player_a {
            &self.board.snake_a
        } else {
            &self.board.snake_b
        };
        snake.current_direction.unwrap_or(ByteFightAction::North)
    }

    fn policy_to_absolute(&self, action: ByteFightPolicyAction) -> ByteFightAction {
        let direction = self.active_direction() as u8;
        match action {
            ByteFightPolicyAction::Forward => {
                ByteFightAction::new(direction).expect("valid direction")
            }
            ByteFightPolicyAction::Left => {
                ByteFightAction::new((direction + 6) % 8).expect("valid direction")
            }
            ByteFightPolicyAction::LeftForward => {
                ByteFightAction::new((direction + 7) % 8).expect("valid direction")
            }
            ByteFightPolicyAction::Right => {
                ByteFightAction::new((direction + 2) % 8).expect("valid direction")
            }
            ByteFightPolicyAction::RightForward => {
                ByteFightAction::new((direction + 1) % 8).expect("valid direction")
            }
            ByteFightPolicyAction::Trap => ByteFightAction::Trap,
            ByteFightPolicyAction::EndTurn => ByteFightAction::EndTurn,
        }
    }
}

impl GameNotation for ByteFight {
    type Error = PenError;

    fn to_notation(&self) -> String {
        ByteFightPen::from(self).0
    }

    fn from_notation(s: &str) -> Result<Self, Self::Error> {
        let pen = ByteFightPen(s.to_string());
        let board = pen.into_board()?;
        Ok(ByteFight { board })
    }
}

#[cfg(test)]
mod tests {
    use crate::Action;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_action_trait() {
        assert_eq!(ByteFightPolicyAction::Forward.to_index(), 0);
        assert_eq!(ByteFightPolicyAction::EndTurn.to_index(), 6);
        assert_eq!(
            ByteFightPolicyAction::from_index(0),
            Some(ByteFightPolicyAction::Forward)
        );
        assert_eq!(
            ByteFightPolicyAction::from_index(6),
            Some(ByteFightPolicyAction::EndTurn)
        );
        assert_eq!(ByteFightPolicyAction::from_index(7), None);
    }

    #[test]
    fn test_pen_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let board = game::Board::new_random(&mut rng);
        let pen = ByteFightPen::from(&board);
        let rebuilt = pen.clone().into_board().expect("valid pen");
        assert_eq!(pen.0, ByteFightPen::from(&rebuilt).0);
    }

    #[test]
    fn test_notation_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let game = ByteFight {
            board: game::Board::new_random(&mut rng),
        };

        let notation = game.to_notation();
        let restored = ByteFight::from_notation(&notation).expect("valid notation");

        // Compare via notation since board equality might differ in internal state
        assert_eq!(notation, restored.to_notation());
    }

    #[rstest]
    #[case(1)]
    #[case(7)]
    #[case(13)]
    #[case(23)]
    #[case(42)]
    #[case(77)]
    #[case(101)]
    #[case(123)]
    #[case(256)]
    #[case(999)]
    fn test_random_move_sequence_roundtrip(#[case] seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for _ in 0..5 {
            let mut board = game::Board::new_random(&mut rng);
            let mut snapshots = Vec::new();
            let mut snapshot_pens = Vec::new();
            snapshots.push(board.clone());
            snapshot_pens.push(ByteFightPen::from(&board).0);

            let mut rollbacks = Vec::new();

            for _ in 0..150 {
                let valid: Vec<_> = board.get_valid_moves().actions().collect();
                if valid.is_empty() {
                    break;
                }
                let idx = (rng.next_u64() as usize) % valid.len();
                let action = valid[idx];
                let rollback = board.apply_move(action).expect("valid move");
                rollbacks.push(rollback);
                snapshots.push(board.clone());
                snapshot_pens.push(ByteFightPen::from(&board).0);
            }

            for (idx, rollback) in rollbacks.into_iter().rev().enumerate() {
                board.rollback(rollback);
                let snapshot_idx = snapshots.len() - 2 - idx;
                assert_eq!(board, snapshots[snapshot_idx]);
                assert_eq!(ByteFightPen::from(&board).0, snapshot_pens[snapshot_idx]);
            }
        }
    }

    #[rstest]
    #[case(1)]
    #[case(42)]
    #[case(99)]
    fn test_notation_roundtrip_after_moves(#[case] seed: u64) {
        use crate::Environment;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game = ByteFight {
            board: game::Board::new_random(&mut rng),
        };

        // Apply some random moves
        for _ in 0..20 {
            let valid: Vec<_> = game.valid_actions().collect();
            if valid.is_empty() {
                break;
            }
            let idx = (rng.next_u64() as usize) % valid.len();
            game.apply_action(valid[idx]);
        }

        let notation = game.to_notation();
        let restored = ByteFight::from_notation(&notation).expect("valid notation");
        assert_eq!(notation, restored.to_notation());
    }

    #[test]
    fn test_relative_valid_actions_from_direction() {
        use crate::Environment;

        let game = ByteFight {
            board: game::Board::new_from_state(
                (5, 5),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![(2, 2)],
                1,
                2,
                0,
                Some(ByteFightAction::North),
                vec![(4, 4)],
                1,
                2,
                0,
                Some(ByteFightAction::North),
                0,
                1,
                true,
                0,
                9999,
                false,
            ),
        };

        let actions: Vec<_> = game.valid_actions().collect();
        assert_eq!(
            actions,
            vec![
                ByteFightPolicyAction::Forward,
                ByteFightPolicyAction::Left,
                ByteFightPolicyAction::LeftForward,
                ByteFightPolicyAction::Right,
                ByteFightPolicyAction::RightForward,
            ]
        );
    }
}
