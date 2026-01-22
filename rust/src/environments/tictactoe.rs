use ndarray::{ArrayView1, ArrayViewMut, Ix1};
use std::fmt;

use crate::{Action, Environment, GameNotation, Player, TerminalState};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TicTacToeAction(pub u8);

impl Action for TicTacToeAction {
    fn to_index(self) -> usize {
        self.0 as usize
    }

    fn from_index(index: usize) -> Option<Self> {
        (index < 9).then_some(TicTacToeAction(index as u8))
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct TicTacToe {
    /// Board state: 0 = empty, 1 = PlayerA (X), -1 = PlayerB (O)
    pub board: [i8; 9],
    current_player: Player,
    move_count: u8,
}

impl TicTacToe {
    const WIN_PATTERNS: [[usize; 3]; 8] = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ];

    pub fn check_winner(&self) -> Option<Player> {
        for pattern in &Self::WIN_PATTERNS {
            let a = self.board[pattern[0]];
            let b = self.board[pattern[1]];
            let c = self.board[pattern[2]];
            if a != 0 && a == b && b == c {
                return Some(if a == 1 {
                    Player::PlayerA
                } else {
                    Player::PlayerB
                });
            }
        }
        None
    }
}

pub struct TicTacToeRollback {
    cell: u8,
    previous_player: Player,
}

impl Environment for TicTacToe {
    type ObsElem = i8;
    type ObsDim = Ix1;
    type Action = TicTacToeAction;
    type RollbackState = TicTacToeRollback;
    const NUM_ACTIONS: usize = 9;
    const OBS_SHAPE: Ix1 = Ix1(9);

    fn new() -> Self {
        TicTacToe {
            board: [0; 9],
            current_player: Player::PlayerA,
            move_count: 0,
        }
    }

    fn is_terminal(&self) -> Option<TerminalState> {
        if let Some(winner) = self.check_winner() {
            return Some(TerminalState::Win(winner));
        }
        if self.move_count == 9 {
            return Some(TerminalState::Draw);
        }
        None
    }

    fn valid_actions(&self) -> impl Iterator<Item = Self::Action> {
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &cell)| cell == 0)
            .map(|(i, _)| TicTacToeAction(i as u8))
    }

    fn current_player(&self) -> Player {
        self.current_player
    }

    fn observation(&self, mut out: ArrayViewMut<i8, Ix1>) {
        out.assign(&ArrayView1::from(&self.board));
    }

    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState {
        let cell = action.0;
        let previous_player = self.current_player;

        // 1 = PlayerA, -1 = PlayerB
        self.board[cell as usize] = self.current_player as i8;
        self.current_player = match self.current_player {
            Player::PlayerA => Player::PlayerB,
            Player::PlayerB => Player::PlayerA,
        };
        self.move_count += 1;

        TicTacToeRollback {
            cell,
            previous_player,
        }
    }

    fn rollback(&mut self, rollback: Self::RollbackState) {
        self.board[rollback.cell as usize] = 0;
        self.current_player = rollback.previous_player;
        self.move_count -= 1;
    }
}

#[derive(Debug)]
pub struct TicTacToeNotationError(String);

impl fmt::Display for TicTacToeNotationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TicTacToeNotationError {}

impl GameNotation for TicTacToe {
    type Error = TicTacToeNotationError;

    /// Format: "XO_X_O___|A" (9 cells + current player)
    /// X=PlayerA, O=PlayerB, _=empty
    fn to_notation(&self) -> String {
        let mut s = String::with_capacity(11);
        for &cell in &self.board {
            s.push(match cell {
                1 => 'X',
                -1 => 'O',
                _ => '_',
            });
        }
        s.push('|');
        s.push(match self.current_player {
            Player::PlayerA => 'A',
            Player::PlayerB => 'B',
        });
        s
    }

    fn from_notation(s: &str) -> Result<Self, Self::Error> {
        let parts: Vec<&str> = s.split('|').collect();
        if parts.len() != 2 {
            return Err(TicTacToeNotationError(
                "expected format: BOARD|PLAYER".into(),
            ));
        }

        let board_str = parts[0];
        let player_str = parts[1];

        if board_str.len() != 9 {
            return Err(TicTacToeNotationError("board must have 9 cells".into()));
        }

        let mut board = [0i8; 9];
        let mut move_count = 0u8;
        for (i, ch) in board_str.chars().enumerate() {
            board[i] = match ch {
                'X' => {
                    move_count += 1;
                    1
                }
                'O' => {
                    move_count += 1;
                    -1
                }
                '_' => 0,
                _ => return Err(TicTacToeNotationError(format!("invalid cell char: {}", ch))),
            };
        }

        let current_player = match player_str {
            "A" => Player::PlayerA,
            "B" => Player::PlayerB,
            _ => return Err(TicTacToeNotationError("player must be A or B".into())),
        };

        Ok(TicTacToe {
            board,
            current_player,
            move_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_trait() {
        assert_eq!(TicTacToeAction(0).to_index(), 0);
        assert_eq!(TicTacToeAction(8).to_index(), 8);
        assert_eq!(TicTacToeAction::from_index(0), Some(TicTacToeAction(0)));
        assert_eq!(TicTacToeAction::from_index(8), Some(TicTacToeAction(8)));
        assert_eq!(TicTacToeAction::from_index(9), None);
        assert_eq!(TicTacToe::NUM_ACTIONS, 9);
    }

    #[test]
    fn test_new_game() {
        let game = TicTacToe::new();
        assert_eq!(game.current_player(), Player::PlayerA);
        assert_eq!(game.is_terminal(), None);
        assert_eq!(game.valid_actions().count(), 9);
    }

    #[test]
    fn test_apply_and_rollback() {
        let mut game = TicTacToe::new();

        let rollback = game.apply_action(TicTacToeAction(4));
        assert_eq!(game.board[4], 1); // PlayerA = 1
        assert_eq!(game.current_player(), Player::PlayerB);
        assert_eq!(game.valid_actions().count(), 8);

        game.rollback(rollback);
        assert_eq!(game.board[4], 0);
        assert_eq!(game.current_player(), Player::PlayerA);
        assert_eq!(game.valid_actions().count(), 9);
    }

    #[test]
    fn test_player_b_moves() {
        let mut game = TicTacToe::new();

        game.apply_action(TicTacToeAction(0)); // PlayerA
        game.apply_action(TicTacToeAction(4)); // PlayerB

        assert_eq!(game.board[0], 1); // PlayerA = 1
        assert_eq!(game.board[4], -1); // PlayerB = -1
    }

    #[test]
    fn test_win_detection() {
        let mut game = TicTacToe::new();

        // X wins with top row
        game.apply_action(TicTacToeAction(0));
        game.apply_action(TicTacToeAction(3));
        game.apply_action(TicTacToeAction(1));
        game.apply_action(TicTacToeAction(4));
        game.apply_action(TicTacToeAction(2));

        assert_eq!(
            game.is_terminal(),
            Some(TerminalState::Win(Player::PlayerA))
        );
    }

    #[test]
    fn test_draw() {
        let mut game = TicTacToe::new();

        // X O X
        // X O O
        // O X X
        let moves = [0, 1, 2, 4, 3, 5, 7, 6, 8];
        for &m in &moves {
            game.apply_action(TicTacToeAction(m));
        }

        assert_eq!(game.is_terminal(), Some(TerminalState::Draw));
    }

    #[test]
    fn test_notation_roundtrip() {
        // Test empty board
        let game = TicTacToe::new();
        let notation = game.to_notation();
        assert_eq!(notation, "_________|A");
        let restored = TicTacToe::from_notation(&notation).unwrap();
        assert_eq!(game, restored);

        // Test after some moves
        let mut game = TicTacToe::new();
        game.apply_action(TicTacToeAction(0)); // X at 0
        game.apply_action(TicTacToeAction(4)); // O at 4
        game.apply_action(TicTacToeAction(8)); // X at 8

        let notation = game.to_notation();
        assert_eq!(notation, "X___O___X|B");
        let restored = TicTacToe::from_notation(&notation).unwrap();
        assert_eq!(game, restored);
    }

    #[test]
    fn test_notation_errors() {
        assert!(TicTacToe::from_notation("invalid").is_err());
        assert!(TicTacToe::from_notation("XXXXXXXX|A").is_err()); // 8 cells
        assert!(TicTacToe::from_notation("_________|C").is_err()); // invalid player
        assert!(TicTacToe::from_notation("____Z____|A").is_err()); // invalid char
    }
}
