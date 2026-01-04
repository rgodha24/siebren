use crate::{Action, Environment, Player, TerminalState};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TicTacToeAction(pub u8);

impl Action for TicTacToeAction {
    const NUM_ACTIONS: usize = 9;

    fn to_index(self) -> usize {
        self.0 as usize
    }

    fn from_index(index: usize) -> Option<Self> {
        (index < 9).then_some(TicTacToeAction(index as u8))
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct TicTacToe {
    pub board: [u8; 9], // 0 = empty, 1 = PlayerA (X), 2 = PlayerB (O)
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
    type Observation = [u8; 9];
    type Action = TicTacToeAction;
    type RollbackState = TicTacToeRollback;

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

    fn observation(&self) -> Self::Observation {
        self.board
    }

    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState {
        let cell = action.0;
        let previous_player = self.current_player;

        self.board[cell as usize] = match self.current_player {
            Player::PlayerA => 1,
            Player::PlayerB => 2,
        };
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
        assert_eq!(TicTacToeAction::NUM_ACTIONS, 9);
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
        assert_eq!(game.board[4], 1);
        assert_eq!(game.current_player(), Player::PlayerB);
        assert_eq!(game.valid_actions().count(), 8);

        game.rollback(rollback);
        assert_eq!(game.board[4], 0);
        assert_eq!(game.current_player(), Player::PlayerA);
        assert_eq!(game.valid_actions().count(), 9);
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
}
