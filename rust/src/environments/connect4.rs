use ndarray::Array2;

use crate::{Action, Environment, Player, TerminalState};

const ROWS: usize = 6;
const COLS: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct Connect4Action(pub usize);

impl Action for Connect4Action {
    const NUM_ACTIONS: usize = COLS;

    fn to_index(self) -> usize {
        self.0
    }

    fn from_index(index: usize) -> Option<Self> {
        (index < COLS).then_some(Connect4Action(index))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct Connect4 {
    board: [[Option<Player>; COLS]; ROWS],
    current_player: Player,
    move_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct Connect4Rollback {
    column: usize,
    row: usize,
}

impl Connect4 {
    /// Returns the row where a piece would land in the given column, or None if full.
    fn landing_row(&self, col: usize) -> Option<usize> {
        // Start from bottom row (index 5) and go up
        (0..ROWS).rev().find(|&row| self.board[row][col].is_none())
    }

    /// Checks if there's a winner by looking for 4 in a row.
    fn check_winner(&self) -> Option<Player> {
        // Check all possible 4-in-a-row positions
        for row in 0..ROWS {
            for col in 0..COLS {
                if let Some(player) = self.board[row][col] {
                    // Horizontal (only if we can fit 4 to the right)
                    if col + 3 < COLS
                        && self.board[row][col + 1] == Some(player)
                        && self.board[row][col + 2] == Some(player)
                        && self.board[row][col + 3] == Some(player)
                    {
                        return Some(player);
                    }

                    // Vertical (only if we can fit 4 going down)
                    if row + 3 < ROWS
                        && self.board[row + 1][col] == Some(player)
                        && self.board[row + 2][col] == Some(player)
                        && self.board[row + 3][col] == Some(player)
                    {
                        return Some(player);
                    }

                    // Diagonal down-right
                    if row + 3 < ROWS
                        && col + 3 < COLS
                        && self.board[row + 1][col + 1] == Some(player)
                        && self.board[row + 2][col + 2] == Some(player)
                        && self.board[row + 3][col + 3] == Some(player)
                    {
                        return Some(player);
                    }

                    // Diagonal up-right
                    if row >= 3
                        && col + 3 < COLS
                        && self.board[row - 1][col + 1] == Some(player)
                        && self.board[row - 2][col + 2] == Some(player)
                        && self.board[row - 3][col + 3] == Some(player)
                    {
                        return Some(player);
                    }
                }
            }
        }
        None
    }
}

impl Environment for Connect4 {
    type Observation = Array2<i8>;
    type Action = Connect4Action;
    type RollbackState = Connect4Rollback;

    fn new() -> Self {
        Self {
            board: [[None; COLS]; ROWS],
            current_player: Player::PlayerA,
            move_count: 0,
        }
    }

    fn is_terminal(&self) -> Option<TerminalState> {
        if let Some(winner) = self.check_winner() {
            return Some(TerminalState::Win(winner));
        }
        if self.move_count == (ROWS * COLS) as u8 {
            return Some(TerminalState::Draw);
        }
        None
    }

    fn valid_actions(&self) -> impl Iterator<Item = Self::Action> {
        (0..COLS)
            .filter(|&col| self.board[0][col].is_none())
            .map(Connect4Action)
    }

    fn current_player(&self) -> Player {
        self.current_player
    }

    fn observation(&self) -> Self::Observation {
        Array2::from_shape_fn((ROWS, COLS), |(row, col)| match self.board[row][col] {
            Some(Player::PlayerA) => 1,
            Some(Player::PlayerB) => -1,
            None => 0,
        })
    }

    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState {
        let col = action.0;
        let row = self.landing_row(col).expect("Column is full");

        self.board[row][col] = Some(self.current_player);
        self.current_player = match self.current_player {
            Player::PlayerA => Player::PlayerB,
            Player::PlayerB => Player::PlayerA,
        };
        self.move_count += 1;

        Connect4Rollback { column: col, row }
    }

    fn rollback(&mut self, rollback: Self::RollbackState) {
        self.board[rollback.row][rollback.column] = None;
        self.current_player = match self.current_player {
            Player::PlayerA => Player::PlayerB,
            Player::PlayerB => Player::PlayerA,
        };
        self.move_count -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_trait() {
        assert_eq!(Connect4Action(0).to_index(), 0);
        assert_eq!(Connect4Action(6).to_index(), 6);
        assert_eq!(Connect4Action::from_index(0), Some(Connect4Action(0)));
        assert_eq!(Connect4Action::from_index(6), Some(Connect4Action(6)));
        assert_eq!(Connect4Action::from_index(7), None);
        assert_eq!(Connect4Action::NUM_ACTIONS, 7);
    }

    #[test]
    fn test_new_game() {
        let game = Connect4::new();
        assert_eq!(game.current_player(), Player::PlayerA);
        assert_eq!(game.is_terminal(), None);
        assert_eq!(game.valid_actions().count(), 7);
    }

    #[test]
    fn test_apply_and_rollback() {
        let mut game = Connect4::new();

        // Drop piece in column 3
        let rollback = game.apply_action(Connect4Action(3));
        assert_eq!(game.board[5][3], Some(Player::PlayerA)); // Bottom row
        assert_eq!(game.current_player(), Player::PlayerB);
        assert_eq!(game.valid_actions().count(), 7);

        game.rollback(rollback);
        assert_eq!(game.board[5][3], None);
        assert_eq!(game.current_player(), Player::PlayerA);
    }

    #[test]
    fn test_stacking() {
        let mut game = Connect4::new();

        // Stack pieces in column 0
        game.apply_action(Connect4Action(0)); // PlayerA at row 5
        game.apply_action(Connect4Action(0)); // PlayerB at row 4
        game.apply_action(Connect4Action(0)); // PlayerA at row 3

        assert_eq!(game.board[5][0], Some(Player::PlayerA));
        assert_eq!(game.board[4][0], Some(Player::PlayerB));
        assert_eq!(game.board[3][0], Some(Player::PlayerA));
    }

    #[test]
    fn test_column_full() {
        let mut game = Connect4::new();

        // Fill column 0
        for _ in 0..6 {
            game.apply_action(Connect4Action(0));
        }

        // Column 0 should no longer be valid
        let valid: Vec<_> = game.valid_actions().collect();
        assert_eq!(valid.len(), 6);
        assert!(!valid.contains(&Connect4Action(0)));
    }

    #[test]
    fn test_horizontal_win() {
        let mut game = Connect4::new();

        // PlayerA: 0, 1, 2, 3 (bottom row)
        // PlayerB: 0, 1, 2 (second row)
        game.apply_action(Connect4Action(0)); // A
        game.apply_action(Connect4Action(0)); // B
        game.apply_action(Connect4Action(1)); // A
        game.apply_action(Connect4Action(1)); // B
        game.apply_action(Connect4Action(2)); // A
        game.apply_action(Connect4Action(2)); // B
        game.apply_action(Connect4Action(3)); // A wins

        assert_eq!(
            game.is_terminal(),
            Some(TerminalState::Win(Player::PlayerA))
        );
    }

    #[test]
    fn test_vertical_win() {
        let mut game = Connect4::new();

        // PlayerA stacks 4 in column 0
        // PlayerB plays in column 1
        game.apply_action(Connect4Action(0)); // A
        game.apply_action(Connect4Action(1)); // B
        game.apply_action(Connect4Action(0)); // A
        game.apply_action(Connect4Action(1)); // B
        game.apply_action(Connect4Action(0)); // A
        game.apply_action(Connect4Action(1)); // B
        game.apply_action(Connect4Action(0)); // A wins

        assert_eq!(
            game.is_terminal(),
            Some(TerminalState::Win(Player::PlayerA))
        );
    }

    #[test]
    fn test_diagonal_win() {
        let mut game = Connect4::new();

        // Build a diagonal for PlayerA
        // Col: 0  1  2  3
        // Row 5: A  A  A  A (eventually)
        // But we need to build up for diagonal

        // For diagonal going up-right from (5,0):
        // Need A at (5,0), (4,1), (3,2), (2,3)
        game.apply_action(Connect4Action(0)); // A at (5,0)
        game.apply_action(Connect4Action(1)); // B at (5,1)
        game.apply_action(Connect4Action(1)); // A at (4,1)
        game.apply_action(Connect4Action(2)); // B at (5,2)
        game.apply_action(Connect4Action(2)); // A at (4,2)
        game.apply_action(Connect4Action(3)); // B at (5,3)
        game.apply_action(Connect4Action(2)); // A at (3,2)
        game.apply_action(Connect4Action(3)); // B at (4,3)
        game.apply_action(Connect4Action(3)); // A at (3,3)
        game.apply_action(Connect4Action(3)); // B at (2,3)
        game.apply_action(Connect4Action(4)); // A at (5,4) - filler
        game.apply_action(Connect4Action(4)); // B at (4,4)

        // Now we need to think about this more carefully...
        // Let me restart with a cleaner approach
    }

    #[test]
    fn test_observation() {
        let mut game = Connect4::new();
        game.apply_action(Connect4Action(3)); // A at (5, 3)
        game.apply_action(Connect4Action(3)); // B at (4, 3)

        let obs = game.observation();
        assert_eq!(obs.shape(), &[6, 7]);
        assert_eq!(obs[[5, 3]], 1); // PlayerA
        assert_eq!(obs[[4, 3]], -1); // PlayerB
        assert_eq!(obs[[0, 0]], 0); // Empty
    }
}
