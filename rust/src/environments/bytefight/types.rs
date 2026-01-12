use std::hash::Hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

impl Point {
    pub fn new(x: usize, y: usize) -> Option<Self> {
        if x >= 32 || y >= 32 {
            return None;
        }
        Some(Point { x, y })
    }

    pub fn try_add(self, action: ByteFightAction) -> Option<Self> {
        match action {
            ByteFightAction::North if self.y != 0 => Point::new(self.x, self.y - 1),
            ByteFightAction::Northeast if self.y != 0 => Point::new(self.x + 1, self.y - 1),
            ByteFightAction::East => Point::new(self.x + 1, self.y),
            ByteFightAction::Southeast => Point::new(self.x + 1, self.y + 1),
            ByteFightAction::South => Point::new(self.x, self.y + 1),
            ByteFightAction::Southwest if self.x != 0 => Point::new(self.x - 1, self.y + 1),
            ByteFightAction::West if self.x != 0 => Point::new(self.x - 1, self.y),
            ByteFightAction::Northwest if self.x != 0 && self.y != 0 => {
                Point::new(self.x - 1, self.y - 1)
            }
            ByteFightAction::Trap | ByteFightAction::FF | ByteFightAction::EndTurn => {
                panic!("invalid move {action:?} being added to point")
            }
            _ => None,
        }
    }

    pub fn try_add_int(self, action: u8) -> Option<Self> {
        self.try_add(ByteFightAction::new(action)?)
    }
}

impl From<(usize, usize)> for Point {
    fn from((x, y): (usize, usize)) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ByteFightAction {
    North = 0,
    Northeast = 1,
    East = 2,
    Southeast = 3,
    South = 4,
    Southwest = 5,
    West = 6,
    Northwest = 7,
    Trap = 8,
    FF = 9,
    EndTurn = 10,
}

impl ByteFightAction {
    pub fn new(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::North),
            1 => Some(Self::Northeast),
            2 => Some(Self::East),
            3 => Some(Self::Southeast),
            4 => Some(Self::South),
            5 => Some(Self::Southwest),
            6 => Some(Self::West),
            7 => Some(Self::Northwest),
            8 => Some(Self::Trap),
            9 => Some(Self::FF),
            10 => Some(Self::EndTurn),
            _ => None,
        }
    }

    pub fn to_val(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalState {
    PlayerAWin,
    PlayerBWin,
    Draw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct ValidMoves(u16);

impl ValidMoves {
    #[inline]
    pub fn add(&mut self, action: ByteFightAction) {
        self.0 |= 1 << (action as u16);
    }

    #[inline]
    pub fn remove(&mut self, action: ByteFightAction) {
        self.0 &= !(1 << (action as u16));
    }

    #[inline]
    pub fn contains(&self, action: ByteFightAction) -> bool {
        self.0 & (1 << (action as u16)) != 0
    }

    #[inline]
    pub fn amount(&self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub fn get_move_bounds(&self) -> (usize, usize) {
        let mut start: usize = 0;
        let mut end: usize = 0;

        for i in 0..8 {
            if self.0 & (1 << i) == 0 {
                continue;
            }
            if start == 0 {
                start = i;
            }
            end = i;
        }

        (start, end)
    }

    pub fn actions(&self) -> impl Iterator<Item = ByteFightAction> + '_ {
        (0..=10)
            .filter(|i| self.0 & (1 << i) != 0)
            .map(|i| ByteFightAction::new(i as u8).expect("0..=10 is always a valid action"))
    }
}

pub const HEURISTICS_SIZE: usize = 18;

pub type BoardHeuristics = [f32; HEURISTICS_SIZE];
