use std::hash::{Hash, Hasher};

use super::types::Point;

#[derive(Debug, Clone)]
pub struct Map {
    wall_bitmask: [u32; 32],
    apple_bitmask: [u32; 32],
    dimensions: (usize, usize),
    trap_mask: [[i16; 32]; 32],
    portals: Vec<(Point, Point)>,
    turn_count: usize,
}

impl Map {
    pub fn new(dimensions: (usize, usize), turn_count: usize) -> Self {
        Map {
            apple_bitmask: [0; 32],
            wall_bitmask: [0; 32],
            dimensions,
            trap_mask: [[0; 32]; 32],
            portals: Vec::new(),
            turn_count,
        }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    pub fn is_empty(&self, point: Point) -> bool {
        self.wall_bitmask[point.y] & (1 << point.x) == 0
            && self.apple_bitmask[point.y] & (1 << point.x) == 0
    }

    pub fn is_wall(&self, point: Point) -> bool {
        self.wall_bitmask[point.y] & (1 << point.x) != 0
    }

    pub fn is_apple(&self, point: Point) -> bool {
        self.apple_bitmask[point.y] & (1 << point.x) != 0
    }

    pub fn portal(&self, point: Point) -> Option<Point> {
        self.portals.iter().find_map(|&(p1, p2)| {
            if p1 == point {
                Some(p2)
            } else if p2 == point {
                Some(p1)
            } else {
                None
            }
        })
    }

    pub fn portals(&self) -> &[(Point, Point)] {
        &self.portals
    }

    pub fn trap(&self, point: Point) -> i16 {
        self.trap_mask[point.y][point.x]
    }

    pub fn become_apple(&mut self, point: Point) {
        self.apple_bitmask[point.y] |= 1 << point.x;
    }

    pub fn become_empty(&mut self, point: Point) {
        self.wall_bitmask[point.y] &= !(1 << point.x);
        self.apple_bitmask[point.y] &= !(1 << point.x);
    }

    pub fn update_trap(&mut self, point: Point, trap: i16) {
        self.trap_mask[point.y][point.x] = trap;
    }

    pub fn become_wall(&mut self, point: Point) {
        self.wall_bitmask[point.y] |= 1 << point.x;
    }

    pub fn add_portal(&mut self, p1: Point, p2: Point) {
        self.portals.push((p1, p2));
    }

    pub fn bitmasks(&self) -> (&[u32; 32], &[u32; 32], [u32; 32], [u32; 32]) {
        let mut snake_a_traps: [u32; 32] = [0; 32];
        let mut snake_b_traps: [u32; 32] = [0; 32];

        for y in 0..32 {
            for x in 0..32 {
                if self.trap_mask[y][x] > self.turn_count as i16 {
                    snake_a_traps[y] |= 1 << x;
                } else if -self.trap_mask[y][x] > self.turn_count as i16 {
                    snake_b_traps[y] |= 1 << x;
                }
            }
        }

        (
            &self.wall_bitmask,
            &self.apple_bitmask,
            snake_a_traps,
            snake_b_traps,
        )
    }

    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    pub fn size(&self) -> usize {
        self.dimensions.0 * self.dimensions.1
    }

    pub fn move_forward_turn(&mut self) {
        self.turn_count += 1;
    }

    pub fn move_backward_turn(&mut self) {
        self.turn_count -= 1;
    }
}

impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.turn_count == other.turn_count
            && self.portals == other.portals
            && self.apple_bitmask == other.apple_bitmask
            && self.wall_bitmask == other.wall_bitmask
            && self
                .trap_mask
                .iter()
                .flatten()
                .zip(other.trap_mask.iter().flatten())
                .all(|(left, right)| {
                    (left.abs() <= self.turn_count as i16
                        && right.abs() <= self.turn_count as i16)
                        || left == right
                })
    }
}

impl Eq for Map {}

impl Hash for Map {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.wall_bitmask.hash(state);
        self.apple_bitmask.hash(state);
        self.dimensions.hash(state);
        self.portals.hash(state);
        self.turn_count.hash(state);
        for row in &self.trap_mask {
            for value in row {
                let normalized = if value.abs() <= self.turn_count as i16 {
                    0
                } else {
                    *value
                };
                normalized.hash(state);
            }
        }
    }
}

pub fn add_padding_walls(map: &mut Map) {
    let (width, height) = map.dimensions();
    for x in width..32 {
        for y in 0..32 {
            map.become_wall(Point { x, y });
        }
    }
    for y in height..32 {
        for x in 0..32 {
            map.become_wall(Point { x, y });
        }
    }
}
