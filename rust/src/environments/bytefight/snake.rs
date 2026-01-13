use std::collections::VecDeque;

use super::types::{ByteFightAction, Point};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Snake {
    pub max_length_reached: usize,
    pub queued_length: usize,
    pub traps_this_turn: usize,
    pub current_direction: Option<ByteFightAction>,
    pub segment_queue: VecDeque<Point>,
    pub sacrifice: usize,
    pub total_apples: usize,
}

impl Snake {
    pub fn can_afford_movement(&self, min_player_size: usize) -> bool {
        self.sacrifice - 1 <= self.length() - min_player_size
    }

    pub fn eat_apple(&mut self) {
        self.queued_length += 2;
        self.max_length_reached = self.max_length_reached.max(self.length());
        self.total_apples += 1;
    }

    pub fn removed_on_point_sacrifice(&self, point: &Point) -> bool {
        let cells_lost = if self.sacrifice >= self.queued_length {
            self.sacrifice - self.queued_length
        } else {
            0
        };
        for i in 0..std::cmp::min(cells_lost, self.segment_queue.len()) {
            if self
                .segment_queue
                .get(self.segment_queue.len() - 1 - i)
                .is_some_and(|p2| p2 == point)
            {
                return true;
            }
        }
        false
    }

    pub fn apply_sacrifice(&mut self, sacrifice: usize) -> Result<Vec<Point>, ()> {
        let cells_lost = if sacrifice <= self.queued_length {
            self.queued_length -= sacrifice;
            0
        } else {
            let cells_lost = sacrifice - self.queued_length;
            self.queued_length = 0;
            cells_lost
        };

        if cells_lost >= self.segment_queue.len() {
            return Err(());
        }

        let cells_lost = (0..cells_lost)
            .map(|_| self.segment_queue.pop_back().expect("segment queue empty"))
            .collect();

        Ok(cells_lost)
    }

    pub fn push_move(&mut self, action: ByteFightAction) -> Result<(Point, Vec<Point>), ()> {
        let cells_lost = self.apply_sacrifice(self.sacrifice)?;
        self.sacrifice += 1;
        self.current_direction = Some(action);

        Ok((self.segment_queue[0].try_add(action).unwrap(), cells_lost))
    }

    pub fn can_place_trap(&self, min_player_size: usize) -> bool {
        let max_traps = self.max_length_reached / 2;

        (max_traps > self.traps_this_turn)
            && (self.segment_queue.len() > 2)
            && (self.length() > min_player_size)
    }

    pub fn length(&self) -> usize {
        self.segment_queue.len() + self.queued_length
    }
}
