use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

use rand::seq::SliceRandom;
use rand::Rng;

use super::map::{add_padding_walls, Map};
use super::snake::Snake;
use super::types::{
    BitpackedObservation, ByteFightAction, Point, TerminalState, ValidMoves, OBS_CELLS, OBS_SIDE,
};

pub const APPLE_REWARD: usize = 2;
const TRAP_LIFETIME: i16 = 100;
pub const TRAP_SACRIFICE: usize = 3;
const DECAY_TIMELINE: [(usize, usize); 4] = [(1000, 15), (1600, 10), (1800, 5), (1950, 2)];
const DECAY_NOT_APPLIED_PLACEHOLDER: usize = 9999;

pub const LAST_TURN: usize = 2000;

// 16x16-only map set for training/selection.
// These are the only currently-defined maps that fit within a 16x16 observation window.
// Excluded (commented out from MAPS_JSON): pillars, great_divide, empty_large, ssspline,
// combustible_lemons, arena, ladder, compasss, diamonds, ssspiral, lol, attrition.
const MAPS_JSON: &str = r#"{
    "cage": "11,11#1,5#9,5#5#2##30,1,Vertical#1010101010101010101010101010101010101010101000101010100000101010000010101010001010101010101010101010101010101010101010101#0",
    "empty": "9,9#1,4#7,4#5#2##20,1,Vertical#000000000000000000000000000000000000000000000000000000000000000000000000000000000#0",
    "recurve": "13,13#2,6#10,6#4#2#6,0,6,12_6,12,6,0#50,1,Vertical#0100000000010001000000010000010000010000000100010000000010001000000000000000000000000000000000000000000000001000100000000100010000000100000100000100000001000100000000010#0"
  }"#;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    pub map: Map,
    pub apple_timeline: Vec<(usize, Point)>,
    pub apple_timeline_ptr: usize,
    pub snake_a: Snake,
    pub snake_b: Snake,
    pub is_player_a: bool,
    pub min_player_size: usize,
    pub(crate) decay_countdown: usize,
    pub(crate) cached_decay_interval: usize,
    pub(crate) is_decaying: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackState {
    EndTurn {
        old_num_traps: usize,
        old_sacrifice_val: usize,
        prev_cached_decay_interval: usize,
        prev_decay_countdown: usize,
        decayed_point: Option<Vec<Point>>,
        snake_a_ate_during_collision: bool,
        snake_b_ate_during_collision: bool,
        snake_a_max_len: usize,
        snake_b_max_len: usize,
        prev_apple_timeline_ptr: usize,
        apples_placed_index: Vec<usize>,
    },
    ApplyMove {
        prev_trap_val: i16,
        sacrificed_points: Vec<Point>,
        prev_queued_length: usize,
        prev_max_length_reached: usize,
        prev_direction: Option<u8>,
        head_was_apple: bool,
    },
    ApplyTrap {
        old_trap_val: i16,
        trap: Point,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symmetry {
    Horizontal,
    Vertical,
    Origin,
}

impl Symmetry {
    fn reflect(self, point: Point, width: usize, height: usize) -> Point {
        match self {
            Symmetry::Horizontal => Point {
                x: point.x,
                y: height - 1 - point.y,
            },
            Symmetry::Vertical => Point {
                x: width - 1 - point.x,
                y: point.y,
            },
            Symmetry::Origin => Point {
                x: width - 1 - point.x,
                y: height - 1 - point.y,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AppleSpec {
    Timeline(Vec<(usize, Point)>),
    Spawn {
        rate: usize,
        count: usize,
        symmetry: Symmetry,
    },
}

#[derive(Debug, Clone)]
struct MapDefinition {
    width: usize,
    height: usize,
    start_a: Point,
    start_b: Point,
    start_size: usize,
    min_player_size: usize,
    portals: Vec<(Point, Point)>,
    walls: Vec<Point>,
    apple_spec: AppleSpec,
}

impl MapDefinition {
    fn from_map_string(map_str: &str) -> Result<Self, String> {
        let parts: Vec<&str> = map_str.split('#').collect();
        if parts.len() != 9 {
            return Err(format!("expected 9 map parts, got {}", parts.len()));
        }

        let (width, height) = parse_pair(parts[0], ',')?;
        let start_a = parse_point(parts[1])?;
        let start_b = parse_point(parts[2])?;
        let start_size = parse_usize(parts[3], "start_size")?;
        let min_player_size = parse_usize(parts[4], "min_player_size")?;
        let portals = parse_portals_section(parts[5])?;
        let walls = parse_walls_bits(parts[7], width, height)?;

        let is_record = parse_usize(parts[8], "is_record")? == 1;
        let apple_spec = if is_record {
            let timeline = parse_apple_timeline(parts[6])?;
            AppleSpec::Timeline(timeline)
        } else {
            let stats: Vec<&str> = parts[6].split(',').collect();
            if stats.len() != 3 {
                return Err("invalid apple stats section".to_string());
            }
            let rate = parse_usize(stats[0], "apple_rate")?;
            let count = parse_usize(stats[1], "num_apples")?;
            let symmetry = match stats[2] {
                "Horizontal" => Symmetry::Horizontal,
                "Vertical" => Symmetry::Vertical,
                "Origin" => Symmetry::Origin,
                _ => return Err("unknown symmetry".to_string()),
            };
            AppleSpec::Spawn {
                rate,
                count,
                symmetry,
            }
        };

        Ok(MapDefinition {
            width,
            height,
            start_a,
            start_b,
            start_size,
            min_player_size,
            portals,
            walls,
            apple_spec,
        })
    }

    fn build_initial_state(&self, rng: &mut impl Rng) -> Board {
        let apple_timeline = self.apple_spec.generate_timeline(
            rng,
            self.width,
            self.height,
            &self.portals,
            &self.walls,
            self.start_a,
            self.start_b,
        );

        let apples_now: Vec<Point> = apple_timeline
            .iter()
            .filter(|(turn, _)| *turn == 0)
            .map(|(_, point)| *point)
            .collect();

        let snake_a = Snake {
            sacrifice: 1,
            max_length_reached: self.start_size,
            queued_length: self.start_size.saturating_sub(1),
            traps_this_turn: 0,
            current_direction: ByteFightAction::new((rng.next_u64() % 8) as u8),
            segment_queue: VecDeque::from([self.start_a]),
            total_apples: 0,
        };
        let snake_b = Snake {
            sacrifice: 1,
            max_length_reached: self.start_size,
            queued_length: self.start_size.saturating_sub(1),
            traps_this_turn: 0,
            current_direction: ByteFightAction::new((rng.next_u64() % 8) as u8),
            segment_queue: VecDeque::from([self.start_b]),
            total_apples: 0,
        };

        let mut map = Map::new((self.width, self.height), 0);
        for wall in &self.walls {
            map.become_wall(*wall);
        }
        for (p1, p2) in &self.portals {
            map.add_portal(*p1, *p2);
        }
        for apple in &apples_now {
            map.become_apple(*apple);
            if let Some(portal) = map.portal(*apple) {
                map.become_apple(portal);
            }
        }
        add_padding_walls(&mut map);

        let mut board = Board {
            map,
            apple_timeline,
            apple_timeline_ptr: 0,
            snake_a,
            snake_b,
            is_player_a: true,
            min_player_size: self.min_player_size,
            decay_countdown: 0,
            cached_decay_interval: DECAY_NOT_APPLIED_PLACEHOLDER,
            is_decaying: false,
        };

        board.fix_apple_head_collisions();
        let _ = board.apply_decay();
        board
    }
}

impl AppleSpec {
    fn generate_timeline(
        &self,
        rng: &mut impl Rng,
        width: usize,
        height: usize,
        portals: &[(Point, Point)],
        walls: &[Point],
        start_a: Point,
        start_b: Point,
    ) -> Vec<(usize, Point)> {
        match self {
            AppleSpec::Timeline(timeline) => timeline.clone(),
            AppleSpec::Spawn {
                rate,
                count,
                symmetry,
            } => {
                let portal_map = portal_lookup(portals);
                let wall_set: HashSet<Point> = walls.iter().copied().collect();
                let mut considered: HashSet<Point> = HashSet::new();
                considered.insert(start_a);
                considered.insert(start_b);

                let mut select_from = Vec::new();
                for y in 0..height {
                    for x in 0..width {
                        let point = Point { x, y };
                        if wall_set.contains(&point) {
                            continue;
                        }
                        if considered.contains(&point) {
                            continue;
                        }
                        select_from.push(point);
                        considered.insert(point);
                        considered.insert(symmetry.reflect(point, width, height));
                    }
                }

                let mut apples = Vec::new();
                let mut first_round = select_from.clone();
                add_apple_spawns(
                    &mut first_round,
                    *count,
                    *symmetry,
                    &portal_map,
                    &mut apples,
                    0,
                    width,
                    height,
                    rng,
                );

                let mut later_round = select_from;
                later_round.push(start_a);
                later_round.push(start_b);

                let mut spawn_round = *rate;
                while spawn_round < LAST_TURN {
                    let mut picks = later_round.clone();
                    add_apple_spawns(
                        &mut picks,
                        *count,
                        *symmetry,
                        &portal_map,
                        &mut apples,
                        spawn_round,
                        width,
                        height,
                        rng,
                    );
                    spawn_round += *rate;
                }

                apples
            }
        }
    }
}

fn portal_lookup(portals: &[(Point, Point)]) -> HashMap<Point, Point> {
    let mut map = HashMap::new();
    for (p1, p2) in portals {
        map.insert(*p1, *p2);
        map.insert(*p2, *p1);
    }
    map
}

fn add_apple_spawns(
    picks: &mut Vec<Point>,
    count: usize,
    symmetry: Symmetry,
    portals: &HashMap<Point, Point>,
    apples: &mut Vec<(usize, Point)>,
    turn_num: usize,
    width: usize,
    height: usize,
    rng: &mut impl Rng,
) {
    picks.shuffle(rng);
    let mut apple_count = 0;
    let mut idx = 0;

    while apple_count < count && idx < picks.len() {
        let point = picks[idx];
        let reflection = symmetry.reflect(point, width, height);

        if point == reflection {
            apple_count += 1;
            apples.push((turn_num, point));
        } else {
            apple_count += 2;
            apples.push((turn_num, point));
            apples.push((turn_num, reflection));
        }

        if let Some(portal) = portals.get(&point) {
            apples.push((turn_num, *portal));
            if point != reflection {
                if let Some(ref_portal) = portals.get(&reflection) {
                    apples.push((turn_num, *ref_portal));
                }
            }
        }

        idx += 1;
    }
}

fn map_definitions() -> &'static Vec<MapDefinition> {
    static MAPS: OnceLock<Vec<MapDefinition>> = OnceLock::new();
    MAPS.get_or_init(|| {
        let parsed: HashMap<String, String> =
            serde_json::from_str(MAPS_JSON).expect("invalid bytefight maps json");
        parsed
            .into_values()
            .map(|map_str| MapDefinition::from_map_string(&map_str))
            .collect::<Result<Vec<_>, _>>()
            .expect("invalid bytefight map string")
    })
}

impl Board {
    pub fn new_random(rng: &mut impl Rng) -> Self {
        let maps = map_definitions();
        let valid_maps: Vec<&MapDefinition> = maps
            .iter()
            .filter(|map| map.width <= OBS_SIDE && map.height <= OBS_SIDE)
            .collect();
        let len = valid_maps.len();
        assert!(len > 0, "no bytefight maps with dimensions <= 16x16");
        let idx = (rng.next_u64() as usize) % len;
        let map = valid_maps[idx];
        map.build_initial_state(rng)
    }

    pub fn bitpacked_observation_16x16(&self) -> BitpackedObservation {
        const WALL_BIT: u8 = 1 << 0;
        const APPLE_BIT: u8 = 1 << 1;
        const OWN_BODY_BIT: u8 = 1 << 2;
        const OWN_HEAD_BIT: u8 = 1 << 3;
        const OWN_TRAP_BIT: u8 = 1 << 4;
        const OPP_BODY_BIT: u8 = 1 << 5;
        const OPP_HEAD_BIT: u8 = 1 << 6;
        const OPP_TRAP_BIT: u8 = 1 << 7;

        let (wall_bitmask, apple_bitmask, snake_a_traps, snake_b_traps) = self.map.bitmasks();

        let (own_snake, opp_snake, own_traps, opp_traps) = if self.is_player_a {
            (
                &self.snake_a,
                &self.snake_b,
                snake_a_traps.as_slice(),
                snake_b_traps.as_slice(),
            )
        } else {
            (
                &self.snake_b,
                &self.snake_a,
                snake_b_traps.as_slice(),
                snake_a_traps.as_slice(),
            )
        };

        let mut obs: BitpackedObservation = [0; OBS_CELLS];
        for y in 0..OBS_SIDE {
            let walls = wall_bitmask[y];
            let apples = apple_bitmask[y];
            let own_trap_row = own_traps[y];
            let opp_trap_row = opp_traps[y];
            for x in 0..OBS_SIDE {
                let mask = 1u32 << x;
                let mut cell = 0u8;
                if walls & mask != 0 {
                    cell |= WALL_BIT;
                }
                if apples & mask != 0 {
                    cell |= APPLE_BIT;
                }
                if own_trap_row & mask != 0 {
                    cell |= OWN_TRAP_BIT;
                }
                if opp_trap_row & mask != 0 {
                    cell |= OPP_TRAP_BIT;
                }
                obs[y * OBS_SIDE + x] = cell;
            }
        }

        for (i, segment) in own_snake.segment_queue.iter().enumerate() {
            if segment.x >= OBS_SIDE || segment.y >= OBS_SIDE {
                continue;
            }
            let bit = if i == 0 { OWN_HEAD_BIT } else { OWN_BODY_BIT };
            obs[segment.y * OBS_SIDE + segment.x] |= bit;
        }

        for (i, segment) in opp_snake.segment_queue.iter().enumerate() {
            if segment.x >= OBS_SIDE || segment.y >= OBS_SIDE {
                continue;
            }
            let bit = if i == 0 { OPP_HEAD_BIT } else { OPP_BODY_BIT };
            obs[segment.y * OBS_SIDE + segment.x] |= bit;
        }

        obs
    }

    pub fn new_from_state(
        (width, height): (usize, usize),
        queued_apples: Vec<(usize, (usize, usize))>,
        apples: Vec<(usize, usize)>,
        walls: Vec<(usize, usize)>,
        portals: Vec<((usize, usize), (usize, usize))>,
        traps: Vec<(i16, (usize, usize))>,
        a_snake: Vec<(usize, usize)>,
        a_queued_length: usize,
        a_max_length_reached: usize,
        a_apples_eaten: usize,
        a_direction: Option<ByteFightAction>,
        b_snake: Vec<(usize, usize)>,
        b_queued_length: usize,
        b_max_length_reached: usize,
        b_apples_eaten: usize,
        b_direction: Option<ByteFightAction>,
        turn_num: usize,
        min_player_size: usize,
        is_player_a: bool,
        decay_countdown_value: usize,
        cached_decay_interval_value: isize,
        is_decaying_value: bool,
    ) -> Self {
        let snake_a = Snake {
            sacrifice: 1,
            max_length_reached: a_max_length_reached,
            queued_length: a_queued_length,
            traps_this_turn: 0,
            current_direction: Some(a_direction.unwrap_or(ByteFightAction::North)),
            segment_queue: a_snake.into_iter().map(Point::from).collect(),
            total_apples: a_apples_eaten,
        };
        let snake_b = Snake {
            sacrifice: 1,
            max_length_reached: b_max_length_reached,
            queued_length: b_queued_length,
            traps_this_turn: 0,
            current_direction: Some(b_direction.unwrap_or(ByteFightAction::North)),
            segment_queue: b_snake.into_iter().map(Point::from).collect(),
            total_apples: b_apples_eaten,
        };

        let mut board = Board {
            map: Map::new((width, height), turn_num),
            apple_timeline_ptr: 0,
            apple_timeline: queued_apples
                .into_iter()
                .map(|(turn, point)| {
                    (
                        turn,
                        Point {
                            x: point.0,
                            y: point.1,
                        },
                    )
                })
                .collect(),
            is_player_a,
            min_player_size,
            snake_a,
            snake_b,
            decay_countdown: decay_countdown_value,
            cached_decay_interval: cached_decay_interval_value
                .try_into()
                .unwrap_or(DECAY_NOT_APPLIED_PLACEHOLDER),
            is_decaying: is_decaying_value,
        };

        for (lifetime, loc) in traps {
            let value = lifetime + (lifetime.signum() * turn_num as i16);
            board.map.update_trap(loc.into(), value);
        }
        for (x, y) in walls {
            board.map.become_wall(Point { x, y });
        }
        for ((x1, y1), (x2, y2)) in portals {
            board
                .map
                .add_portal(Point { x: x1, y: y1 }, Point { x: x2, y: y2 });
        }
        for (x, y) in apples {
            let point = Point { x, y };
            if board.map.is_wall(point) {
                continue;
            }
            board.map.become_apple(point);
            if let Some(portal) = board.map.portal(point) {
                board.map.become_apple(portal);
            }
        }
        board.fix_apple_head_collisions();
        add_padding_walls(&mut board.map);

        let _ = board.apply_decay();
        board
    }

    pub fn terminal_state(&self) -> Option<TerminalState> {
        if self.get_valid_moves().amount() == 0 {
            if self.is_player_a {
                Some(TerminalState::PlayerBWin)
            } else {
                Some(TerminalState::PlayerAWin)
            }
        } else if self.map.turn_count() > LAST_TURN {
            match self
                .snake_a
                .total_apples
                .cmp(&self.snake_b.total_apples)
                .then(self.snake_a.length().cmp(&self.snake_b.length()))
            {
                std::cmp::Ordering::Less => Some(TerminalState::PlayerBWin),
                std::cmp::Ordering::Equal => Some(TerminalState::Draw),
                std::cmp::Ordering::Greater => Some(TerminalState::PlayerAWin),
            }
        } else {
            None
        }
    }

    pub fn get_valid_moves(&self) -> ValidMoves {
        let mut valid_moves = ValidMoves::default();
        let active_snake = if self.is_player_a {
            &self.snake_a
        } else {
            &self.snake_b
        };

        if active_snake.length() < self.min_player_size {
            return valid_moves;
        }
        if self.map.turn_count() > LAST_TURN {
            return valid_moves;
        }

        if active_snake.can_afford_movement(self.min_player_size) {
            let head = active_snake
                .segment_queue
                .front()
                .expect("snake head missing");

            let offsets = if active_snake.current_direction.is_some() {
                6..11
            } else {
                0..9
            };

            for offset in offsets {
                let direction_int = (offset
                    + active_snake
                        .current_direction
                        .unwrap_or(ByteFightAction::North) as u8)
                    % 8;

                let Some(new_loc) = head.try_add_int(direction_int) else {
                    continue;
                };

                let not_wall = !self.map.is_wall(new_loc);
                let not_snake = active_snake.removed_on_point_sacrifice(&new_loc)
                    || (!self.snake_a.segment_queue.contains(&new_loc)
                        && !self.snake_b.segment_queue.contains(&new_loc));
                let portal_is_valid = self.map.portal(new_loc).is_none_or(|p_loc| {
                    let not_wall = !self.map.is_wall(p_loc);
                    let not_snake = active_snake.removed_on_point_sacrifice(&p_loc)
                        || (!self.snake_a.segment_queue.contains(&p_loc)
                            && !self.snake_b.segment_queue.contains(&p_loc));
                    not_wall && not_snake
                });
                let not_trap = self.map.trap(new_loc).abs() <= (self.map.turn_count() as i16)
                    || (self.is_player_a == (self.map.trap(new_loc) > 0));

                let apple_reward = if self.map.is_apple(new_loc) {
                    APPLE_REWARD
                } else {
                    0
                };

                let can_facetank_trap = TRAP_SACRIFICE + active_snake.sacrifice - 1
                    <= active_snake.length() + apple_reward - self.min_player_size;

                if not_wall && not_snake && (not_trap || can_facetank_trap) && portal_is_valid {
                    valid_moves
                        .add(ByteFightAction::new(direction_int).expect("direction_int is valid"));
                }
            }
        }

        if active_snake.can_place_trap(self.min_player_size) {
            valid_moves.add(ByteFightAction::Trap);
        }

        if active_snake.sacrifice > 1 {
            valid_moves.add(ByteFightAction::EndTurn);
        }

        valid_moves
    }

    pub fn heuristics(&self) -> [u8; 18] {
        let (wall_bitmask, apple_bitmask, snake_a_traps, snake_b_traps) = self.map.bitmasks();

        let mut snake_a_obstacle_mask = wall_bitmask.clone();
        for i in 0..32 {
            snake_a_obstacle_mask[i] |= snake_b_traps[i];
        }
        for snake_b_segment in self.snake_b.segment_queue.iter() {
            snake_a_obstacle_mask[snake_b_segment.y] |= 1 << snake_b_segment.x;
        }
        let mut snake_a_seed_arr: [u32; 32] = [0; 32];
        Board::seed_directed_array(
            &mut snake_a_seed_arr,
            self.snake_a.segment_queue.front().expect("snake_a head"),
            self.snake_a.current_direction,
        );
        let (snake_a_apple_dist, snake_a_reach) = self.run_apple_count_flood_fill(
            &mut snake_a_seed_arr,
            &snake_a_obstacle_mask,
            *apple_bitmask,
        );

        let mut snake_b_obstacle_mask = wall_bitmask.clone();
        for i in 0..32 {
            snake_b_obstacle_mask[i] |= snake_a_traps[i];
        }
        for snake_a_segment in self.snake_a.segment_queue.iter() {
            snake_b_obstacle_mask[snake_a_segment.y] |= 1 << snake_a_segment.x;
        }
        let mut snake_b_seed_arr: [u32; 32] = [0; 32];
        Board::seed_directed_array(
            &mut snake_b_seed_arr,
            self.snake_b.segment_queue.front().expect("snake_b head"),
            self.snake_b.current_direction,
        );
        let (snake_b_apple_dist, snake_b_reach) = self.run_apple_count_flood_fill(
            &mut snake_b_seed_arr,
            &snake_b_obstacle_mask,
            *apple_bitmask,
        );

        let board_size = self.map.size() as f32;
        let snake_a_head = self.snake_a.segment_queue.front().expect("snake_a head");
        let snake_b_head = self.snake_b.segment_queue.front().expect("snake_b head");
        let distance_between_snakes = usize::max(
            snake_a_head.x.abs_diff(snake_b_head.x),
            snake_a_head.y.abs_diff(snake_b_head.y),
        );

        let apples_eaten_diff =
            ((self.snake_a.total_apples as f32) - (self.snake_b.total_apples as f32)) / 10.0;

        let turn_ratio = (self.map.turn_count() as f32 / 2000.0).clamp(0.0, 1.0);
        let snake_a_reach_ratio = (snake_a_reach as f32 / board_size).clamp(0.0, 1.0);
        let snake_b_reach_ratio = (snake_b_reach as f32 / board_size).clamp(0.0, 1.0);

        if self.is_player_a {
            [
                Self::encode_signed_feature(turn_ratio),
                self.linorm(distance_between_snakes as f32, 32.0),
                self.linorm(self.snake_a.length() as f32, 32.0),
                self.linorm(self.snake_b.length() as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[0] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[1] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[2] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[3] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[0] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[1] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[2] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[3] as f32, 32.0),
                self.linorm(self.snake_a.sacrifice as f32, 32.0),
                self.linorm(self.snake_a.traps_this_turn as f32, 16.0),
                self.linorm(self.snake_a.max_length_reached as f32, 64.0),
                self.linorm(self.snake_b.max_length_reached as f32, 64.0),
                Self::encode_signed_feature(snake_a_reach_ratio),
                Self::encode_signed_feature(snake_b_reach_ratio),
                self.linorm(apples_eaten_diff, 10.0),
            ][..18]
                .try_into()
                .expect("heuristics size")
        } else {
            [
                Self::encode_signed_feature(turn_ratio),
                self.linorm(distance_between_snakes as f32, 32.0),
                self.linorm(self.snake_b.length() as f32, 32.0),
                self.linorm(self.snake_a.length() as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[0] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[1] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[2] as f32, 32.0),
                self.dropnorm(snake_b_apple_dist[3] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[0] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[1] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[2] as f32, 32.0),
                self.dropnorm(snake_a_apple_dist[3] as f32, 32.0),
                self.linorm(self.snake_b.sacrifice as f32, 32.0),
                self.linorm(self.snake_b.traps_this_turn as f32, 16.0),
                self.linorm(self.snake_b.max_length_reached as f32, 64.0),
                self.linorm(self.snake_a.max_length_reached as f32, 64.0),
                Self::encode_signed_feature(snake_b_reach_ratio),
                Self::encode_signed_feature(snake_a_reach_ratio),
                self.linorm(-apples_eaten_diff, 10.0),
            ][..18]
                .try_into()
                .expect("heuristics size")
        }
    }

    pub fn apply_move(&mut self, action: ByteFightAction) -> Result<RollbackState, ()> {
        match action {
            ByteFightAction::Trap => self.apply_trap(),
            ByteFightAction::EndTurn => self.apply_end_turn(),
            ByteFightAction::FF => Err(()),
            direction => self.apply_movement(direction),
        }
    }

    pub fn rollback(&mut self, state: RollbackState) {
        match state {
            RollbackState::EndTurn {
                old_num_traps,
                old_sacrifice_val,
                prev_cached_decay_interval,
                prev_decay_countdown,
                decayed_point,
                snake_a_ate_during_collision,
                snake_b_ate_during_collision,
                snake_a_max_len,
                snake_b_max_len,
                prev_apple_timeline_ptr,
                apples_placed_index,
            } => {
                self.decay_countdown = prev_decay_countdown;
                let snake = if self.is_player_a {
                    &mut self.snake_a
                } else {
                    &mut self.snake_b
                };
                match decayed_point.as_deref() {
                    None => {}
                    Some([]) => {
                        snake.queued_length += 1;
                        self.is_decaying = !self.is_decaying;
                    }
                    Some([point]) => {
                        snake.segment_queue.push_back(*point);
                        self.is_decaying = !self.is_decaying;
                    }
                    _ => unreachable!(),
                }

                self.cached_decay_interval = prev_cached_decay_interval;
                if snake_a_ate_during_collision {
                    self.snake_a.queued_length -= APPLE_REWARD;
                    self.snake_a.total_apples -= 1;
                    self.snake_a.max_length_reached = snake_a_max_len;
                    let head = self.snake_a.segment_queue.front().expect("snake_a head");
                    self.map.become_apple(*head);
                    if let Some(portal) = self.map.portal(*head) {
                        self.map.become_apple(portal);
                    }
                }
                if snake_b_ate_during_collision {
                    self.snake_b.queued_length -= APPLE_REWARD;
                    self.snake_b.total_apples -= 1;
                    self.snake_b.max_length_reached = snake_b_max_len;
                    let head = self.snake_b.segment_queue.front().expect("snake_b head");
                    self.map.become_apple(*head);
                    if let Some(portal) = self.map.portal(*head) {
                        self.map.become_apple(portal);
                    }
                }

                self.apple_timeline_ptr = prev_apple_timeline_ptr;
                for idx in apples_placed_index {
                    self.map.become_empty(self.apple_timeline[idx].1);
                    if let Some(portal) = self.map.portal(self.apple_timeline[idx].1) {
                        self.map.become_empty(portal);
                    }
                }

                self.is_player_a = !self.is_player_a;
                let snake = if self.is_player_a {
                    &mut self.snake_a
                } else {
                    &mut self.snake_b
                };
                snake.traps_this_turn = old_num_traps;
                snake.sacrifice = old_sacrifice_val;

                self.map.move_backward_turn();
            }
            RollbackState::ApplyMove {
                prev_trap_val,
                sacrificed_points,
                prev_queued_length,
                prev_max_length_reached,
                prev_direction,
                head_was_apple,
            } => {
                let snake = if self.is_player_a {
                    &mut self.snake_a
                } else {
                    &mut self.snake_b
                };
                let head = snake.segment_queue.front().cloned().expect("snake head");
                self.map.update_trap(head, prev_trap_val);
                if let Some(portal) = self.map.portal(head) {
                    self.map.update_trap(portal, prev_trap_val);
                }

                if head_was_apple {
                    self.map.become_apple(head);
                    if let Some(portal) = self.map.portal(head) {
                        self.map.become_apple(portal);
                    }
                    snake.total_apples -= 1;
                }

                let _ = snake.segment_queue.pop_front();
                for point in sacrificed_points.into_iter().rev() {
                    snake.segment_queue.push_back(point);
                }

                snake.current_direction = prev_direction.and_then(ByteFightAction::new);
                snake.queued_length = prev_queued_length;
                snake.max_length_reached = prev_max_length_reached;
                snake.sacrifice -= 1;
            }
            RollbackState::ApplyTrap { old_trap_val, trap } => {
                self.map.update_trap(trap, old_trap_val);
                if let Some(portal) = self.map.portal(trap) {
                    self.map.update_trap(portal, old_trap_val);
                }
                let snake = if self.is_player_a {
                    &mut self.snake_a
                } else {
                    &mut self.snake_b
                };

                snake.traps_this_turn -= 1;
                snake.segment_queue.push_back(trap);
            }
        }
    }

    fn spawn_apples(&mut self) -> (Vec<usize>, bool, bool) {
        let mut apples_placed = Vec::new();
        while self.apple_timeline_ptr < self.apple_timeline.len()
            && self.apple_timeline[self.apple_timeline_ptr].0 <= self.map.turn_count()
        {
            let spawn_point = self.apple_timeline[self.apple_timeline_ptr].1;
            if self.map.is_empty(spawn_point) {
                self.map.become_apple(spawn_point);
                apples_placed.push(self.apple_timeline_ptr);
                if let Some(portal_location) = self.map.portal(spawn_point) {
                    self.map.become_apple(portal_location);
                }
            }

            self.apple_timeline_ptr += 1;
        }

        let (place_a, place_b) = self.fix_apple_head_collisions();
        (apples_placed, place_a, place_b)
    }

    fn update_decay_interval(&mut self) {
        if self.decay_countdown != 0 {
            return;
        }

        for (turn, interval) in &DECAY_TIMELINE {
            if self.map.turn_count() < *turn {
                break;
            }

            self.cached_decay_interval = *interval;
        }
    }

    fn apply_decay(&mut self) -> Result<Option<Vec<Point>>, ()> {
        if self.cached_decay_interval == DECAY_NOT_APPLIED_PLACEHOLDER {
            return Ok(None);
        }

        let decayed = if self.is_decaying || self.decay_countdown == 0 {
            let decayed = if self.is_player_a {
                self.snake_a.apply_sacrifice(1)?
            } else {
                self.snake_b.apply_sacrifice(1)?
            };
            self.is_decaying = !self.is_decaying;

            Some(decayed)
        } else {
            None
        };
        self.decay_countdown = (self.decay_countdown + 1) % self.cached_decay_interval;

        Ok(decayed)
    }

    fn apply_movement(&mut self, action: ByteFightAction) -> Result<RollbackState, ()> {
        let current_snake = if self.is_player_a {
            &mut self.snake_a
        } else {
            &mut self.snake_b
        };

        let prev_queued_length = current_snake.queued_length;
        let prev_max_length_reached = current_snake.max_length_reached;
        let prev_direction = current_snake.current_direction.map(|a| a as u8);

        if !current_snake.can_afford_movement(self.min_player_size) {
            return Err(());
        }
        let Ok((new_head, mut cells_lost)) = current_snake.push_move(action) else {
            return Err(());
        };

        if self.map.is_wall(new_head) {
            return Err(());
        }

        let portal = self.map.portal(new_head);
        if let Some(portal) = portal {
            if self.map.is_wall(portal) {
                return Err(());
            }
            current_snake.segment_queue.push_front(portal);
        } else {
            current_snake.segment_queue.push_front(new_head);
        }

        let mut head_was_apple = false;
        if self.map.is_apple(new_head) {
            head_was_apple = true;
            current_snake.eat_apple();
            self.map.become_empty(new_head);
            if let Some(portal) = portal {
                self.map.become_empty(portal);
            }
        }

        let old_trap_val = self.map.trap(new_head);
        let trap_val = self.map.trap(new_head);
        if trap_val.abs() > self.map.turn_count() as i16 {
            let is_player_a_trap = trap_val > 0;
            let is_enemy_trap = is_player_a_trap ^ self.is_player_a;
            if is_enemy_trap {
                match current_snake.apply_sacrifice(3) {
                    Ok(mut sacrifice) => {
                        cells_lost.append(&mut sacrifice);
                    }
                    Err(_) => {
                        return Err(());
                    }
                }
                self.map.update_trap(new_head, 0);
                if let Some(portal) = portal {
                    self.map.update_trap(portal, 0);
                }
            } else {
                let trap_val = self.map.turn_count() as i16 + TRAP_LIFETIME;

                self.map
                    .update_trap(new_head, trap_val * if self.is_player_a { 1 } else { -1 });
                if let Some(portal) = portal {
                    self.map
                        .update_trap(portal, trap_val * if self.is_player_a { 1 } else { -1 });
                }
            }
        }

        Ok(RollbackState::ApplyMove {
            prev_trap_val: old_trap_val,
            sacrificed_points: cells_lost,
            prev_queued_length,
            prev_max_length_reached,
            prev_direction,
            head_was_apple,
        })
    }

    fn apply_trap(&mut self) -> Result<RollbackState, ()> {
        let snake = if self.is_player_a {
            &mut self.snake_a
        } else {
            &mut self.snake_b
        };

        if !snake.can_place_trap(self.min_player_size) {
            return Err(());
        }

        snake.traps_this_turn += 1;
        let trap = snake.segment_queue.pop_back().ok_or(())?;

        let old_trap_val = self.map.trap(trap);
        let trap_val = self.map.turn_count() as i16 + TRAP_LIFETIME;
        self.map
            .update_trap(trap, trap_val * if self.is_player_a { 1 } else { -1 });
        if let Some(portal) = self.map.portal(trap) {
            self.map
                .update_trap(portal, trap_val * if self.is_player_a { 1 } else { -1 });
        }

        Ok(RollbackState::ApplyTrap { old_trap_val, trap })
    }

    fn apply_end_turn(&mut self) -> Result<RollbackState, ()> {
        let current_snake = if self.is_player_a {
            &mut self.snake_a
        } else {
            &mut self.snake_b
        };
        if current_snake.sacrifice == 1 {
            return Err(());
        }
        self.map.move_forward_turn();

        let traps_this_turn = current_snake.traps_this_turn;
        let sacrifice = current_snake.sacrifice;
        let prev_cached_decay_interval = self.cached_decay_interval;
        let prev_decay_countdown = self.decay_countdown;
        let prev_apple_timeline_ptr = self.apple_timeline_ptr;

        current_snake.traps_this_turn = 0;
        current_snake.sacrifice = 1;

        let snake_a_max_len = self.snake_a.max_length_reached;
        let snake_b_max_len = self.snake_b.max_length_reached;

        self.is_player_a = !self.is_player_a;
        let (apples_placed_index, snake_a_ate_during_collision, snake_b_ate_during_collision) =
            self.spawn_apples();

        self.update_decay_interval();
        let decayed = self.apply_decay().expect(
            "Snake length became 0. This means we made multiple incorrect moves and something is irrecoverably wrong.",
        );

        Ok(RollbackState::EndTurn {
            old_num_traps: traps_this_turn,
            old_sacrifice_val: sacrifice,
            prev_cached_decay_interval,
            decayed_point: decayed,
            prev_decay_countdown,
            snake_a_ate_during_collision,
            snake_b_ate_during_collision,
            snake_a_max_len,
            snake_b_max_len,
            prev_apple_timeline_ptr,
            apples_placed_index,
        })
    }

    fn fix_apple_head_collisions(&mut self) -> (bool, bool) {
        let mut snake_a_ate_during_collision = false;
        let mut snake_b_ate_during_collision = false;
        if let Some(&snake_a_location) = self.snake_a.segment_queue.front() {
            if self.map.is_apple(snake_a_location) {
                snake_a_ate_during_collision = true;
                self.snake_a.queued_length += APPLE_REWARD;
                self.snake_a.total_apples += 1;
                self.snake_a.max_length_reached =
                    self.snake_a.max_length_reached.max(self.snake_a.length());

                self.map.become_empty(snake_a_location);
                if let Some(alt_location) = self.map.portal(snake_a_location) {
                    self.map.become_empty(alt_location);
                }
            }
        }

        if let Some(&snake_b_location) = self.snake_b.segment_queue.front() {
            if self.map.is_apple(snake_b_location) {
                snake_b_ate_during_collision = true;
                self.snake_b.queued_length += APPLE_REWARD;
                self.snake_b.total_apples += 1;
                self.snake_b.max_length_reached =
                    self.snake_b.max_length_reached.max(self.snake_b.length());

                self.map.become_empty(snake_b_location);
                if let Some(alt_location) = self.map.portal(snake_b_location) {
                    self.map.become_empty(alt_location);
                }
            }
        }

        (snake_a_ate_during_collision, snake_b_ate_during_collision)
    }

    fn seed_directed_array(
        seed_arr: &mut [u32; 32],
        origin: &Point,
        facing_dir: Option<ByteFightAction>,
    ) {
        if let Some(dir) = facing_dir {
            for offset in 6..11 {
                if let Some(new_dir) = origin.try_add_int(((dir as u8) + offset) % 8) {
                    seed_arr[new_dir.y] |= 1 << new_dir.x;
                }
            }
        } else {
            seed_arr[origin.y] |= 0b11 << origin.x;
            seed_arr[origin.y] |= 0xC0000000 >> (31 - origin.x);

            if origin.y < 31 {
                seed_arr[origin.y + 1] |= 0b11 << origin.x;
                seed_arr[origin.y + 1] |= 0xC0000000 >> (31 - origin.x);
            }
            if origin.y > 0 {
                seed_arr[origin.y - 1] |= 0b11 << origin.x;
                seed_arr[origin.y - 1] |= 0xC0000000 >> (31 - origin.x);
            }
            seed_arr[origin.y] &= !(1 << origin.x);
        }
    }

    fn run_apple_count_flood_fill(
        &self,
        seed_arr: &mut [u32; 32],
        obstacles: &[u32; 32],
        mut apples: [u32; 32],
    ) -> ([u32; 4], u32) {
        let mut apples_found: u32 = 0;
        let mut apple_loc: [u32; 4] = [512; 4];
        let mut apple_pntr = self.apple_timeline_ptr;

        for i in 0..32 {
            apples_found += (seed_arr[i] & apples[i]).count_ones();
        }

        for i in 0..std::cmp::min(apples_found, 4) {
            apple_loc[i as usize] = 1;
        }

        for epoch in 0..32 {
            for i in 0..32 {
                seed_arr[i] |= seed_arr[i] << 1 | seed_arr[i] >> 1;
            }
            for i in 1..32 {
                seed_arr[i - 1] |= seed_arr[i]
            }
            for i in 0..31 {
                seed_arr[31 - i] |= seed_arr[30 - i]
            }

            for i in 0..32 {
                seed_arr[i] &= !obstacles[i];
            }

            for (p1, p2) in self.map.portals() {
                let p1_reachable = (seed_arr[p1.y] >> p1.x) & 1;
                let p2_reachable = (seed_arr[p2.y] >> p2.x) & 1;
                seed_arr[p1.y] |= p2_reachable << p1.x;
                seed_arr[p2.y] |= p1_reachable << p2.x;
            }

            while apple_pntr < self.apple_timeline.len()
                && self.apple_timeline[apple_pntr].0 <= self.map.turn_count() + 2 * epoch
            {
                apples[self.apple_timeline[apple_pntr].1.y] |=
                    1 << self.apple_timeline[apple_pntr].1.x;
                apple_pntr += 1;
            }

            let mut apples_this_turn = 0;
            for i in 0..32 {
                apples_this_turn += (seed_arr[i] & apples[i]).count_ones();
            }
            for i in apples_found..std::cmp::min(apples_this_turn, 4) {
                apple_loc[i as usize] = (epoch + 1) as u32;
                apples_found += 1;
            }
        }

        let mut reached_tiles: u32 = 0;
        for i in 0..32 {
            reached_tiles += seed_arr[i].count_ones();
        }

        (apple_loc, reached_tiles)
    }

    fn linorm(&self, x: f32, softmax: f32) -> u8 {
        let abs_x = x.abs();
        let normalized = if abs_x <= softmax {
            (0.8 / softmax) * x
        } else {
            x.signum() * (1.0 - (0.2 * softmax) / abs_x)
        };
        Self::encode_signed_feature(normalized)
    }

    fn dropnorm(&self, x: f32, softmax: f32) -> u8 {
        let normalized = if x <= softmax {
            1.0 - (0.75 / softmax) * x
        } else {
            0.0
        };
        Self::encode_signed_feature(normalized)
    }

    #[inline]
    fn encode_signed_feature(value: f32) -> u8 {
        let clamped = value.clamp(-1.0, 1.0);
        let quantized = (clamped * 127.0).round() as i16 + 128;
        quantized.clamp(0, 255) as u8
    }
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid {}", label))
}

fn parse_pair(value: &str, delimiter: char) -> Result<(usize, usize), String> {
    let mut iter = value.split(delimiter);
    let first = iter.next().ok_or_else(|| "missing first".to_string())?;
    let second = iter.next().ok_or_else(|| "missing second".to_string())?;
    if iter.next().is_some() {
        return Err("too many parts".to_string());
    }
    Ok((
        parse_usize(first, "pair_x")?,
        parse_usize(second, "pair_y")?,
    ))
}

fn parse_point(value: &str) -> Result<Point, String> {
    let (x, y) = parse_pair(value, ',')?;
    Ok(Point { x, y })
}

fn parse_portals_section(value: &str) -> Result<Vec<(Point, Point)>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    let mut portals = Vec::new();
    for portal in value.split('_') {
        if portal.is_empty() {
            continue;
        }
        let parts: Vec<&str> = portal.split(',').collect();
        if parts.len() != 4 {
            return Err("invalid portal entry".to_string());
        }
        let p1 = Point {
            x: parse_usize(parts[0], "portal_x1")?,
            y: parse_usize(parts[1], "portal_y1")?,
        };
        let p2 = Point {
            x: parse_usize(parts[2], "portal_x2")?,
            y: parse_usize(parts[3], "portal_y2")?,
        };
        portals.push((p1, p2));
    }
    Ok(portals)
}

fn parse_apple_timeline(value: &str) -> Result<Vec<(usize, Point)>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    let mut timeline = Vec::new();
    for entry in value.split('_') {
        if entry.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.split(',').collect();
        if parts.len() != 3 {
            return Err("invalid apple entry".to_string());
        }
        let turn = parse_usize(parts[0], "apple_turn")?;
        let point = Point {
            x: parse_usize(parts[1], "apple_x")?,
            y: parse_usize(parts[2], "apple_y")?,
        };
        timeline.push((turn, point));
    }
    Ok(timeline)
}

fn parse_walls_bits(bits: &str, width: usize, height: usize) -> Result<Vec<Point>, String> {
    if bits.len() != width * height {
        return Err("wall bit length mismatch".to_string());
    }
    let mut walls = Vec::new();
    for (i, ch) in bits.chars().enumerate() {
        if ch == '1' {
            let x = i % width;
            let y = i / width;
            walls.push(Point { x, y });
        }
    }
    Ok(walls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_and_rollback_move() {
        let mut board = Board::new_from_state(
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
            None,
            vec![(4, 4)],
            1,
            2,
            0,
            None,
            0,
            1,
            true,
            0,
            DECAY_NOT_APPLIED_PLACEHOLDER as isize,
            false,
        );
        let snapshot = board.clone();
        let rollback = board.apply_move(ByteFightAction::East).expect("valid move");
        board.rollback(rollback);
        assert_eq!(board, snapshot);
    }
}
