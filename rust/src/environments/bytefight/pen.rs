use std::collections::VecDeque;
use std::fmt;

use super::game::Board;
use super::map::{add_padding_walls, Map};
use super::snake::Snake;
use super::types::{ByteFightAction, Point};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ByteFightPen(pub String);

#[derive(Debug)]
pub struct PenError(String);

impl PenError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PenError {}

impl ByteFightPen {
    pub fn from_board(board: &Board) -> Self {
        ByteFightPen(board_to_pen(board))
    }

    pub fn into_board(self) -> Result<Board, PenError> {
        board_from_pen(&self.0)
    }
}

impl From<Board> for ByteFightPen {
    fn from(board: Board) -> Self {
        ByteFightPen::from_board(&board)
    }
}

impl From<&Board> for ByteFightPen {
    fn from(board: &Board) -> Self {
        ByteFightPen::from_board(board)
    }
}

fn board_to_pen(board: &Board) -> String {
    let (width, height) = board.map.dimensions();
    let mut sections = Vec::new();
    sections.push(format!("{}x{}", width, height));
    sections.push(format!("t{}", board.map.turn_count()));
    sections.push(format!("p{}", if board.is_player_a { "A" } else { "B" }));
    sections.push(format!("m{}", board.min_player_size));
    sections.push(format!(
        "d{},{},{}",
        board.decay_countdown,
        board.cached_decay_interval,
        if board.is_decaying { 1 } else { 0 }
    ));
    sections.push(format!("a{}", board.apple_timeline_ptr));
    sections.push(format!("w{}", encode_walls(&board.map)));
    sections.push(format!("o{}", encode_portals(&board.map)));
    sections.push(format!("f{}", encode_apples(&board.map)));
    sections.push(format!("q{}", encode_timeline(&board.apple_timeline)));
    sections.push(format!("r{}", encode_traps(&board.map)));
    sections.push(format!("A{}", encode_snake(&board.snake_a)));
    sections.push(format!("B{}", encode_snake(&board.snake_b)));
    sections.join("|")
}

fn board_from_pen(pen: &str) -> Result<Board, PenError> {
    let parts: Vec<&str> = pen.split('|').collect();
    if parts.len() != 13 {
        return Err(PenError::new("invalid PEN section count"));
    }

    let (width, height) = parse_dimensions(parts[0])?;
    let turn_count = parse_prefixed_usize(parts[1], 't')?;
    let is_player_a = parse_prefixed_player(parts[2])?;
    let min_player_size = parse_prefixed_usize(parts[3], 'm')?;
    let (decay_countdown, cached_decay_interval, is_decaying) = parse_decay(parts[4], 'd')?;
    let apple_timeline_ptr = parse_prefixed_usize(parts[5], 'a')?;
    let walls = parse_prefixed_walls(parts[6], width, height)?;
    let portals = parse_prefixed_portals(parts[7])?;
    let apples = parse_prefixed_points(parts[8], 'f')?;
    let apple_timeline = parse_prefixed_timeline(parts[9], 'q')?;
    let traps = parse_prefixed_traps(parts[10], 'r')?;
    let snake_a = parse_prefixed_snake(parts[11], 'A')?;
    let snake_b = parse_prefixed_snake(parts[12], 'B')?;

    let mut map = Map::new((width, height), turn_count);
    for wall in walls {
        map.become_wall(wall);
    }
    for (p1, p2) in portals {
        map.add_portal(p1, p2);
    }
    for apple in apples {
        map.become_apple(apple);
    }
    for (trap, point) in traps {
        map.update_trap(point, trap);
    }
    add_padding_walls(&mut map);

    Ok(Board {
        map,
        apple_timeline,
        apple_timeline_ptr,
        snake_a,
        snake_b,
        is_player_a,
        min_player_size,
        decay_countdown,
        cached_decay_interval,
        is_decaying,
    })
}

fn encode_walls(map: &Map) -> String {
    let (width, height) = map.dimensions();
    let mut rows = Vec::with_capacity(height);
    for y in 0..height {
        let mut row = String::new();
        let mut empty_run = 0usize;
        for x in 0..width {
            if map.is_wall(Point { x, y }) {
                if empty_run > 0 {
                    row.push_str(&empty_run.to_string());
                    empty_run = 0;
                }
                row.push('#');
            } else {
                empty_run += 1;
            }
        }
        if empty_run > 0 {
            row.push_str(&empty_run.to_string());
        }
        rows.push(row);
    }
    rows.join("/")
}

fn encode_portals(map: &Map) -> String {
    if map.portals().is_empty() {
        return "-".to_string();
    }
    map.portals()
        .iter()
        .map(|(p1, p2)| format!("{},{}~{},{}", p1.x, p1.y, p2.x, p2.y))
        .collect::<Vec<_>>()
        .join(";")
}

fn encode_apples(map: &Map) -> String {
    let (width, height) = map.dimensions();
    let mut apples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if map.is_apple(Point { x, y }) {
                apples.push(Point { x, y });
            }
        }
    }
    if apples.is_empty() {
        return "-".to_string();
    }
    apples
        .into_iter()
        .map(|point| format!("{},{}", point.x, point.y))
        .collect::<Vec<_>>()
        .join(";")
}

fn encode_timeline(timeline: &[(usize, Point)]) -> String {
    if timeline.is_empty() {
        return "-".to_string();
    }
    timeline
        .iter()
        .map(|(turn, point)| format!("{},{},{}", turn, point.x, point.y))
        .collect::<Vec<_>>()
        .join(";")
}

fn encode_traps(map: &Map) -> String {
    let (width, height) = map.dimensions();
    let mut traps = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let value = map.trap(Point { x, y });
            if value != 0 && value.abs() > map.turn_count() as i16 {
                traps.push(format!("{},{},{}", value, x, y));
            }
        }
    }
    if traps.is_empty() {
        return "-".to_string();
    }
    traps.join(";")
}

fn encode_snake(snake: &Snake) -> String {
    let direction = snake
        .current_direction
        .map(|dir| (dir as u8).to_string())
        .unwrap_or_else(|| "-".to_string());
    let segments = snake
        .segment_queue
        .iter()
        .map(|point| format!("{},{}", point.x, point.y))
        .collect::<Vec<_>>()
        .join(">");
    format!(
        "{},{},{},{},{},{}:{}",
        direction,
        snake.queued_length,
        snake.max_length_reached,
        snake.total_apples,
        snake.sacrifice,
        snake.traps_this_turn,
        segments
    )
}

fn parse_dimensions(value: &str) -> Result<(usize, usize), PenError> {
    let mut iter = value.split('x');
    let width = iter.next().ok_or_else(|| PenError::new("missing width"))?;
    let height = iter.next().ok_or_else(|| PenError::new("missing height"))?;
    if iter.next().is_some() {
        return Err(PenError::new("invalid dimensions"));
    }
    Ok((parse_usize(width, "width")?, parse_usize(height, "height")?))
}

fn parse_prefixed_usize(section: &str, prefix: char) -> Result<usize, PenError> {
    let value = section
        .strip_prefix(prefix)
        .ok_or_else(|| PenError::new("missing prefix"))?;
    parse_usize(value, "value")
}

fn parse_prefixed_player(section: &str) -> Result<bool, PenError> {
    let value = section
        .strip_prefix('p')
        .ok_or_else(|| PenError::new("missing player prefix"))?;
    match value {
        "A" => Ok(true),
        "B" => Ok(false),
        _ => Err(PenError::new("invalid player")),
    }
}

fn parse_decay(section: &str, prefix: char) -> Result<(usize, usize, bool), PenError> {
    let value = section
        .strip_prefix(prefix)
        .ok_or_else(|| PenError::new("missing decay prefix"))?;
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 3 {
        return Err(PenError::new("invalid decay section"));
    }
    let countdown = parse_usize(parts[0], "decay_countdown")?;
    let interval = parse_usize(parts[1], "decay_interval")?;
    let is_decaying = match parts[2] {
        "1" => true,
        "0" => false,
        _ => return Err(PenError::new("invalid decay flag")),
    };
    Ok((countdown, interval, is_decaying))
}

fn parse_prefixed_walls(
    section: &str,
    width: usize,
    height: usize,
) -> Result<Vec<Point>, PenError> {
    let value = section
        .strip_prefix('w')
        .ok_or_else(|| PenError::new("missing walls prefix"))?;
    let rows: Vec<&str> = value.split('/').collect();
    if rows.len() != height {
        return Err(PenError::new("wall rows mismatch"));
    }

    let mut walls = Vec::new();
    for (y, row) in rows.into_iter().enumerate() {
        let mut x = 0usize;
        let mut digits = String::new();
        for ch in row.chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
                continue;
            }
            if !digits.is_empty() {
                let run = parse_usize(&digits, "wall run")?;
                x += run;
                digits.clear();
            }
            if ch == '#' {
                if x >= width {
                    return Err(PenError::new("wall row overflow"));
                }
                walls.push(Point { x, y });
                x += 1;
            } else {
                return Err(PenError::new("invalid wall token"));
            }
        }
        if !digits.is_empty() {
            let run = parse_usize(&digits, "wall run")?;
            x += run;
        }
        if x != width {
            return Err(PenError::new("wall row width mismatch"));
        }
    }

    Ok(walls)
}

fn parse_prefixed_portals(section: &str) -> Result<Vec<(Point, Point)>, PenError> {
    let value = section
        .strip_prefix('o')
        .ok_or_else(|| PenError::new("missing portals prefix"))?;
    if value == "-" {
        return Ok(Vec::new());
    }
    let mut portals = Vec::new();
    for entry in value.split(';') {
        let (left, right) = entry
            .split_once('~')
            .ok_or_else(|| PenError::new("invalid portal entry"))?;
        portals.push((parse_point(left)?, parse_point(right)?));
    }
    Ok(portals)
}

fn parse_prefixed_points(section: &str, prefix: char) -> Result<Vec<Point>, PenError> {
    let value = section
        .strip_prefix(prefix)
        .ok_or_else(|| PenError::new("missing points prefix"))?;
    if value == "-" {
        return Ok(Vec::new());
    }
    value.split(';').map(parse_point).collect()
}

fn parse_prefixed_timeline(section: &str, prefix: char) -> Result<Vec<(usize, Point)>, PenError> {
    let value = section
        .strip_prefix(prefix)
        .ok_or_else(|| PenError::new("missing timeline prefix"))?;
    if value == "-" {
        return Ok(Vec::new());
    }
    let mut timeline = Vec::new();
    for entry in value.split(';') {
        let parts: Vec<&str> = entry.split(',').collect();
        if parts.len() != 3 {
            return Err(PenError::new("invalid timeline entry"));
        }
        let turn = parse_usize(parts[0], "timeline_turn")?;
        let point = Point {
            x: parse_usize(parts[1], "timeline_x")?,
            y: parse_usize(parts[2], "timeline_y")?,
        };
        timeline.push((turn, point));
    }
    Ok(timeline)
}

fn parse_prefixed_traps(section: &str, prefix: char) -> Result<Vec<(i16, Point)>, PenError> {
    let value = section
        .strip_prefix(prefix)
        .ok_or_else(|| PenError::new("missing traps prefix"))?;
    if value == "-" {
        return Ok(Vec::new());
    }
    let mut traps = Vec::new();
    for entry in value.split(';') {
        let parts: Vec<&str> = entry.split(',').collect();
        if parts.len() != 3 {
            return Err(PenError::new("invalid trap entry"));
        }
        let trap = parse_i16(parts[0], "trap_value")?;
        let point = Point {
            x: parse_usize(parts[1], "trap_x")?,
            y: parse_usize(parts[2], "trap_y")?,
        };
        traps.push((trap, point));
    }
    Ok(traps)
}

fn parse_prefixed_snake(section: &str, prefix: char) -> Result<Snake, PenError> {
    let value = section
        .strip_prefix(prefix)
        .ok_or_else(|| PenError::new("missing snake prefix"))?;
    let (meta, body) = value
        .split_once(':')
        .ok_or_else(|| PenError::new("invalid snake section"))?;
    let parts: Vec<&str> = meta.split(',').collect();
    if parts.len() != 6 {
        return Err(PenError::new("invalid snake metadata"));
    }
    let direction = if parts[0] == "-" {
        None
    } else {
        Some(
            ByteFightAction::new(parse_usize(parts[0], "direction")? as u8)
                .ok_or_else(|| PenError::new("invalid direction value"))?,
        )
    };

    let queued_length = parse_usize(parts[1], "queued_length")?;
    let max_length_reached = parse_usize(parts[2], "max_length")?;
    let total_apples = parse_usize(parts[3], "total_apples")?;
    let sacrifice = parse_usize(parts[4], "sacrifice")?;
    let traps_this_turn = parse_usize(parts[5], "traps_this_turn")?;

    let segment_queue = if body.is_empty() {
        VecDeque::new()
    } else {
        body.split('>')
            .map(parse_point)
            .collect::<Result<VecDeque<_>, _>>()?
    };

    Ok(Snake {
        max_length_reached,
        queued_length,
        traps_this_turn,
        current_direction: direction,
        segment_queue,
        sacrifice,
        total_apples,
    })
}

fn parse_usize(value: &str, label: &str) -> Result<usize, PenError> {
    value
        .parse::<usize>()
        .map_err(|_| PenError::new(format!("invalid {}", label)))
}

fn parse_i16(value: &str, label: &str) -> Result<i16, PenError> {
    value
        .parse::<i16>()
        .map_err(|_| PenError::new(format!("invalid {}", label)))
}

fn parse_pair(value: &str, delimiter: char) -> Result<(usize, usize), PenError> {
    let mut iter = value.split(delimiter);
    let first = iter.next().ok_or_else(|| PenError::new("missing first"))?;
    let second = iter.next().ok_or_else(|| PenError::new("missing second"))?;
    if iter.next().is_some() {
        return Err(PenError::new("too many parts"));
    }
    Ok((
        parse_usize(first, "pair_x")?,
        parse_usize(second, "pair_y")?,
    ))
}

fn parse_point(value: &str) -> Result<Point, PenError> {
    let (x, y) = parse_pair(value, ',')?;
    Ok(Point { x, y })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pen_roundtrip_with_state() {
        let mut board = Board::new_from_state(
            (6, 6),
            vec![(0, (1, 1)), (5, (2, 2))],
            vec![(0, 0), (3, 4)],
            vec![],
            vec![((0, 0), (5, 5))],
            vec![(5, (4, 1)), (-5, (1, 4))],
            vec![(2, 2)],
            1,
            2,
            0,
            Some(ByteFightAction::East),
            vec![(4, 4)],
            1,
            2,
            0,
            Some(ByteFightAction::West),
            3,
            1,
            true,
            2,
            12,
            true,
        );
        board.apple_timeline_ptr = 1;
        let pen = ByteFightPen::from(&board);
        let rebuilt = pen.clone().into_board().expect("valid pen");
        assert_eq!(pen.0, ByteFightPen::from(&rebuilt).0);
    }
}
