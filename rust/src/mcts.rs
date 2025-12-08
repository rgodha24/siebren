use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::{Action, Environment, Player, TerminalState};

pub trait Evaluator<E: Environment> {
    /// Returns (policy, value) where policy is over all actions and value is in [-1, 1]
    /// from current player's perspective.
    fn evaluate(&self, env: &E) -> (Vec<f32>, f32);
}

#[derive(Clone)]
pub struct MCTSConfig {
    pub num_simulations: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

struct Node<A> {
    player: Player,
    visit_count: u32,
    value_sum: f32,
    prior: f32,
    children: Vec<(A, Node<A>)>,
}

impl<A> Node<A> {
    fn new(prior: f32, player: Player) -> Self {
        Self {
            player,
            visit_count: 0,
            value_sum: 0.0,
            prior,
            children: Vec::new(),
        }
    }

    #[inline]
    fn q(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }

    #[inline]
    fn is_expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

pub struct MCTS<'a, E: Environment, V: Evaluator<E>> {
    config: &'a MCTSConfig,
    evaluator: &'a V,
    _phantom: std::marker::PhantomData<E>,
}

impl<'a, E: Environment, V: Evaluator<E>> MCTS<'a, E, V> {
    pub fn new(evaluator: &'a V, config: &'a MCTSConfig) -> Self {
        Self {
            config,
            evaluator,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn search(&self, env: &mut E, rng: &mut impl Rng) -> Vec<u32> {
        let mut root = Node::new(0.0, env.current_player());

        self.expand(env, &mut root);
        self.add_dirichlet_noise(&mut root, rng);

        for _ in 0..self.config.num_simulations {
            self.run_simulation(env, &mut root);
        }

        let mut counts = vec![0u32; E::Action::NUM_ACTIONS];
        for (action, child) in &root.children {
            counts[action.to_index()] = child.visit_count;
        }
        counts
    }

    fn run_simulation(&self, env: &mut E, root: &mut Node<E::Action>) {
        let mut rollbacks = Vec::with_capacity(64);
        self.traverse_and_expand(env, root, &mut rollbacks);

        for rb in rollbacks.into_iter().rev() {
            env.rollback(rb);
        }
    }

    /// Q values stored from the perspective of the node's player.
    /// Returns value from perspective of the node's player.
    fn traverse_and_expand(
        &self,
        env: &mut E,
        node: &mut Node<E::Action>,
        rollbacks: &mut Vec<E::RollbackState>,
    ) -> f32 {
        if let Some(term) = env.is_terminal() {
            let v = match term {
                TerminalState::Win(winner) => {
                    if winner == node.player {
                        1.0
                    } else {
                        -1.0
                    }
                }
                TerminalState::Draw => 0.0,
            };
            node.visit_count += 1;
            node.value_sum += v;
            return v;
        }

        if !node.is_expanded() {
            self.expand(env, node);
            let (_, value) = self.evaluator.evaluate(env);
            // value is from current_player's perspective, which equals node.player
            node.visit_count += 1;
            node.value_sum += value;
            return value;
        }

        let action = self.select_action(node);
        rollbacks.push(env.apply_action(action));

        let child = node
            .children
            .iter_mut()
            .find(|(a, _)| *a == action)
            .map(|(_, c)| c)
            .unwrap();

        let child_value = self.traverse_and_expand(env, child, rollbacks);

        // Convert child's value to this node's perspective
        let value = if child.player == node.player {
            child_value
        } else {
            -child_value
        };
        node.visit_count += 1;
        node.value_sum += value;

        value
    }

    fn select_action(&self, node: &Node<E::Action>) -> E::Action {
        let sqrt_n = (node.visit_count as f32).sqrt();

        node.children
            .iter()
            .map(|(action, child)| {
                // Q is from child's perspective; convert to parent's for comparison
                let q = if child.player == node.player {
                    child.q()
                } else {
                    -child.q()
                };
                let ucb = q + self.config.c_puct * child.prior * sqrt_n
                    / (1.0 + child.visit_count as f32);
                (action, ucb)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| *action)
            .expect("select_action called on node with no children")
    }

    fn expand(&self, env: &mut E, node: &mut Node<E::Action>) {
        let (policy, _) = self.evaluator.evaluate(env);
        let valid: Vec<_> = env.valid_actions().collect();

        let mut priors: Vec<(E::Action, f32)> = valid
            .into_iter()
            .map(|a| (a, policy[a.to_index()].max(0.0)))
            .collect();

        let sum: f32 = priors.iter().map(|(_, p)| p).sum();
        if sum > 1e-8 {
            for (_, p) in &mut priors {
                *p /= sum;
            }
        } else {
            let uniform = 1.0 / priors.len() as f32;
            for (_, p) in &mut priors {
                *p = uniform;
            }
        }

        node.children = priors
            .into_iter()
            .map(|(a, prior)| {
                let rollback = env.apply_action(a);
                let child_player = env.current_player();
                env.rollback(rollback);
                (a, Node::new(prior, child_player))
            })
            .collect();
    }

    fn add_dirichlet_noise(&self, root: &mut Node<E::Action>, rng: &mut impl Rng) {
        if root.children.is_empty() {
            return;
        }

        let noise = sample_dirichlet(root.children.len(), self.config.dirichlet_alpha, rng);

        let eps = self.config.dirichlet_epsilon;
        for ((_, child), n) in root.children.iter_mut().zip(noise) {
            child.prior = (1.0 - eps) * child.prior + eps * n;
        }
    }
}

fn sample_dirichlet(n: usize, alpha: f32, rng: &mut impl Rng) -> Vec<f32> {
    let gamma = Gamma::new(alpha, 1.0).expect("invalid gamma params");
    let mut samples: Vec<f32> = (0..n).map(|_| gamma.sample(rng)).collect();
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        let uniform = 1.0 / n as f32;
        for s in &mut samples {
            *s = uniform;
        }
    }
    samples
}

pub fn visits_to_policy(visits: &[u32], temperature: f32) -> Vec<f32> {
    if temperature < 1e-8 {
        let mut policy = vec![0.0; visits.len()];
        if let Some(idx) = visits
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
        {
            policy[idx] = 1.0;
        }
        return policy;
    }

    let inv_t = 1.0 / temperature;
    let powered: Vec<f32> = visits.iter().map(|&v| (v as f32).powf(inv_t)).collect();
    let sum: f32 = powered.iter().sum();

    if sum > 0.0 {
        powered.into_iter().map(|p| p / sum).collect()
    } else {
        vec![0.0; visits.len()]
    }
}

pub fn sample_action_index(policy: &[f32], rng: &mut impl Rng) -> Option<usize> {
    let r: f32 = rng.random();
    let mut cum = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cum += p;
        if r < cum {
            return Some(i);
        }
    }
    policy.iter().rposition(|&p| p > 0.0)
}

pub fn best_action_index(visits: &[u32]) -> Option<usize> {
    visits
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::{TicTacToe, TicTacToeAction};
    use crate::Player;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    struct UniformEvaluator;

    impl Evaluator<TicTacToe> for UniformEvaluator {
        fn evaluate(&self, _env: &TicTacToe) -> (Vec<f32>, f32) {
            (vec![1.0 / 9.0; 9], 0.0)
        }
    }

    struct SmartEvaluator;

    impl Evaluator<TicTacToe> for SmartEvaluator {
        fn evaluate(&self, _env: &TicTacToe) -> (Vec<f32>, f32) {
            let policy = vec![0.12, 0.05, 0.12, 0.05, 0.20, 0.05, 0.12, 0.05, 0.12];
            (policy, 0.0)
        }
    }

    struct TacticalEvaluator;

    impl Evaluator<TicTacToe> for TacticalEvaluator {
        fn evaluate(&self, env: &TicTacToe) -> (Vec<f32>, f32) {
            let current = env.current_player();

            for action in env.valid_actions() {
                let mut test_env = env.clone();
                test_env.apply_action(action);
                if let Some(TerminalState::Win(winner)) = test_env.is_terminal() {
                    if winner == current {
                        let mut policy = vec![0.0; 9];
                        policy[action.to_index()] = 1.0;
                        return (policy, 0.9);
                    }
                }
            }

            let mut blocking_moves = Vec::new();
            for action in env.valid_actions() {
                let mut test_env = env.clone();
                test_env.board[action.to_index()] = match current {
                    Player::PlayerA => 2,
                    Player::PlayerB => 1,
                };
                if test_env.check_winner().is_some() {
                    blocking_moves.push(action.to_index());
                }
            }

            if !blocking_moves.is_empty() {
                let mut policy = vec![0.0; 9];
                let prob = 1.0 / blocking_moves.len() as f32;
                for idx in blocking_moves {
                    policy[idx] = prob;
                }
                return (policy, -0.3);
            }

            let valid: Vec<_> = env.valid_actions().collect();
            let mut policy = vec![0.0; 9];
            let prob = 1.0 / valid.len() as f32;
            for a in valid {
                policy[a.to_index()] = prob;
            }
            (policy, 0.0)
        }
    }

    #[test]
    fn test_visits_to_policy_with_temperature() {
        let visits = vec![100, 50, 25, 25];

        let policy = visits_to_policy(&visits, 1.0);
        assert!((policy[0] - 0.5).abs() < 0.01);
        assert!((policy[1] - 0.25).abs() < 0.01);

        let policy = visits_to_policy(&visits, 0.0);
        assert_eq!(policy[0], 1.0);
        assert_eq!(policy[1], 0.0);
    }

    #[test]
    fn test_visits_to_policy_low_temperature() {
        let visits = vec![100, 90, 5, 5];

        let policy = visits_to_policy(&visits, 0.5);
        assert!(policy[0] > policy[1]);
        assert!(policy[1] > policy[2]);

        let policy_high = visits_to_policy(&visits, 2.0);
        assert!(policy_high[0] < policy[0]);
    }

    #[test]
    fn test_best_action_index() {
        let visits = vec![10, 50, 30, 5];
        assert_eq!(best_action_index(&visits), Some(1));

        let empty: Vec<u32> = vec![];
        assert_eq!(best_action_index(&empty), None);
    }

    #[test]
    fn test_sample_action_index() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let policy = vec![0.0, 0.0, 1.0, 0.0];

        for _ in 0..10 {
            assert_eq!(sample_action_index(&policy, &mut rng), Some(2));
        }
    }

    #[test]
    fn test_mcts_returns_valid_visits() {
        let config = MCTSConfig {
            num_simulations: 100,
            ..Default::default()
        };
        let evaluator = UniformEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let mut game = TicTacToe::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let visits = mcts.search(&mut game, &mut rng);

        assert_eq!(visits.len(), 9);

        let total: u32 = visits.iter().sum();
        assert!(total > 0);

        for action in game.valid_actions() {
            assert!(visits[action.to_index()] > 0);
        }
    }

    #[test]
    fn test_mcts_env_unchanged_after_search() {
        let config = MCTSConfig {
            num_simulations: 50,
            ..Default::default()
        };
        let evaluator = UniformEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let mut game = TicTacToe::new();
        game.apply_action(TicTacToeAction(4));

        let board_before = game.board;
        let player_before = game.current_player();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let _ = mcts.search(&mut game, &mut rng);

        assert_eq!(game.board, board_before);
        assert_eq!(game.current_player(), player_before);
    }

    #[test]
    fn test_mcts_finds_winning_move() {
        let config = MCTSConfig {
            num_simulations: 100,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.0,
        };
        let evaluator = TacticalEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let mut game = TicTacToe::new();
        game.apply_action(TicTacToeAction(0));
        game.apply_action(TicTacToeAction(3));
        game.apply_action(TicTacToeAction(1));
        game.apply_action(TicTacToeAction(4));

        let (policy, value) = evaluator.evaluate(&game);
        assert!(policy[2] > 0.9);
        assert!(value > 0.5);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let visits = mcts.search(&mut game, &mut rng);

        let best = best_action_index(&visits).unwrap();
        assert_eq!(best, 2);
    }

    #[test]
    fn test_mcts_blocks_opponent_win() {
        let config = MCTSConfig {
            num_simulations: 100,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.1,
        };
        let evaluator = TacticalEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let mut game = TicTacToe::new();
        game.apply_action(TicTacToeAction(0));
        game.apply_action(TicTacToeAction(3));
        game.apply_action(TicTacToeAction(1));

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let visits = mcts.search(&mut game, &mut rng);

        let best = best_action_index(&visits).unwrap();
        assert_eq!(best, 2);
    }

    #[test]
    fn test_mcts_with_uniform_explores_all_moves() {
        let config = MCTSConfig {
            num_simulations: 500,
            c_puct: 2.0,
            dirichlet_alpha: 0.5,
            dirichlet_epsilon: 0.25,
        };
        let evaluator = UniformEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let mut game = TicTacToe::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let visits = mcts.search(&mut game, &mut rng);

        for (i, &v) in visits.iter().enumerate() {
            assert!(v > 0, "Move {} should have some visits", i);
        }
    }

    #[test]
    fn test_mcts_with_smart_evaluator() {
        let config = MCTSConfig {
            num_simulations: 100,
            ..Default::default()
        };
        let evaluator = SmartEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let mut game = TicTacToe::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let visits = mcts.search(&mut game, &mut rng);

        assert!(visits[4] > 0);
    }

    #[test]
    fn test_mcts_determinism_with_seed() {
        let config = MCTSConfig {
            num_simulations: 50,
            ..Default::default()
        };
        let evaluator = UniformEvaluator;

        let mut game1 = TicTacToe::new();
        let mut game2 = TicTacToe::new();

        let mcts = MCTS::new(&evaluator, &config);

        let mut rng1 = ChaCha8Rng::seed_from_u64(12345);
        let mut rng2 = ChaCha8Rng::seed_from_u64(12345);

        let visits1 = mcts.search(&mut game1, &mut rng1);
        let visits2 = mcts.search(&mut game2, &mut rng2);

        assert_eq!(visits1, visits2);
    }

    #[test]
    fn test_mcts_config_affects_search() {
        let evaluator = UniformEvaluator;

        let config_few = MCTSConfig {
            num_simulations: 10,
            ..Default::default()
        };
        let config_many = MCTSConfig {
            num_simulations: 100,
            ..Default::default()
        };

        let mut game1 = TicTacToe::new();
        let mut game2 = TicTacToe::new();

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let mcts_few = MCTS::new(&evaluator, &config_few);
        let mcts_many = MCTS::new(&evaluator, &config_many);

        let visits_few = mcts_few.search(&mut game1, &mut rng1);
        let visits_many = mcts_many.search(&mut game2, &mut rng2);

        let total_few: u32 = visits_few.iter().sum();
        let total_many: u32 = visits_many.iter().sum();

        assert!(total_many > total_few);
    }
}
