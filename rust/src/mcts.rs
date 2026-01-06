//! Async MCTS implementation for GPU-batched inference.
//!
//! This is an async adaptation of the sync MCTS in `mcts.rs`.
//! The key difference is that `expand()` awaits the evaluator.

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::eval::Evaluator;
use crate::{Action, Environment, Player, TerminalState};

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

    /// Run MCTS search and return visit counts for each action.
    pub async fn search(&self, env: &mut E, rng: &mut impl Rng) -> Vec<u32> {
        let mut root = Node::new(0.0, env.current_player());

        self.expand(env, &mut root).await;
        self.add_dirichlet_noise(&mut root, rng);

        for _ in 0..self.config.num_simulations {
            self.run_simulation(env, &mut root).await;
        }

        let mut counts = vec![0u32; E::NUM_ACTIONS];
        for (action, child) in &root.children {
            counts[action.to_index()] = child.visit_count;
        }
        counts
    }

    async fn run_simulation(&self, env: &mut E, root: &mut Node<E::Action>) {
        let mut rollbacks = Vec::with_capacity(64);
        self.traverse_and_expand(env, root, &mut rollbacks).await;

        for rb in rollbacks.into_iter().rev() {
            env.rollback(rb);
        }
    }

    /// Q values stored from the perspective of the node's player.
    /// Returns value from perspective of the node's player.
    async fn traverse_and_expand(
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
            let value = self.expand(env, node).await;
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

        // Box::pin for recursive async call
        let child_value = Box::pin(self.traverse_and_expand(env, child, rollbacks)).await;

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

    async fn expand(&self, env: &mut E, node: &mut Node<E::Action>) -> f32 {
        let (policy, value) = self.evaluator.evaluate(env).await;
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

        value
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
        samples.fill(uniform);
    }
    samples
}

/// Convert visit counts to a policy distribution.
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

/// Sample an action index from a policy distribution.
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

/// Get the action index with most visits.
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
    use crate::environments::TicTacToe;
    use crate::eval::UniformEvaluator;
    use crate::executor::Executor;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_async_mcts_returns_valid_visits() {
        let config = MCTSConfig {
            num_simulations: 100,
            ..Default::default()
        };
        let evaluator = UniformEvaluator;
        let mcts = MCTS::new(&evaluator, &config);

        let game = Rc::new(RefCell::new(TicTacToe::new()));
        let rng = Rc::new(RefCell::new(ChaCha8Rng::seed_from_u64(42)));
        let result: Rc<RefCell<Option<Vec<u32>>>> = Rc::new(RefCell::new(None));

        let game_clone = game.clone();
        let rng_clone = rng.clone();
        let result_clone = result.clone();

        let fut = async move {
            let visits = mcts
                .search(&mut *game_clone.borrow_mut(), &mut *rng_clone.borrow_mut())
                .await;
            *result_clone.borrow_mut() = Some(visits);
        };

        // Use a dummy event for the executor
        let event = event_listener::Event::new();
        let executor = Executor::new(|| event.listen());
        executor.run(vec![Box::pin(fut)], || false);

        let visits = result.borrow().clone().unwrap();
        assert_eq!(visits.len(), 9);

        let total: u32 = visits.iter().sum();
        assert!(total > 0);

        // All valid actions should have some visits
        for action in game.borrow().valid_actions() {
            assert!(visits[action.to_index()] > 0);
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
    fn test_best_action_index() {
        let visits = vec![10, 50, 30, 5];
        assert_eq!(best_action_index(&visits), Some(1));

        let empty: Vec<u32> = vec![];
        assert_eq!(best_action_index(&empty), None);
    }
}
