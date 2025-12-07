use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::{Action, Environment, TerminalState};

/// Trait for neural network evaluation - implement this to plug in your model
pub trait Evaluator<E: Environment> {
    /// Returns (policy, value)
    /// - policy: probability distribution over ALL actions (length = E::NUM_ACTIONS)
    /// - value: position evaluation in [-1, 1] from current player's perspective
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
    visit_count: u32,
    value_sum: f32,
    prior: f32,
    children: Vec<(A, Node<A>)>,
}

impl<A> Node<A> {
    fn new(prior: f32) -> Self {
        Self {
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

    /// Run MCTS search and return visit counts for each action index
    pub fn search(&self, env: &mut E, rng: &mut impl Rng) -> Vec<u32> {
        let mut root = Node::new(0.0);

        // Expand root with network policy
        self.expand(env, &mut root);
        self.add_dirichlet_noise(&mut root, rng);

        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.run_simulation(env, &mut root);
        }

        // Return visit counts indexed by action
        let mut counts = vec![0u32; E::Action::NUM_ACTIONS];
        for (action, child) in &root.children {
            counts[action.to_index()] = child.visit_count;
        }
        counts
    }

    fn run_simulation(&self, env: &mut E, root: &mut Node<E::Action>) {
        let mut rollbacks = Vec::with_capacity(64);
        self.traverse_and_expand(env, root, &mut rollbacks);

        // Rollback all moves
        for rb in rollbacks.into_iter().rev() {
            env.rollback(rb);
        }
    }

    /// Traverse tree, expand leaf, and backpropagate.
    /// Returns value from the perspective of the node we started at.
    fn traverse_and_expand(
        &self,
        env: &mut E,
        node: &mut Node<E::Action>,
        rollbacks: &mut Vec<E::RollbackState>,
    ) -> f32 {
        // Terminal check
        if let Some(term) = env.is_terminal() {
            let v = match term {
                TerminalState::Win(p) if p == env.current_player() => 1.0,
                TerminalState::Win(_) => -1.0,
                TerminalState::Draw => 0.0,
            };
            node.visit_count += 1;
            node.value_sum += v;
            return v;
        }

        // If not expanded, expand and return network value
        if !node.is_expanded() {
            self.expand(env, node);
            let (_, value) = self.evaluator.evaluate(env);
            node.visit_count += 1;
            node.value_sum += value;
            return value;
        }

        // Select best child via UCB
        let action = self.select_action(node);
        rollbacks.push(env.apply_action(action));

        // Find child and recurse
        let child = node
            .children
            .iter_mut()
            .find(|(a, _)| *a == action)
            .map(|(_, c)| c)
            .unwrap();

        let child_value = self.traverse_and_expand(env, child, rollbacks);

        // Value for this node is negated (opponent's gain is our loss)
        let value = -child_value;
        node.visit_count += 1;
        node.value_sum += value;

        value
    }

    fn select_action(&self, node: &Node<E::Action>) -> E::Action {
        let sqrt_n = (node.visit_count as f32).sqrt();

        node.children
            .iter()
            .map(|(action, child)| {
                // UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                let ucb = child.q()
                    + self.config.c_puct * child.prior * sqrt_n / (1.0 + child.visit_count as f32);
                (action, ucb)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| *action)
            .expect("select_action called on node with no children")
    }

    fn expand(&self, env: &E, node: &mut Node<E::Action>) {
        let (policy, _) = self.evaluator.evaluate(env);
        let valid: Vec<_> = env.valid_actions().collect();

        // Collect priors for valid actions
        let mut priors: Vec<(E::Action, f32)> = valid
            .into_iter()
            .map(|a| (a, policy[a.to_index()].max(0.0)))
            .collect();

        // Normalize
        let sum: f32 = priors.iter().map(|(_, p)| p).sum();
        if sum > 1e-8 {
            for (_, p) in &mut priors {
                *p /= sum;
            }
        } else {
            // Uniform over valid actions if policy is all zeros
            let uniform = 1.0 / priors.len() as f32;
            for (_, p) in &mut priors {
                *p = uniform;
            }
        }

        // Create children
        node.children = priors
            .into_iter()
            .map(|(a, prior)| (a, Node::new(prior)))
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

/// Sample from a symmetric Dirichlet distribution with concentration parameter `alpha`.
/// Uses the Gamma distribution method: sample Gamma(alpha, 1) for each dimension, then normalize.
fn sample_dirichlet(n: usize, alpha: f32, rng: &mut impl Rng) -> Vec<f32> {
    let gamma = Gamma::new(alpha, 1.0).expect("invalid gamma params");
    let mut samples: Vec<f32> = (0..n).map(|_| gamma.sample(rng)).collect();
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // Fallback to uniform if all samples are zero (shouldn't happen with valid alpha)
        let uniform = 1.0 / n as f32;
        for s in &mut samples {
            *s = uniform;
        }
    }
    samples
}

// ============================================================================
// Helper functions for training
// ============================================================================

/// Convert visit counts to policy with temperature
pub fn visits_to_policy(visits: &[u32], temperature: f32) -> Vec<f32> {
    if temperature < 1e-8 {
        // Temperature ~0: deterministic, put all mass on best action
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

/// Sample action index from policy
pub fn sample_action_index(policy: &[f32], rng: &mut impl Rng) -> Option<usize> {
    let r: f32 = rng.random();
    let mut cum = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cum += p;
        if r < cum {
            return Some(i);
        }
    }
    // Fallback: return last non-zero action
    policy.iter().rposition(|&p| p > 0.0)
}

/// Get best action index (most visits)
pub fn best_action_index(visits: &[u32]) -> Option<usize> {
    visits
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
}
