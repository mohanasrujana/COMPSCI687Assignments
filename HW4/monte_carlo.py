import numpy as np
import math
from typing import Dict, List, Tuple, Optional

State = Tuple[int, int]
Action = str

def rotate_right(delta: Tuple[int, int]) -> Tuple[int, int]:
    dr, dc = delta
    return (dc, -dr)

def rotate_left(delta: Tuple[int, int]) -> Tuple[int, int]:
    dr, dc = delta
    return (-dc, dr)

class Environment:
    def __init__(self, gamma: float = 0.925, reward: float = 0.0, terminal: bool = False):
        self.gamma = gamma
        self.reward = reward
        self.terminal = terminal
        self.rows = 5
        self.cols = 5
        self.furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
        self.monsters = [(0, 3), (4, 1)]
        self.food = (4, 4)
        self.catnip = None
        self.actions = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1),
            'AR': (0, 1)
        }
        self.action_names = list(self.actions.keys())
        self._set_transitions()
        self._set_rewards()

    def is_blocked(self, state: Tuple[int, int]) -> bool:
        return state in self.furniture

    def _set_transitions(self):
        self.transitions = {}
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                self.transitions[state] = {}
                for action_name in self.action_names:
                    self.transitions[state][action_name] = {}
                    new_prob = self._transition_probability(state, action_name)
                    for s_next, probability in new_prob.items():
                        if probability > 0:
                            self.transitions[state][action_name][s_next] = probability

    def _transition_probability(self, state: State, action: Action) -> Dict[State, float]:
        r, c = state
        true_dir = self.actions[action]
        right = rotate_right(true_dir)
        left = rotate_left(true_dir)
        direction_probs = [
            (true_dir, 0.70),
            (right, 0.12),
            (left, 0.12),
            ((0, 0), 0.06),
        ]
        ns = {}
        for rc, probability in direction_probs:
            dr, dc = rc
            dr_final, dc_final = r + dr, c + dc
            
            if (dr_final < 0 or dr_final >= self.rows or dc_final < 0 or dc_final >= self.cols) or ((dr_final, dc_final) in self.furniture):
                position = state
            else:
                position = (dr_final, dc_final)
            ns[position] = ns.get(position, 0.0) + probability
        
        return ns
    
    def _set_rewards(self):
        self.rewards = {}
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                self.rewards[state] = {}
                for action_name in self.action_names:
                    self.rewards[state][action_name] = {}
                    for next_state, p in self.transitions[state][action_name].items():
                        reward = self._reward_function(state, action_name, next_state)
                        self.rewards[state][action_name][next_state] = reward

    def _reward_function(self, s: State, a: Action, ns: State) -> float:
        ''' R(s,a,s') '''
        if ns == self.food:
            return 10.0
        elif self.terminal and ns == self.catnip:
            return self.reward
        if ns in self.monsters:
            return -8.0
        if not self.terminal and ns == self.catnip:
            return self.reward
        
        return -0.05
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        if state == self.food:
            return True
        if self.terminal and state == self.catnip:
            return True
        return False
    
    def all_states(self) -> List[Tuple[int, int]]:
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                states.append((r, c))
        return states


class ValueIteration:
    def __init__(self, env: Environment, theta: float = 0.0001, max_iters: int = 10000):
        self.env = env
        self.theta = theta
        self.max_iters = max_iters
        self.values = {}
        self.policy = {}
        self.iterations = 0
        self._initialize_state_values()

    def _initialize_state_values(self):
        for state in self.env.all_states():
            self.values[state] = 0.0
    
    def run(self):
        while self.iterations < self.max_iters:
            delta = 0.0
            vfs = {}
            for s in self.env.all_states():
                if self.env.is_terminal(s):
                    vfs[s] = 0.0
                    self.policy[s] = "G"  
                    continue
                best_val = -math.inf
                best_action = "AU"
                for a in self.env.action_names:
                    q = 0.0
                    for s_next, p in self.env.transitions[s][a].items():
                        q += p * (self.env.rewards[s][a][s_next] + self.env.gamma * self.values[s_next])
                    if q > best_val:
                        best_val = q
                        best_action = a
                
                vfs[s] = best_val
                self.policy[s] = best_action
                delta = max(delta, abs(vfs[s] - self.values[s]))
            
            self.values = vfs
            self.iterations += 1
            if delta < self.theta:
                break
        return self.values, self.policy, self.iterations

def arrow_formatting(action_code: str) -> str:
    return {
        "AU": "↑",
        "AD": "↓",
        "AL": "←",
        "AR": "→",
        "G": "G",
    }[action_code]

def run_format(env: Environment) -> None:
    vi = ValueIteration(env)
    values, policy, iters = vi.run()
    print(f"\n Gamma: {env.gamma}")
    print(f"No. of Iterations: {iters}")

    print("\nValue Function")
    for r in range(env.rows):
        row_vals = []
        for c in range(env.cols):
            if hasattr(env, 'is_blocked') and env.is_blocked((r, c)):
                row_vals.append("0.0000")
            else:
                row_vals.append(f"{values[(r, c)]:.4f}")
        print("\t".join(row_vals))

    print("\nPolicy")
    for r in range(env.rows):
        row_pol = []
        for c in range(env.cols):
            if hasattr(env, 'is_blocked') and env.is_blocked((r, c)):
                row_pol.append("")
            else:
                row_pol.append(arrow_formatting(policy[(r, c)]))
        print("\t".join(row_pol))
    
    print("\n")


class MonteCarlo:
    
    def __init__(self, env: Environment, policy: Dict[State, Action], first_visit: bool = True):
        self.env = env
        self.policy = policy  # Deterministic policy: state -> action
        self.first_visit = first_visit
        self.values = {}
        self.returns = {}  # Dictionary of lists: state -> list of returns
        self._initialize_values()
    
    def _initialize_values(self):
        """Initialize value estimates to zero for all states"""
        for state in self.env.all_states():
            self.values[state] = 0.0
            self.returns[state] = []
    
    def _get_initial_state(self) -> State:
        """Sample initial state from uniform distribution over non-furniture states"""
        valid_states = [s for s in self.env.all_states() 
                       if not self.env.is_blocked(s)]
        idx = np.random.randint(len(valid_states))
        return valid_states[idx]
    
    def _sample_trajectory(self) -> List[Tuple[State, Action, float]]:
        """Sample a trajectory following the policy"""
        trajectory = []
        state = self._get_initial_state()
        
        while not self.env.is_terminal(state):
            # Get action from policy (deterministic)
            action = self.policy[state]
            
            # Sample next state according to transition probabilities
            next_states = list(self.env.transitions[state][action].keys())
            probs = list(self.env.transitions[state][action].values())
            next_state_idx = np.random.choice(len(next_states), p=probs)
            next_state = next_states[next_state_idx]
            
            # Get reward
            reward = self.env.rewards[state][action][next_state]
            
            trajectory.append((state, action, reward))
            state = next_state
        
        return trajectory
    
    def _compute_returns(self, trajectory: List[Tuple[State, Action, float]]) -> List[float]:
        """Compute returns for each step in the trajectory"""
        returns = []
        G = 0.0
        
        # Compute returns backwards
        for i in range(len(trajectory) - 1, -1, -1):
            _, _, reward = trajectory[i]
            G = reward + self.env.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self, trajectory: List[Tuple[State, Action, float]]):
        """Update value estimates based on trajectory"""
        returns = self._compute_returns(trajectory)
        
        if self.first_visit:
            # First-Visit: only use first occurrence of each state
            visited_states = set()
            for i, (state, _, _) in enumerate(trajectory):
                if state not in visited_states:
                    visited_states.add(state)
                    self.returns[state].append(returns[i])
                    # Update value estimate as average of all returns
                    self.values[state] = np.mean(self.returns[state])
        else:
            # Every-Visit: use all occurrences of each state
            for i, (state, _, _) in enumerate(trajectory):
                self.returns[state].append(returns[i])
                # Update value estimate as average of all returns
                self.values[state] = np.mean(self.returns[state])
    
    def run(self, max_iterations: int = 100000, convergence_threshold: float = 0.1, 
            true_values: Optional[Dict[State, float]] = None) -> Tuple[Dict[State, float], int]:
        """
        Run Monte Carlo policy evaluation until convergence
        
        Args:
            max_iterations: Maximum number of trajectories
            convergence_threshold: Max-norm threshold for convergence
            true_values: True value function for convergence checking
        
        Returns:
            (value_estimates, iterations_used)
        """
        for iteration in range(max_iterations):
            trajectory = self._sample_trajectory()
            self.update(trajectory)
            
            # Check convergence if true values provided
            if true_values is not None:
                max_norm = max(abs(self.values[s] - true_values[s]) 
                              for s in self.env.all_states() 
                              if not self.env.is_blocked(s) and not self.env.is_terminal(s))
                if max_norm <= convergence_threshold:
                    return self.values, iteration + 1
        
        return self.values, max_iterations


class MonteCarloControl:
    """Monte Carlo Control with ε-soft policies"""
    
    def __init__(self, env: Environment, epsilon: float = 0.1):
        self.env = env
        self.epsilon = epsilon
        self.q_values = {}  # Q(s, a) estimates
        self.policy = {}    # π(s, a) probabilities
        self.returns = {}   # Dictionary: (state, action) -> list of returns
        self._initialize()
    
    def _initialize(self):
        """Initialize Q-values to zero and policy to uniform random"""
        for state in self.env.all_states():
            if self.env.is_terminal(state):
                continue
            self.q_values[state] = {}
            self.policy[state] = {}
            
            # Initialize Q(s, a) = 0 for all actions
            for action in self.env.action_names:
                self.q_values[state][action] = 0.0
                self.returns[(state, action)] = []
            
            # Initialize policy to uniform random
            num_actions = len(self.env.action_names)
            for action in self.env.action_names:
                self.policy[state][action] = 1.0 / num_actions
    
    def _get_initial_state(self) -> State:
        """Sample initial state from uniform distribution over non-furniture states"""
        valid_states = [s for s in self.env.all_states() 
                       if not self.env.is_blocked(s)]
        idx = np.random.randint(len(valid_states))
        return valid_states[idx]
    
    def _sample_action(self, state: State) -> Action:
        """Sample action according to current policy"""
        actions = list(self.policy[state].keys())
        probs = list(self.policy[state].values())
        return np.random.choice(actions, p=probs)
    
    def _sample_trajectory(self) -> List[Tuple[State, Action, float]]:
        """Sample a trajectory following the current policy"""
        trajectory = []
        state = self._get_initial_state()
        
        while not self.env.is_terminal(state):
            action = self._sample_action(state)
            
            # Sample next state according to transition probabilities
            next_states = list(self.env.transitions[state][action].keys())
            probs = list(self.env.transitions[state][action].values())
            next_state_idx = np.random.choice(len(next_states), p=probs)
            next_state = next_states[next_state_idx]
            
            # Get reward
            reward = self.env.rewards[state][action][next_state]
            
            trajectory.append((state, action, reward))
            state = next_state
        
        return trajectory
    
    def _compute_returns(self, trajectory: List[Tuple[State, Action, float]]) -> List[float]:
        """Compute returns for each step in the trajectory"""
        returns = []
        G = 0.0
        
        # Compute returns backwards
        for i in range(len(trajectory) - 1, -1, -1):
            _, _, reward = trajectory[i]
            G = reward + self.env.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def _update_policy(self, state: State):
        """Update policy using ε-soft policy update with tie-breaking"""
        # Find optimal actions (may be multiple)
        q_vals = [self.q_values[state][a] for a in self.env.action_names]
        max_q = max(q_vals)
        optimal_actions = [a for a in self.env.action_names 
                          if abs(self.q_values[state][a] - max_q) < 1e-10]
        
        num_actions = len(self.env.action_names)
        num_optimal = len(optimal_actions)
        
        # Update policy according to ε-soft formula
        for action in self.env.action_names:
            if action in optimal_actions:
                self.policy[state][action] = (1 - self.epsilon) / num_optimal + self.epsilon / num_actions
            else:
                self.policy[state][action] = self.epsilon / num_actions
    
    def update(self, trajectory: List[Tuple[State, Action, float]]):
        """Update Q-values and policy based on trajectory (First-Visit)"""
        returns = self._compute_returns(trajectory)
        visited_pairs = set()
        
        for i, (state, action, _) in enumerate(trajectory):
            if self.env.is_terminal(state):
                continue
            
            pair = (state, action)
            if pair not in visited_pairs:
                visited_pairs.add(pair)
                self.returns[pair].append(returns[i])
                # Update Q-value as average of all returns
                self.q_values[state][action] = np.mean(self.returns[pair])
                # Update policy for this state
                self._update_policy(state)
    
    def get_value_function(self) -> Dict[State, float]:
        """Compute value function from Q-values and policy: V(s) = Σ_a π(s,a) Q(s,a)"""
        values = {}
        for state in self.env.all_states():
            if self.env.is_terminal(state):
                values[state] = 0.0
            elif state in self.policy:
                values[state] = sum(self.policy[state][a] * self.q_values[state][a] 
                                   for a in self.env.action_names)
            else:
                values[state] = 0.0
        return values
    
    def run(self, num_iterations: int, epsilon_schedule: Optional[callable] = None) -> Dict[State, float]:
        """
        Run Monte Carlo control for specified number of iterations
        
        Args:
            num_iterations: Number of trajectories to sample
            epsilon_schedule: Optional function f(iteration) -> epsilon
        
        Returns:
            Final value function estimate
        """
        for iteration in range(num_iterations):
            # Update epsilon if schedule provided
            if epsilon_schedule is not None:
                self.epsilon = epsilon_schedule(iteration)
            
            trajectory = self._sample_trajectory()
            self.update(trajectory)
        
        return self.get_value_function()

