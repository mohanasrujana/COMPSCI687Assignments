from dataclasses import dataclass
from re import S
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from environment import Environment

State = Tuple[int, int]
Action = str


def arrow_symbol(action_code: Action) -> str:
    return {
        "AU": "↑",
        "AD": "↓",
        "AL": "←",
        "AR": "→",
        "G": "G",
    }[action_code]


def max_norm(a: Dict[State, float], b: Dict[State, float], ignore: Optional[Set[State]] = None) -> float:
    ks = a.keys()
    if ignore:
        ks = [k for k in ks if k not in ignore]
    return max(abs(a[s] - b[s]) for s in ks)


def format_value(values: Dict[State, float], env: Environment) -> List[List[str]]:
    table = []
    for r in range(env.rows):
        rowmat = []
        for c in range(env.cols):
            if env.is_blocked((r, c)):
                rowmat.append("X")
            else:
                rowmat.append(f"{values[(r, c)]:.4f}")
        table.append(rowmat)
    return table


def format_policy(policy: Dict[State, Action], env: Environment) -> List[List[str]]:
    grid = []
    for r in range(env.rows):
        rowmat = []
        for c in range(env.cols):
            state = (r, c)
            if env.is_blocked(state):
                rowmat.append("X")
            else:
                act = policy.get(state, "G")
                rowmat.append(arrow_symbol(act))
        grid.append(rowmat)
    return grid


def greedypol_q(q_values: np.ndarray, env: Environment) -> Dict[State, Action]:
    pol = {}
    acts = env.action_names
    for s in env.all_states():
        if env.is_blocked(s):
            pol[s] = "X"
            continue
        if env.is_terminal(s):
            pol[s] = "G"
            continue
        r, c = s
        bi = int(np.argmax(q_values[r, c]))
        pol[s] = acts[bi]
    return pol


def s_0(env: Environment, rng: np.random.Generator) -> State:
    candidates = [s for s in env.all_states() if not env.is_blocked(s)]
    i = int(rng.integers(len(candidates)))
    return candidates[i]


def trans_state(env: Environment, state: State, action: Action, rng: np.random.Generator) -> Tuple[State, float, bool]:
    trans_dic = env.transitions[state][action]
    states = list(trans_dic.keys())
    probs = np.array(list(trans_dic.values()), dtype=float)
    probs = probs / probs.sum()
    i = int(rng.choice(len(states), p=probs))
    ns = states[i]
    r = env.rewards[state][action][ns]
    done = env.is_terminal(ns)
    return ns, r, done


@dataclass
class TDConfiguration:
    alpha: float
    delta: float
    min_episodes: int
    max_episodes: int
    max_steps: int


class TDLearning:
    def __init__(self, env: Environment, policy: Dict[State, Action], config: TDConfiguration):
        self.env = env
        self.policy = policy
        self.alpha = config.alpha
        self.delta = config.delta
        self.min_episodes = config.min_episodes
        self.max_episodes = config.max_episodes
        self.max_steps = config.max_steps

    def run(self, seed: int) -> Tuple[Dict[State, float], int]:
        rng = np.random.default_rng(seed)
        vals = {state: 0.0 for state in self.env.all_states()}
        vals_old,eps = vals.copy(),0

        while eps < self.max_episodes:
            eps += 1
            state = s_0(self.env, rng)
            done = self.env.is_terminal(state)
            steps = 0
            while not done and steps < self.max_steps:
                action = self.policy[state]
                ns, r, done = trans_state(self.env, state, action, rng)
                tt = r
                if not done:
                    tt += self.env.gamma * vals[ns]
                vals[state] += self.alpha * (tt - vals[state])
                state = ns
                steps += 1

            delta = max(abs(vals[s] - vals_old[s]) for s in vals.keys())
            if eps >= self.min_episodes and delta <= self.delta:
                break
            vals_old = vals.copy()

        return vals, eps


class BaseTDLA:
    def __init__(
        self,
        env: Environment,
        alpha: float,
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float,
        q_init: float,
        max_steps: int,
    ):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_steps = max_steps
        self.actions = env.action_names
        self.num_actions = len(self.actions)
        self.q_values = np.full((env.rows, env.cols, self.num_actions), q_init, dtype=float)
        for s in env.all_states():
            if env.is_terminal(s):
                r, c = s
                self.q_values[r, c, :] = 0.0

    def _indices(self, state: State) -> Tuple[int, int]:
        return state

    def sample_start_state(self, rng: np.random.Generator) -> State:
        s = s_0(self.env, rng)
        while self.env.is_terminal(s):
            s = s_0(self.env, rng)
        return s

    def greedy_acti(self, state: State) -> int:
        r, c = self._indices(state)
        return int(np.argmax(self.q_values[r, c]))

    def greedye(self, state: State, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.num_actions))
        return self.greedy_acti(state)

    def decaye(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def value_from_q(self, epsilon: float) -> Dict[State, float]:
        vals = {}
        for state in self.env.all_states():
            r, c = state
            q_row = self.q_values[r, c]
            best_value = np.max(q_row)
            optimal_actions = np.flatnonzero(np.isclose(q_row, best_value))
            probs = np.full(self.num_actions, epsilon / self.num_actions)
            if len(optimal_actions) > 0:
                probs[optimal_actions] += (1 - epsilon) / len(optimal_actions)
            vals[state] = float(np.dot(probs, q_row))
        return vals

    def greedy_policy(self) -> Dict[State, Action]:
        pol = {}
        for state in self.env.all_states():
            if self.env.is_blocked(state):
                pol[state] = "X"
                continue
            if self.env.is_terminal(state):
                pol[state] = "G"
                continue
            bi = self.greedy_acti(state)
            pol[state] = self.actions[bi]
        return pol


class SarsaAgent(BaseTDLA):
    def train(self, num_episodes: int, seed: int, optimal_values: Dict[State, float]) -> Dict[str, List[float]]:
        rng = np.random.default_rng(seed)
        acts_cum,shist,msehist = 0,[],[]

        for x in range(num_episodes):
            s = self.sample_start_state(rng)
            a_dash = self.greedye(s, rng)
            done = False

            stps = 0
            while not done and stps < self.max_steps:
                a = self.actions[a_dash]
                ns, r, done = trans_state(self.env, s, a, rng)
                stps += 1
                acts_cum += 1
                if done:
                    tt = r
                else:
                    a_1_dash = self.greedye(ns, rng)
                    nr, nc = ns
                    tt = r + self.env.gamma * self.q_values[nr, nc, a_1_dash]

                r, c = s
                self.q_values[r, c, a_dash] += self.alpha * (tt - self.q_values[r, c, a_dash])

                if done:
                    break

                s, a_dash = ns, a_1_dash

            shist.append(acts_cum)
            msehist.append(self._mse(optimal_values, self.value_from_q(self.epsilon)))
            self.decaye()

        return {
            "steps_history": shist,
            "mse_history": msehist,
            "q_values": self.q_values,
        }

    def _mse(self, optimal: Dict[State, float], estimate: Dict[State, float]) -> float:
        errs = [(estimate[s] - optimal[s]) ** 2 for s in optimal.keys()]
        return float(np.mean(errs))


class QLearningAgent(BaseTDLA):
    def train(self, num_episodes: int, seed: int, optimal_values: Dict[State, float]) -> Dict[str, List[float]]:
        rng = np.random.default_rng(seed)
        acts_cum,shist,msehist = 0,[],[]

        for x in range(num_episodes):
            s = self.sample_start_state(rng)
            done = False
            stps = 0

            while not done and stps < self.max_steps:
                a_dash = self.greedye(s, rng)
                a = self.actions[a_dash]
                ns, r, done = trans_state(self.env, s, a, rng)
                r, c = s
                stps += 1
                acts_cum += 1
                tt = r
                if not done:
                    nr, nc = ns
                    tt += self.env.gamma * np.max(self.q_values[nr, nc])

                self.q_values[r, c, a_dash] += self.alpha * (tt - self.q_values[r, c, a_dash])

                s = ns

            shist.append(acts_cum)
            msehist.append(self._mse(optimal_values, self._value_from_greedy_policy()))
            self.decaye()

        return {
            "steps_history": shist,
            "mse_history": msehist,
            "q_values": self.q_values,
        }

    def _value_from_greedy_policy(self) -> Dict[State, float]:
        vals = {}
        for state in self.env.all_states():
            r, c = state
            vals[state] = float(np.max(self.q_values[r, c]))
        return vals

    def _mse(self, optimal: Dict[State, float], estimate: Dict[State, float]) -> float:
        errs = [(estimate[s] - optimal[s]) ** 2 for s in optimal.keys()]
        return float(np.mean(errs))

