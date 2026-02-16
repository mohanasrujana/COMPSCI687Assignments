import numpy as np
from typing import Dict, Tuple

# Type definitions
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
                    for next_state,p in self.transitions[state][action_name].items():
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
     
     def all_states(self) -> Tuple[int, int]:
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                states.append((r, c))
        return states