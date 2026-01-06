from environment import Environment
import math

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

if __name__ == "__main__":
    # Q1
    env = Environment(gamma=0.925)
    print("Q1")
    run_format(env)

    # Q2
    env2 = Environment(gamma=0.2)
    print("Q2")
    run_format(env2)

    # Q3
    env3 = Environment(gamma=0.925, reward=5.0, terminal=False)
    env3.catnip = (0, 1) if env3.reward != 0.0 else None
    env3._set_rewards()
    print("Q3")
    run_format(env3)

    # Q4 
    print("Q4")
    env4_base = Environment(gamma=0.925, reward=5.0, terminal=True)
    env4_base.catnip = (0, 1) if env4_base.reward != 0.0 else None
    env4_base._set_rewards() 
    run_format(env4_base)
    base_gamma = 0.925
    policy_change_gamma = None
    vi4_base = ValueIteration(env4_base)
    values_base_ref, policy_base_ref, _ = vi4_base.run()
    
    for g in [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        env4 = Environment(gamma=g, reward=5.0, terminal=True)
        env4.catnip = (0, 1) if env4.reward != 0.0 else None
        env4._set_rewards()
        vi4 = ValueIteration(env4)
        values, policy, iters = vi4.run()
        policy_changed = False
        for state in env4.all_states():
            if env4.is_terminal(state):
                continue
            if policy[state] != policy_base_ref[state]:
                policy_changed = True
                break
        
        if policy_changed:
            start_value_diff = abs(values[(0, 0)] - values_base_ref[(0, 0)])
            state_11_value_diff = abs(values[(1, 1)] - values_base_ref[(1, 1)])
            if start_value_diff > 0.05 or state_11_value_diff > 0.05:
                policy_change_gamma = g
                break
    
    if policy_change_gamma is not None:
        env4_change = Environment(gamma=policy_change_gamma, reward=5.0, terminal=True)
        env4_change.catnip = (0, 1) if env4_change.reward != 0.0 else None
        env4_change._set_rewards()
        run_format(env4_change)
   

    # Q5
    print("Q5")
    base_reward = 5.0
    policy_change_reward = None
    
    env5_base = Environment(gamma=0.925, reward=base_reward, terminal=True)
    env5_base.catnip = (0, 1)
    env5_base._set_rewards()
    vi5_base = ValueIteration(env5_base)
    values_base_ref, policy_base_ref, _ = vi5_base.run()
    print("Reward = 5.0")
    run_format(env5_base)
    reward_list = [4.5, 4.0, 3.9, 3.8, 3.5, 3.0]
    
    for rew in reward_list:
        env5 = Environment(gamma=0.925, reward=rew, terminal=True)
        env5.catnip = (0, 1) if env5.reward != 0.0 else None
        env5._set_rewards()
        vi5 = ValueIteration(env5)
        values, policy, iters = vi5.run()

        policy_changed = False
        for state in env5.all_states():
            if env5.is_terminal(state):
                continue
            if policy[state] != policy_base_ref[state]:
                policy_changed = True
                break
        
        if policy_changed:
            policy_change_reward = rew
            break
    
    if policy_change_reward is not None:
        print(f"Reward = {policy_change_reward}")
        env5_change = Environment(gamma=0.925, reward=policy_change_reward, terminal=True)
        env5_change.catnip = (0, 1)
        env5_change._set_rewards()
        run_format(env5_change)



    