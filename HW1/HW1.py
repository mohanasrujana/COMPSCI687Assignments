import numpy as np
import matplotlib.pyplot as plt
import itertools

class Model:
    
    states  = ["s1","s2","s3","s4","s5","s6","s7"]
    terminal_states = ["s6","s7"]
    actions = ["a1","a2"]
    initial_states = {"s1":0.6,"s2":0.3,"s3":0.1}
    transition_probabilities = {"s1":{"a1":{"s4":1.0},"a2":{"s4":1.0}},
                                    "s2":{"a1":{"s4":0.8,"s5":0.2},"a2":{"s4":0.6,"s5":0.4}},
                                    "s3":{"a1":{"s4":0.9,"s5":0.1},"a2":{"s4":1.0}},
                                    "s4":{"a1":{"s6":1.0},"a2":{"s6":0.3,"s7":0.7}},
                                    "s5":{"a1":{"s6":0.3,"s7":0.7},"a2":{"s7":1.0}}}
    rewards ={"s1":{"a1":7,"a2":10},
            "s2":{"a1":-3,"a2":5},
            "s3":{"a1":4,"a2":-6},
            "s4":{"a1":9,"a2":-1},
            "s5":{"a1":-8,"a2":2}
            }
    
    pi ={"s1":{"a1":0.5,"a2":0.5},
            "s2":{"a1":0.7,"a2":0.3},
            "s3":{"a1":0.9,"a2":0.1},
            "s4":{"a1":0.4,"a2":0.6},
            "s5":{"a1":0.2,"a2":0.8}
            }

class MDP:
    @staticmethod
    def runEpisode(policy, gamma):
        s = np.random.choice(list(Model.initial_states.keys()), p=list(Model.initial_states.values()))
        disc_ret = 0
        count = 0
        while s not in Model.terminal_states:
            
            if s not in policy:
                raise ValueError(f" No Policy of state {s}")
                
            actions = list(policy[s].keys())
            action_probs = list(policy[s].values())
            
            if not np.isclose(sum(action_probs), 1.0):
                action_probs = np.array(action_probs) / sum(action_probs)

            action = np.random.choice(actions, p=action_probs)
            
            available_ns = list(Model.transition_probabilities[s][action].keys())
            ns_probs = list(Model.transition_probabilities[s][action].values())
            ns = np.random.choice(available_ns, p=ns_probs)
            
            reward = Model.rewards[s][action]
            
            s = ns
            disc_state = (gamma ** count) * reward
            disc_ret += disc_state
            count += 1
            
        return disc_ret

def estimate(policy, episode_count, gamma):
    r_all = []
    J_cap = []
    tot_r= 0
    
    for i in range(episode_count):       
        disc_ret = MDP.runEpisode(policy, gamma)
        
        r_all.append(disc_ret)
        tot_r += disc_ret
        J_cap.append(tot_r/ (i + 1))
        
    avg_reward = np.mean(r_all)
    var_reward = np.var(r_all)
    
    return avg_reward, var_reward, J_cap, r_all

def get_true_j_pi(gamma):
    """
    Takes gamma as input
    Returns J(pi) at gamma
    """
    C1,C2=5.22,2.709
    return C1 + gamma * C2


def format_policy(policy):
    formatted_policy = {}
    for state, action in policy.items():
        if action == 'a1':
            formatted_policy[state] = {'a1': 1.0, 'a2': 0.0}
        else:
            formatted_policy[state] = {'a1': 0.0, 'a2': 1.0}
    return formatted_policy

def display_deterministic_pi(pi):
    """
    Takes pi as input
    Returns a formatted version of the deterministic policy.
    """
    formatted = {}
    for s, action_prob_dict in pi.items():
        action = next(a for a, p in action_prob_dict.items() if p == 1.0)
        formatted[s] = action
    return formatted

def q2():
    policy = Model.pi
    gamma = 0.9  
    episode_count = 150000  
    J_avg, variance, J_cap, r = estimate(policy, episode_count, gamma)
    
    return J_avg, variance, J_cap, episode_count, gamma

# Question 2a
def question_2a(J_cap, episode_count, gamma):
    print("\nQuestion 2a \n")
    
    x = list(range(1, episode_count + 1))
    y = J_cap
    
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of episodes (N)")
    plt.ylabel("Estimated J(π)")
    plt.title(f"Estimated J(π) vs Number of episodes (γ={gamma}, N={episode_count})")
    plt.plot(x, y)
    plt.grid(True)
    plt.show()


# Question 2b
def question_2b(J_avg, variance):

    print("\n Question 2b \n")
    print(f"Average of discounted rewards (Estimated J(π)) = {J_avg:.4f}") 
    print(f"Variance of discounted rewards = {variance:.4f}") 

# Question 2c
def question_2c():
    print("\n Question 2c \n")
    
    policy = Model.pi
    gammas = [0.25, 0.5, 0.75, 0.99]
    episode_count = 150000  
    
    results = {}
    print("γ | Estimated J(π)  | True J(π)      ")
    print("--|-----------------|----------------")
    
    for gamma in gammas:
        J_avg, variance, J_cap, r  = estimate(policy, episode_count, gamma)
        results[gamma] = J_avg
        true_j = get_true_j_pi(gamma)
        print(f"{gamma} | {J_avg:.8f} | {true_j:.8f}")
        
    return results

# Question 2d 
def question_2d():
      print("\n Question 2d \n")
      
      gamma = 0.75 
      episode_count = 350000 
      states = ["s1", "s2", "s3", "s4", "s5"]
      actions = ["a1", "a2"]
      all_policies = []
      for a in itertools.product(actions, repeat=len(states)):
          policy = dict(zip(states, a))
          all_policies.append(format_policy(policy))

      best_pi = None
      best_J_cap = -np.inf

      for p in all_policies:
          J_avg, variance, J_cap, r  = estimate(p, episode_count, gamma)
          
          if J_avg > best_J_cap:
              best_J_cap = J_avg
              best_pi = p

              
      
      print("Search Method: Brute Force Search")
      print(f"Best Policy (π*): {display_deterministic_pi(best_pi)}")
      print(f"Estimated Performance J(π*): {best_J_cap:.4f}")
      
      return best_pi, best_J_cap

def main():
    J_avg, variance, J_cap, episode_count, gamma = q2()
    question_2a(J_cap, episode_count, gamma)
    question_2b(J_avg, variance)
    question_2c()
    question_2d()

if __name__ == "__main__":
    main()