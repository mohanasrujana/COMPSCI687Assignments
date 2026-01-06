import numpy as np
from policy import PolicyNet, MIN_POSITION, MAX_POSITION, MIN_VELOCITY, MAX_VELOCITY, TIMEOUT

class MountainCarEnv:

    def __init__(self):
        self.min_position = MIN_POSITION
        self.max_position = MAX_POSITION
        self.min_velocity = MIN_VELOCITY
        self.max_velocity = MAX_VELOCITY
        self.timeout = TIMEOUT
        self.position = None
        self.velocity = None
        self.time_step = 0
        self.goal_reached = False

    def reset(self):
        #gievn d0 = (X0,0) where X0 belongs to [-0.6,-0.4]
        self.position = np.random.uniform(low=-0.6, high=-0.4)
        self.velocity = 0.0
        self.time_step = 0
        self.goal_reached = False
        return np.array([self.position, self.velocity])

    def dynamics(self, action: float):
        #v_t+1 <- v_t + 0.001*a_t - 0.0025*cos(3*x_t)
        v_new = self.velocity + 0.001 * action - 0.0025 * np.cos(3 * self.position)
        v_new = clip(v_new, self.min_velocity, self.max_velocity) #clipping if not in range
        x_new = self.position + v_new
        x_new = clip(x_new, self.min_position, self.max_position) #clipping if not in range

        if x_new == self.min_position:
            v_new = 0.0
        self.goal_reached = x_new >= self.max_position
        self.position = x_new
        self.velocity = v_new
        self.time_step += 1
        
        next_state = np.array([self.position, self.velocity])

        done = False
        reward = -1.0 

        if self.goal_reached:
            reward = 0.0
            done = True

        elif self.time_step >= self.timeout:
            done = True
            
        return next_state, reward, done


def estimate_J(policy: PolicyNet, theta: np.ndarray, num_episodes: int = 15) -> float:
    policy.set_policy_parameters(theta)
    env = MountainCarEnv()
    returns = []    
    for _ in range(num_episodes):
        state = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = policy.act(state)
            state, reward, done = env.dynamics(action)
            episode_return += reward    
        returns.append(episode_return)
    return float(np.mean(returns))

def clip(value, min_val, max_val):
    return np.clip(value, min_val, max_val)
