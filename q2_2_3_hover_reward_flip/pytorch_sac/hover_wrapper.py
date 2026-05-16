import gymnasium as gym

class HoverRewardWrapper(gym.Wrapper):
    def __init__(self, env, hover_reward=200.0):
        super().__init__(env)
        self.hover_reward = hover_reward
        self.has_received_hover_reward = False

    def reset(self, **kwargs):
        self.has_received_hover_reward = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        x, y = obs[0], obs[1]
        
        if not self.has_received_hover_reward:
            if abs(x) < 0.1 and 0.4 < abs(y) < 0.6:
                reward += self.hover_reward
                self.has_received_hover_reward = True
                
        return obs, reward, terminated, truncated, info
