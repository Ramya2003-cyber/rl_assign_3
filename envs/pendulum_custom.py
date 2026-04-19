import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class TargetAnglePendulum (gym.Wrapper):
    def __init__(self,env,target_angle,reward_scale=1.0):
        super().__init__(env)
        self.env=env
        self.target_angle=np.radians(target_angle)
        self.reward_scale=reward_scale
    
    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        cos_theta,sin_theta,angular_velocity=obs
        rad_theta=np.arctan2(sin_theta,cos_theta)
        angular_error=(rad_theta-self.target_angle+np.pi)%(2*np.pi)-np.pi
        custom_reward=-(angular_error**2 + 0.1*angular_velocity**2 + 0.001*action[0]**2)
        custom_reward*=self.reward_scale
        return obs,custom_reward,terminated,truncated,info

