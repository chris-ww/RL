import gym
import gym_chrisww
env = gym.make('chrisww-v0')
env.reset()
for _ in range(500):
    env.render()
    env.step(env.action_space.sample())
env.close()
# 2. To check all env available, uninstalled ones are also shown
from gym import envs 
print(envs.registry.all()