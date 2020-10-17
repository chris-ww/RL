import gym
import gym_chrisww
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import ACER
from stable_baselines.common.policies import FeedForwardPolicy


#Random Policy
env = gym.make('chrisww-v0')
state=env.reset()
for _ in range(500):
    env.render()
    state,reward,done=env.step(env.action_space.sample())
env.close()



#DQN Policy

# Train the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
model.learn(total_timesteps=int(2e5))

# 2. To check all env available, uninstalled ones are also shown
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()