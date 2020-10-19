import gym
import gym_chrisww
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DQN
import tensorflow as tf
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
import numpy as np
import pygame
#from sklearn.model_selection import RandomizedSearchCV

#Demonstration
def human_expert(_obs):
    env.render()
    key=input()
    thisdict = {
        "w": 3,
        "a": 1,
        "s": 2,
        "d": 0
        }
    if key in thisdict:
        return thisdict[key]
    else:
        return env.action_space.sample()
    
#Demonstration
def simple_expert(_obs):
    state=np.reshape(_obs,(5,5,2))
    x1,y1=np.where(state[:,:,0]==True)
    x2,y2=np.where(state[:,:,1]==True)
    if(x1>x2):
        return 1
    if(y1>y2):
        return 3
    if(x1<x2):
        return 0
    else:
        return 2



generate_expert_traj(simple_expert, 'expert_game', env, n_episodes=1000)

dataset = ExpertDataset(expert_path= 'expert_game.npz',
                        traj_limitation=1, batch_size=100)

#Random Policy
env = gym.make('chrisww-v0')
env.reset()
for _ in range(50):
    env.render()
    state,reward,done,info=env.step(env.action_space.sample())
    print(reward)
env.close()



#DQN Policy
model = DQN("MlpPolicy", env, verbose=1,learning_rate=5.5e-4)
model.pretrain(dataset, n_epochs=10000)
model.learn(total_timesteps=int(2e3))

def demonstrate():
    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones==True:
            obs = env.reset()  
    env.close()

def play_game():
    done=False
    obs = env.reset()
    while(done==False):
        env.render()
        pressed_keys = pygame.key.get_pressed()
        thisdict = {
        "w": 3,
        "a": 1,
        "s": 2,
        "d": 0
        }
        action=4
        if pressed_keys[pygame.K_w]:
            action=3
        elif pressed_keys[pygame.K_s]:
            action=2
        elif pressed_keys[pygame.K_a]:
            action=1
        elif pressed_keys[pygame.K_d]:
            action=0
        print(sum(pressed_keys))        
        obs, rewards, done, info = env.step(action)
        env.render()
    env.close()



