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
from gym.envs.registration import register

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
    state=np.reshape(_obs,(10,10,2))
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

def demonstrate(env,model,iter):
    obs = env.reset()
    for _ in range(iter):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones==True:
            obs = env.reset()  
    env.close()

def play_game(env,iter):
    for i in range(iter):
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
            obs, rewards, done, info = env.step(action)
    env.close()




#Random Policy
def play_random(env,iter):
    state=env.reset()
    for i in range(iter):
        env.render()
        state,reward,done,info=env.step(env.action_space.sample())
    env.close()

#level_0= gym.make('chrisww-v0' )
level_1= gym.make('chrisww-v1')
#DQN Policy
model_1 = DQN("MlpPolicy", level_1, verbose=1,learning_rate=1e-4,exploration_fraction=0.2)
model_1.pretrain(dataset,learning_rate=0.00001, n_epochs=100)
model_1.learn(total_timesteps=int(1e4))


generate_expert_traj(simple_expert, 'expert_game', level_1, n_episodes=10000)

dataset = ExpertDataset(expert_path= 'expert_game.npz',
                        traj_limitation=-1, batch_size=100)


