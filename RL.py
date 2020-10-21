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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import random

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
    state=np.reshape(_obs,(10,10,3))
    x1,y1=np.where(state[:,:,0]==True)
    x2,y2=np.where(state[:,:,1]==True)
    able=[]
    if(x1>x2):
        able.append(1)
    if(y1>y2):
        able.append(3)
    if(x1<x2):
        able.append(0)
    else:
        able.append(2)
    if not able:
        return random.randint(0,3)
    choice=random.sample(able,1)

def avoid_red(_obs):
    state=np.reshape(_obs,(10,10,3))
    x1,y1=np.where(state[:,:,0]==True)
    able=[0,1,2,3]
    if(x1>8):
        able.remove(0)
    if(x1<1):
        able.remove(1)
    if(y1>8):
        able.remove(2)
    if(y1<1):
        able.remove(3)
    if(state[x1+1,y1,2]==True):
        able.remove(0)
    if(state[x1-1,y1,2]==True):
        able.remove(1)
    if(state[x1,y1+1,2]==True):
        able.remove(2)
    if(state[x1-1,y1-1,2]==True):
        able.remove(3)
    if not able:
        return random.randint(0,3)
    choice=random.sample(able,1)



def demonstrate(env,model,iter):
    obs = env.reset()
    for _ in range(iter):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if(dones==True):
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


def play_rule_based(env,iter):
    for i in range(iter):
        done=False
        obs = env.reset()
        while(done==False):
            env.render()
            state=np.reshape(obs,(10,10,3))
            x1,y1=np.where(state[:,:,0]==True)
            able=[0,1,2,3]
            if(x1>8):
                able.remove(0)
            else:
                if(state[x1+1,y1,2]==True):
                    able.remove(1)
            if(x1<1):
                able.remove(1)
            else:
                if(state[x1-1,y1,2]==True):
                    able.remove(0)
            if(y1>8):
                able.remove(2)
            else:
                if(state[x1-1,y1-1,2]==True):
                    able.remove(3)
            if(y1<1):
                able.remove(3)
            else:
                if(state[x1,y1+1,2]==True):
                    able.remove(2)
            if not able:
                action=random.randint(0,3)
            else:
                action=random.sample(able,1)
            obs, rewards, done, info = env.step(action)
        env.close()


def get_average_reward(env,model,iter):
    obs = env.reset()
    total_reward=0
    for _ in range(iter):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        total_reward+=rewards
        if dones==True:
            obs = env.reset()  
    env.close()
    return total_reward

#Random Policy
def play_random(env,iter):
    state=env.reset()
    for i in range(iter):
        env.render()
        state,reward,done,info=env.step(env.action_space.sample())
        if(done==True):
            state=env.reset()
    env.close()



#level 0
#level_0= gym.make('chrisww-v0' )
#model_0 = DQN("MlpPolicy", level_0, verbose=1,learning_rate=1e-4,exploration_fraction=0.2)
#model_0.learn(total_timesteps=int(1e4))

#level 1
#generate_expert_traj(simple_expert, 'expert_game', level_1, n_episodes=10000)
#dataset = ExpertDataset(expert_path= 'expert_game.npz',
#                        traj_limitation=-1, batch_size=100)
#level_1= gym.make('chrisww-v1')
#model_1 = DQN("MlpPolicy", level_1, verbose=1,learning_rate=1e-4,exploration_fraction=0.2)
#model_1.pretrain(dataset,learning_rate=0.00001, n_epochs=250)
#model_1.learn(total_timesteps=int(1e4))

#level_2
level_1= gym.make('chrisww-v2',mines=10,r_mines=-1,r_door=0,r_dist=0)
generate_expert_traj(avoid_red, 'expert_game', level_1, n_episodes=1)
dataset = ExpertDataset(expert_path= 'expert_game.npz',
                        traj_limitation=-1, batch_size=100)

space = dict()
space['learning_rate']=loguniform(1e-7, 1e-3)
space['exploration_fraction']= loguniform(0.01, 0.7)
space['layers']= [1, 2,3,4]
space['size']=[10,30,60,120,240]


model_2 = PPO2("MlpPolicy", level_3, verbose=1,learning_rate=1e-4,n_steps=200,policy_kwargs={'layers':[120,120,120]})
model_2.pretrain(dataset,learning_rate=0.01, n_epochs=5)
model_2.learn(total_timesteps=int(1e4))

search = RandomizedSearchCV(model_2, space, n_iter=500, scoring=get_average_Reward, n_jobs=-1, cv=None, random_state=1)


dict(learning_rate = 1e-5,)