"""
Modified from https://github.com/ku2482/sac-discrete.pytorch
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as opt
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os
import numpy as np
import pandas as pd
import yaml
import argparse
from datetime import datetime
from collections import deque
from abc import ABC, abstractmethod
from collections import namedtuple
import scipy.optimize as sciopt
from itertools import count
from itertools import product
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
from os.path import join as joindir
from munch import DefaultMunch
from tqdm import trange
import pickle
import random
import time
import math
import pdb

from utils import update_params, disable_gradients, initialize_weights_he
from utils import Flatten, RunningMeanStats, ZFilter, NDStandardScaler
from env import make_env, VectorEnv
from memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory


class ClassicalAgent(object):

    def __init__(self, args=None, config=None):

        if args is not None:
            # Read config
            with open(args.config) as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            self.config = DefaultMunch.fromDict(config)
            self.method = args.method
            self.config_name = args.config.split('/')[-1].rstrip('.yaml')
        elif config is not None:
            self.config = config
            self.method = config.method
            self.config_name = config.config_name
        else:
            raise NotImplementedError

        self.env_valid = make_env(self.config.env, etype='valid')
        self.env_valid.set_coeff(0)

        self.seed = self.config.basic.seed
        self.set_seed(self.seed)

        self.exp_name = self.method + '_' + self.config_name
        self.exp_time = datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = os.path.join('logs', 
            '{name}-{seed}-{time}'.format(name=self.exp_name, 
            seed=self.seed, time=self.exp_time))
        self.record_dir = os.path.join(self.log_dir, 'record')
        os.makedirs(self.record_dir, exist_ok=True)

        self.gamma = self.config.algo.gamma
        self.evaluate_steps = self.config.algo.evaluate_steps
        self.num_parallel_envs = self.config.env.num_parallel_envs

    @staticmethod
    def set_seed(seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def evaluate(self):
        record = []
        for basestock_level in range(self.config.env.action_space_size * 3):
            if self.method == 'basestock':
                rp_list = [basestock_level]
            elif self.method == 'ss_policy':
                rp_list = range(basestock_level)
            for reorder_point in rp_list:
                record.append(self.evaluate_single(basestock_level, reorder_point))
                print(record[-1])
        record = pd.DataFrame(record)
        record.to_csv(os.path.join(self.record_dir, 'result.csv'))

        stats = self.optimize(record)
        stats = pd.DataFrame(stats)
        stats.to_csv(os.path.join(self.record_dir, 'stats.csv'))

    def optimize(self, data):

        c = - data['mean_return_discount'].values
        A = data['mean_budget_discount'].values.reshape(1, -1)
        ones = np.ones(A.shape)

        record = []
        for B in np.linspace(A.min(), A.max(), 100):
            res = linprog(c, A, B, ones, 1)
            budget = (A @ res.x)[0]
            cost = c @ res.x

            record.append({'budget': budget, 'cost': cost})

        return record

    def policy(self, basestock_level, reorder_point, state):
        if self.num_parallel_envs > 1:
            action = basestock_level - np.sum(state, 1)
            action[action < reorder_point] = 0
            action = np.clip(action, 0, self.config.env.action_space_size - 1)
        else:
            action = basestock_level - np.sum(state)
            action[action < reorder_point] = 0
            action = np.clip(action, 0, self.config.env.action_space_size - 1)
        return action

    def evaluate_single(self, basestock_level, reorder_point):

        num_episodes = 0
        num_steps = 0
        total_return = 0
        total_budget = 0
        total_return_discount = 0
        total_budget_discount = 0

        while True:
            state = self.env_valid.reset()
            episode_steps = 0
            episode_return = 0
            episode_budget = 0
            episode_return_discount = 0
            episode_budget_discount = 0
            done = False
            while not done:
                action = self.policy(basestock_level, reorder_point, state)
                next_state, reward, done, info = self.env_valid.step(action)
                num_steps += 1
                episode_return += np.sum(reward)
                episode_budget += info[-1][self.config.env.budget]
                episode_return_discount += np.sum(reward) * self.gamma ** episode_steps
                episode_budget_discount += info[-1][self.config.env.budget] * self.gamma ** episode_steps
                episode_steps += 1
                state = next_state

            num_episodes += self.num_parallel_envs
            total_return += episode_return
            total_budget += episode_budget
            total_return_discount += episode_return_discount
            total_budget_discount += episode_budget_discount 

            if num_steps > self.evaluate_steps:
                break

        mean_return = total_return / num_episodes
        mean_budget = total_budget / num_episodes
        mean_return_discount = total_return_discount / num_episodes
        mean_budget_discount = total_budget_discount / num_episodes

        return dict(
            basestock_level=basestock_level,
            reorder_point=reorder_point,
            mean_return=mean_return,
            mean_budget=mean_budget,
            mean_return_discount=mean_return_discount,
            mean_budget_discount=mean_budget_discount,
            num_episodes=num_episodes)

    def evaluate_single_ep(self, basestock_level):

        state = self.env_valid.reset()
        episode_steps = 0
        episode_return = 0
        episode_return_discount = 0
        episode_budget_discount = 0
        done = False

        while not done:
            action = self.policy(basestock_level, state)
            next_state, reward, done, info = self.env_valid.step(action)
            episode_return += reward
            episode_return_discount += reward * self.gamma ** episode_steps
            episode_budget_discount += info[-1][self.config.env.budget] * self.gamma ** episode_steps
            episode_steps += 1
            state = next_state

        self.env_valid.record.to_csv(os.path.join(self.record_dir, 'result_detail.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'default.yaml'))
    parser.add_argument('--method', type=str, default='basestock')
    args = parser.parse_args()
    agent = ClassicalAgent(args)
    agent.evaluate()