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
from os.path import join as joindir
from munch import DefaultMunch
from tqdm import trange
import wandb
import pickle
import random
import time
import math
import pdb

from utils import update_params, disable_gradients, initialize_weights_he
from utils import Flatten, RunningMeanStats, ZFilter, NDStandardScaler
from env import make_env, VectorEnv
from memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MLPBase(BaseNetwork):

    def __init__(self, num_channels, hidden=32, layers=3):

        super(MLPBase, self).__init__()

        if layers <= 1:
            self.net = nn.Sequential(
                nn.Linear(num_channels, hidden),
                nn.ReLU(),
            ).apply(initialize_weights_he)
        elif layers == 2:
            self.net = nn.Sequential(
                nn.Linear(num_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ).apply(initialize_weights_he)
        elif layers == 3:
            self.net = nn.Sequential(
                nn.Linear(num_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ).apply(initialize_weights_he)
        elif layers == 4:
            self.net = nn.Sequential(
                nn.Linear(num_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)


class CNNBase(BaseNetwork):

    def __init__(self, num_channels, hidden=32):

        super(CNNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, hidden)
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions,
                 use_dueling=False, hidden=32, layers=3, encoder='CNN'):

        super(BaseNetwork, self).__init__()

        use_l1Q = False 
        if use_l1Q:

            self.encoder = nn.Identity()
            self.head = nn.Linear(num_channels, num_actions)

        else:

            if encoder == 'MLP':
                self.encoder = MLPBase(num_channels, hidden, layers)
            elif encoder == 'CNN':
                self.encoder = CNNBase(num_channels, hidden)

            if not use_dueling:
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, num_actions))
            else:
                self.a_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, num_actions))
                self.v_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, 1))

        self.use_dueling = use_dueling

    def forward(self, states):

        states = self.encoder(states)

        if not self.use_dueling:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions,
                 use_dueling=False, hidden=32, layers=3, encoder='CNN'):

        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(num_channels, num_actions, 
            use_dueling, hidden, layers, encoder)
        self.Q2 = QNetwork(num_channels, num_actions, 
            use_dueling, hidden, layers, encoder)

    def forward(self, states):

        q1 = self.Q1(states)
        q2 = self.Q2(states)

        return q1, q2


class CategoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, 
        hidden=32, layers=3, encoder='MLP'):

        super(CategoricalPolicy, self).__init__()

        if layers <= 0:
            self.encoder = nn.Identity()
        elif encoder == 'MLP':
            self.encoder = MLPBase(num_channels, hidden, layers)
        elif encoder == 'CNN':
            self.encoder = CNNBase(num_channels, hidden)

        if layers <= -1:
            self.head = nn.Linear(num_channels, num_actions)
        elif layers <= 0:
            self.head = nn.Sequential(
                nn.Linear(num_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_actions)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, num_actions)
            )

    def act(self, states):
        
        states = self.encoder(states)
        action_logits = self.head(states)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions.cpu().numpy()

    def sample(self, states):

        states = self.encoder(states)
        action_logits = self.head(states)
        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample().view(-1, 1).cpu().numpy()

        return actions, action_dist


class BaseAgent(ABC):

    def __init__(self, args=None, config=None):

        super().__init__()

        if args is not None:
            # Read config
            with open(args.config) as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            self.config = DefaultMunch.fromDict(config)
            self.config_name = args.config.split('/')[-1].rstrip('.yaml')
        elif config is not None:
            self.config = config
            self.config_name = config.config_name
        else:
            raise NotImplementedError

        # Set up
        self.seed = self.config.basic.seed
        self.set_seed(self.seed)
        self.set_backend()

        # Directory
        self.exp_name = self.config_name
        self.exp_time = datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = os.path.join('logs', 
            '{name}-{seed}-{time}'.format(name=self.exp_name, 
            seed=self.seed, time=self.exp_time))
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        self.record_dir = os.path.join(self.log_dir, 'record')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)

        # Constants
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = self.config.algo.num_steps
        self.batch_size = self.config.algo.batch_size
        self.gamma = self.config.algo.gamma
        self.gamma_n = self.config.algo.gamma ** self.config.algo.multi_step
        self.start_steps = self.config.algo.start_steps
        self.update_interval = self.config.algo.update_interval
        self.target_update_interval = self.config.algo.target_update_interval
        self.use_per = self.config.algo.use_per
        self.evaluate_steps = self.config.algo.evaluate_steps
        self.log_interval = self.config.algo.log_interval
        self.eval_interval = self.config.algo.eval_interval
        self.num_parallel_envs = self.config.env.num_parallel_envs
        self.target_entropy_ratio = self.config.algo.target_entropy_ratio
        self.verbose = self.config.basic.verbose

        # Create environments
        self.env_train = make_env(self.config.env.copy(), etype='train')
        self.env_valid = make_env(self.config.env.copy(), etype='valid')
        self.env_valid.set_coeff(0)

        if torch.cuda.is_available():
            self.device = self.config.basic.device
        else:
            self.device = 'cpu'

        # LazyMemory efficiently stores FrameStacked states.
        if self.config.algo.use_per:
            beta_steps = (self.config.algo.num_steps - self.config.algo.start_steps) \
                / self.config.algo.update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=self.config.algo.memory_size,
                state_shape=self.env_train.observation_space.shape,
                device=self.device, 
                gamma=self.config.algo.gamma, 
                state_dtype=self.config.env.state_dtype,
                multi_step=self.config.algo.multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=self.config.algo.memory_size,
                state_shape=self.env_train.observation_space.shape,
                device=self.device, 
                gamma=self.config.algo.gamma, 
                state_dtype=self.config.env.state_dtype,
                multi_step=self.config.algo.multi_step)

        self.writer = SummaryWriter(log_dir=self.summary_dir)

    @staticmethod
    def set_seed(seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def set_backend(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.config.accuracy == 'float32':
            torch.set_default_dtype(torch.float32)
        elif self.config.accuracy == 'float64':
            torch.set_default_dtype(torch.float32)

    @abstractmethod
    def normalize_phase(self):
        pass

    def run(self):
        if hasattr(self, 'normalize_phase') and \
            (self.config.algo.normalize_state or \
            self.config.algo.zscore_reward):

            self.normalize_phase()

        self.train()

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    def _normalize_reward(self, reward):

        if self.config.algo.clip_reward:
            reward = np.clip(reward, -1.0, 1.0)
        if self.config.algo.zscore_reward:
            if self.num_parallel_envs > 1:
                reward = np.array([self.reward_filter(r) for r in reward])
            else:
                reward = self.reward_filter(reward)

        return reward

    def _basestock_wrapper(self, action, state_sum):
        action = action - state_sum
        action = np.clip(action, 0, self.config.env.action_space_size - 1)
        return action

    def _normalize_state(self, state):
        if self.config.algo.normalize_state:
            if len(state.shape) == 2:
                state = self.state_scaler.transform(state)
            else:
                state = self.state_scaler.transform(state[None, ...])
        return state

    def train(self):

        if self.num_parallel_envs > 1:
            assert not self.config.algo.use_basestock_wrapper

        state = self.env_train.reset()
        if self.config.algo.use_const_state:
            self.const_state = state
        state_sum = np.sum(state)
        state = self._normalize_state(state)

        episode_return = 0
        episode_budget = 0
        episode_return_discount = 0
        episode_budget_discount = 0
        episode_steps = 0
        episode_actions = []

        budget_estimator = []

        for steps in trange(self.num_steps, desc='Train'):

            # Environment interaction
            if self.config.algo.use_const_state:
                action = self.explore(self.const_state)
            else:
                action = self.explore(state)
            if self.config.algo.use_basestock_wrapper:
                action = self._basestock_wrapper(action, state_sum)
            episode_actions.append(action)
            next_state, reward, done, info = self.env_train.step(action)
            next_state_sum = np.sum(next_state)
            next_state = self._normalize_state(next_state)

            # Count steps
            episode_return += np.sum(reward)
            episode_budget += np.sum(info[-1][self.config.env.budget])
            episode_return_discount += np.sum(reward) \
                * (self.gamma ** (episode_steps // self.num_parallel_envs))
            episode_budget_discount += info[-1][self.config.env.budget] \
                * (self.gamma ** (episode_steps // self.num_parallel_envs))
            episode_steps += self.num_parallel_envs
            self.steps += self.num_parallel_envs

            # Normalization for reward
            reward = self._normalize_reward(reward)

            # Accumulate memory
            if self.num_parallel_envs > 1:
                for i in range(self.num_parallel_envs):
                    self.memory.append(state[i], action[i], reward[i], next_state[i], done)
            else:
                self.memory.append(state, action, reward, next_state, done)

            # Next state
            if done:
                state = self.env_train.reset()
                state_sum = np.sum(state)
                state = self._normalize_state(state)
            else:
                state = next_state
                state_sum = next_state_sum

            # Update budget coeff
            if len(budget_estimator) >= self.config.algo.budget_estimation_episodes \
                and self.steps >= self.start_steps:

                estimated_budget = np.mean(budget_estimator)
                coeff = self.env_train.get_coeff()
                coeff = max(coeff + (estimated_budget - self.config.env.target_budget_discount) \
                     * self.config.algo.budget_dual_stepsize, 0)
                self.env_train.set_coeff(coeff)
                budget_estimator = []

            # Episodic statistics
            if done:

                # For environments with asynchronous termination, 
                #   this statistics is not accurate.

                self.episodes += self.num_parallel_envs 
                episode_steps /= self.num_parallel_envs
                episode_budget /= self.num_parallel_envs
                episode_return /= self.num_parallel_envs
                episode_return_discount /= self.num_parallel_envs
                episode_budget_discount /= self.num_parallel_envs
                self.writer.add_scalar('train/ep_return', episode_return, self.steps)
                self.writer.add_scalar('train/ep_budget', episode_budget, self.steps)
                self.writer.add_scalar('train/ep_length', episode_steps, self.steps)
                self.writer.add_scalar('train/ep_return_discount', 
                    episode_return_discount, self.steps)
                self.writer.add_scalar('train/ep_budget_discount', 
                    episode_budget_discount, self.steps)
                self.writer.add_scalar('train/budget_coeff', self.env_train.get_coeff(), self.steps)

                log_dict = {
                    'train/steps': self.steps,
                    'train/ep_return': episode_return,
                    'train/ep_budget': episode_budget,
                    'train/ep_length': episode_steps,
                    'train/ep_return_discount': episode_return_discount, 
                    'train/ep_budget_discount': episode_budget_discount,
                    'train/budget_coeff': self.env_train.get_coeff(),
                }
                wandb.log(log_dict)


                if self.verbose >= 2:
                    print('Episode: {:<4}  Episode steps: {:<4} Return: {:<5.1f}'.format(
                        self.episodes, episode_steps, episode_return))
                budget_estimator.append(episode_budget_discount)
                episode_return = 0
                episode_budget = 0
                episode_steps = 0
                episode_return_discount = 0
                episode_budget_discount = 0

                # self.writer.add_histogram('train/ep_actions', np.array(episode_actions), self.steps)
                episode_actions = []

            if self.steps % self.update_interval == 0 and self.steps >= self.start_steps:
                self.learn()

            if self.steps % self.target_update_interval == 0 and self.steps >= self.start_steps:
                self.update_target()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1.0

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.steps)

            log_dict = {
                'train/steps': self.steps,
                'loss/Q1': q1_loss.detach().item(),
                'loss/Q2': q2_loss.detach().item(),
                'loss/policy': policy_loss.detach().item(),
                'loss/alpha': entropy_loss.detach().item(),
                'stats/alpha': self.alpha.detach().item(), 
                'stats/mean_Q1': mean_q1,
                'stats/mean_Q2': mean_q2,
                'stats/entropy': entropies.detach().mean().item(),
            }
            wandb.log(log_dict)

    def evaluate(self):

        num_episodes = 0
        num_steps = 0
        total_return = 0
        total_budget = 0
        total_return_discount = 0
        total_budget_discount = 0
        total_actions = []

        while True:
            state = self.env_valid.reset()
            state_sum = np.sum(state)
            state = self._normalize_state(state)
            episode_steps = 0
            episode_return = 0
            episode_budget = 0
            episode_return_discount = 0
            episode_budget_discount = 0
            done = False
            while not done:
                if self.config.algo.use_const_state:
                    action = self.exploit(self.const_state)
                else:
                    action = self.exploit(state)
                if self.config.algo.use_basestock_wrapper:
                    action = self._basestock_wrapper(action, state_sum)
                total_actions.append(action)
                next_state, reward, done, info = self.env_valid.step(action)
                next_state_sum = np.sum(next_state)
                next_state = self._normalize_state(next_state)
                num_steps += 1
                episode_return += np.sum(reward)
                episode_budget += info[-1][self.config.env.budget]
                episode_return_discount += np.sum(reward) * self.gamma ** episode_steps
                episode_budget_discount += info[-1][self.config.env.budget] * self.gamma ** episode_steps
                episode_steps += 1
                state = next_state
                state_sum = next_state_sum

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

        if mean_return > self.best_eval_score:
            self.env_valid.record.to_csv(os.path.join(self.record_dir, 
                'record_{:010d}.csv'.format(self.steps)))
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        self.writer.add_scalar('eval/return', mean_return, self.steps)
        self.writer.add_scalar('eval/budget', mean_budget, self.steps)
        self.writer.add_scalar('eval/return_discount', mean_return_discount, self.steps)
        self.writer.add_scalar('eval/budget_discount', mean_budget_discount, self.steps)
        # self.writer.add_histogram('eval/actions', np.array(total_actions), self.steps)

        log_dict = {
            'eval/steps': self.steps,
            'eval/return': mean_return,
            'eval/budget': mean_budget,
            'eval/return_discount': mean_return_discount, 
            'eval/budget_discount': mean_budget_discount,
        }
        wandb.log(log_dict)

        if self.verbose >= 1:
            print('-' * 60)
            print(f'Num steps: {self.steps:<5}  '
                  f'return: {mean_return:<5.1f}')
            print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

    def __del__(self):
        if hasattr(self, 'env_train'):
            self.env_train.close()
        if hasattr(self, 'env_valid'):
            self.env_valid.close()
        if hasattr(self, 'writer'):
            self.writer.close()


class SacdAgent(BaseAgent):

    def __init__(self, args=None, config=None):

        super().__init__(args=args, config=config)

        # Define networks.
        self.policy = CategoricalPolicy(
            self.env_train.observation_space.shape[0], 
            self.env_train.action_space.n,
            hidden=self.config.algo.hidden_size,
            layers=self.config.algo.num_layers,
            encoder=self.config.env.encoder).to(device=self.device)
        self.online_critic = TwinnedQNetwork(
            self.env_train.observation_space.shape[0], 
            self.env_train.action_space.n,
            use_dueling=self.config.algo.use_dueling,
            hidden=self.config.algo.hidden_size,
            layers=self.config.algo.num_layers,
            encoder=self.config.env.encoder).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env_train.observation_space.shape[0], 
            self.env_train.action_space.n,
            use_dueling=self.config.algo.use_dueling,
            hidden=self.config.algo.hidden_size,
            layers=self.config.algo.num_layers,
            encoder=self.config.env.encoder).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), 
            lr=self.config.algo.lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), 
            lr=self.config.algo.lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), 
            lr=self.config.algo.lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            - np.log(1.0 / self.env_train.action_space.n) \
            * self.target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], 
            lr=self.config.algo.lr)

        self.state_scaler = NDStandardScaler()
        self.reward_filter = ZFilter()

        wandb.login(
            host="https://microsoft-research.wandb.io",
            key="local-fb872608a0ec758c86ab35d24eec1373fe2d9313",
        )
        wandb.init(
            project="inventory_rl", 
            name=self.config_name,
        )

    def normalize_phase(self):
        acc_states = []

        state = self.env_train.reset()
        states = state

        for _ in trange(self.config.algo.normalization_steps // self.num_parallel_envs, \
            desc='normalize_phase'): 

            # random sample action to collect some samples
            if self.num_parallel_envs == 1:
                action = np.random.choice(self.env_train.action_space.n)
                state, reward, done, _ = self.env_train.step(action)
                self.reward_filter.update(reward)
                acc_states.append(state)
                if done:
                    self.env_train.reset()
            else:
                actions = np.random.choice(self.env_train.action_space.n, 
                    size=self.num_parallel_envs)
                states, rewards, done, _ = self.env_train.step(actions)
                for reward in rewards:
                    self.reward_filter.update(reward)
                acc_states.extend(states)
                if done:
                    self.env_train.reset()

        self.state_scaler.fit(acc_states)

    def _to_tensor(self, tensor, dtype=None, device='cpu'):
        if dtype is None: 
            if self.config.basic.accuracy == 'float64':
                dtype = torch.float64
            elif self.config.basic.accuracy == 'float32':
                dtype = torch.float32

        return torch.tensor(tensor, dtype=dtype, device=device)

    def explore(self, states):
        # Act with randomness

        if len(states.shape) == 1:
            states = states[None, ...]

        if self.config.env.state_dtype == 'uint8':
            states = torch.ByteTensor(states).to(self.device).float() / 255.
        else:
            states = self._to_tensor(states, device=self.device)

        with torch.no_grad():
            actions, _ = self.policy.sample(states)

        if self.num_parallel_envs == 1:
            return actions.flatten()[0]
        else:
            return actions.flatten()

    def exploit(self, states):
        # Act without randomness

        if len(states.shape) == 1:
            states = states[None, ...]

        if self.config.env.state_dtype == 'uint8':
            states = torch.ByteTensor(states).to(self.device).float() / 255.
        else:
            states = self._to_tensor(states, device=self.device)

        with torch.no_grad():
            actions = self.policy.act(states)

        if self.num_parallel_envs == 1:
            return actions.flatten()[0]
        else:
            return actions.flatten()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):

        with torch.no_grad():
            _, action_dists = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_dists.probs * (torch.min(next_q1, next_q2) \
                - self.alpha * action_dists.logits)).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        if self.config.algo.no_term:
            return rewards + self.gamma_n * next_q
        else:
            return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        # TODO: mean of q1 and q2 errors
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_dists = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = action_dists.entropy().view(-1, 1)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_dists.probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        target_entropy = self.target_entropy * \
            (1 - (self.steps / self.num_parallel_envs / self.num_steps))
        self.writer.add_scalar('stats/target_entropy', target_entropy, self.steps)
        entropy_loss = - torch.mean(
            self.log_alpha * (target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.state_scaler, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'default.yaml'))

    args = parser.parse_args()
    agent = SacdAgent(args)
    agent.run()
