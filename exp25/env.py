import numpy as np
import pandas as pd
from gym import spaces
from munch import DefaultMunch
from concurrent import futures
import argparse
import yaml
import time
import pdb
import os

class MultipleSKUEnv(object):
    
    def __init__(self, 
        parallel = True,
        num_product = 2,                    # I
        product_lifetime = [4, 4],          # P
        max_lead_time = [3, 3],             # L
        arrival_prob = [1, .75, .50, .25],  # p: from old to new
        max_demand = [10, 10],
        holding_cost = [0.5, 1.5],          # h
        lost_sale_cost = [3, 3],            # l
        fixed_order_cost = [1, 3],          # f
        perish_cost = [30, 30],            # another p
        profit = [0, 0],           
        time_limit = 125,
        action_space = spaces.Discrete(10),
        demand_func = lambda: np.random.randint(3, size=(1,)),
        budget = 'lost_quantity'):

        self.budget = budget
    
        self.env = InventoryEnv(
            parallel=parallel,
            num_product=num_product, 
            product_lifetime=product_lifetime,
            max_lead_time=max_lead_time,
            arrival_prob=arrival_prob,
            max_demand=max_demand,
            holding_cost=holding_cost,
            lost_sale_cost=lost_sale_cost,
            fixed_order_cost=fixed_order_cost,
            perish_cost=perish_cost,
            profit=profit,
            time_limit=time_limit,
            action_space=action_space,
            demand_func=demand_func
        )
        self._record = []

    def seed(self, s):
        np.random.seed(s)

    def close(self):
        pass 
        
    def reset(self):
        self._record = []
        return self.env.reset()
    
    def step(self, action):
        states, rewards, done, info = self.env.step(action)
        self._record.extend(info[:-1])
        return states, rewards, done, info
    
    def get_coeff(self):
        if self.budget == 'lost_quantity':
            return self.env.lost_sale_cost[0]
        elif self.budget == 'perished_quantity':
            return self.env.perish_cost[0]
        else:
            raise NotImplementedError

    def set_coeff(self, value):
        if self.budget == 'lost_quantity':
            self.env.lost_sale_cost = \
                [value for _ in range(len(self.env.lost_sale_cost))]
        elif self.budget == 'perished_quantity':
            self.env.perish_cost = \
                [value for _ in range(len(self.env.perish_cost))]
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        dim = self.env.product_lifetime[0] + self.env.max_lead_time[0]
        return spaces.Box(np.zeros((dim, )), np.inf * np.ones((dim, )), dtype=np.float32)

    @property
    def record(self):
        return pd.DataFrame(self._record)


class SingleSKUEnv(object):
    
    def __init__(self, 
        num_product = 1,                    # I
        product_lifetime = [4],             # P
        max_lead_time = [3],                # L
        arrival_prob = [1, .75, .50, .25],  # p: from old to new
        max_demand = [10],
        holding_cost = [2],                 # h
        lost_sale_cost = [10],              # l
        fixed_order_cost = [1],             # f
        perish_cost = [15],                 # another p
        profit = [5],           
        time_limit = 200,
        action_space = spaces.Discrete(10),
        demand_func = lambda: np.random.randint(3, size=(1,)),
        budget = 'lost_quantity'):

        self.budget = budget
    
        self.env = InventoryEnv(
            num_product=num_product, 
            product_lifetime=product_lifetime,
            max_lead_time=max_lead_time,
            arrival_prob=arrival_prob,
            max_demand=max_demand,
            holding_cost=holding_cost,
            lost_sale_cost=lost_sale_cost,
            fixed_order_cost=fixed_order_cost,
            perish_cost=perish_cost,
            profit=profit,
            time_limit=time_limit,
            action_space=action_space,
            demand_func=demand_func
        )
        self._record = []

    def seed(self, s):
        np.random.seed(s)

    def close(self):
        pass 
        
    def reset(self):
        self._record = []
        return self.env.reset()[0, :]
    
    def step(self, action):
        states, rewards, done, info = self.env.step(np.array([action]))
        self._record.append(info[0])
        return states[0, :], rewards[0], done, info
    
    def get_coeff(self):
        if self.budget == 'lost_quantity':
            return self.env.lost_sale_cost[0]
        elif self.budget == 'perished_quantity':
            return self.env.perish_cost[0]
        else:
            raise NotImplementedError

    def set_coeff(self, value):
        if self.budget == 'lost_quantity':
            self.env.lost_sale_cost[0] = value
        elif self.budget == 'perished_quantity':
            self.env.perish_cost[0] = value
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        dim = self.env.product_lifetime[0] + self.env.max_lead_time[0]
        return spaces.Box(np.zeros((dim, )), np.inf * np.ones((dim, )), dtype=np.float32)

    @property
    def record(self):
        return pd.DataFrame(self._record)
    

class InventoryEnv(object):
    
    def __init__(self,
        parallel = True,
        num_product = 2,                    # I
        product_lifetime = [3, 3],          # P
        max_lead_time = [3, 3],             # L
        arrival_prob = [1, .75, .50, .25],  # p: from old to new
        max_demand = [10, 10],
        holding_cost = [3, 2],              # h
        lost_sale_cost = [10, 15],          # l
        fixed_order_cost = [3, 1],          # f
        perish_cost = [20, 15],             # another p
        profit = [5, 5],           
        time_limit = 200,
        action_space = np.arange(10),
        demand_func = lambda: np.random.randint(3, size=(2,))):

        self.num_product = num_product
        self.product_lifetime = product_lifetime
        self.max_lead_time = max_lead_time
        self.arrival_prob = arrival_prob
        self.max_demand = max_demand
        self.profit = profit
        self.holding_cost = holding_cost
        self.lost_sale_cost = lost_sale_cost
        self.fixed_order_cost = fixed_order_cost
        self.perish_cost = perish_cost
        self.time_limit = time_limit
        self.action_space = action_space
        self.demand_func = demand_func

        self.products = [None] * self.num_product

        self.parallel = parallel and (self.num_product > 1)
        if parallel:
            self.pool = futures.ProcessPoolExecutor(max_workers=50)
    
    def reset(self):

        self.counter = 0

        for i in range(self.num_product):
            self.products[i] = \
                Product(self.product_lifetime[i], self.max_lead_time[i], self.arrival_prob)
        states = [self.products[i].get_state() for i in range(self.num_product)]
        return np.array(states)

    def step(self, action):
        if self.parallel:
            return self.step_parallel(action)
        else:
            return self.step_sequential(action)

    def step_sequential(self, action):

        # t = time.process_time()

        self.counter += 1

        demands = self.demand_func()
        rewards = np.zeros(self.num_product)
        states = []
        info = []
        perished_quantity = 0
        lost_quantity = 0
        for i in range(self.num_product):
            info.append(self.products[i].step(action[i], demands[i]))
            info[-1].update({'SKUno': i})
            # rewards[i] += self.profit[i] * info[-1]['sold_quantity']
            rewards[i] -= self.holding_cost[i] * info[-1]['total_inventory']
            rewards[i] -= self.lost_sale_cost[i] * info[-1]['lost_sales']
            rewards[i] -= self.fixed_order_cost[i] * info[-1]['place_order']
            rewards[i] -= self.perish_cost[i] * info[-1]['perished_quantity']
            perished_quantity += info[-1]['perished_quantity']
            lost_quantity += info[-1]['lost_sales']
            states.append(self.products[i].get_state())

        if self.counter >= self.time_limit:
            done = True
        else:
            done = False

        info.append(dict(perished_quantity=perished_quantity, lost_quantity=lost_quantity))
        states = np.array(states)

        # print(time.process_time() - t)
        return states, rewards, done, info
    
    def step_parallel(self, action):

        # t = time.process_time()

        self.counter += 1

        demands = self.demand_func()
        rewards = []
        states = []
        infos = []
        perished_quantity = 0
        lost_quantity = 0

        for info, reward, state in self.pool.map(_update_single, 
            self.products, action, demands, np.arange(self.num_product), 
            self.holding_cost, self.lost_sale_cost, 
            self.fixed_order_cost, self.perish_cost):

            infos.append(info)
            perished_quantity += info['perished_quantity']
            lost_quantity += info['lost_sales']
            rewards.append(reward)
            states.append(state)
            rewards.append(reward)

        states = np.array(states)
        rewards = np.array(rewards)

        if self.counter >= self.time_limit:
            done = True
        else:
            done = False

        infos.append(dict(perished_quantity=perished_quantity, lost_quantity=lost_quantity))

        # print(time.process_time() - t)
        return states, rewards, done, infos


class Product(object):

    def __init__(self, product_lifetime, max_lead_time, arrival_prob):

        self.product_lifetime = product_lifetime
        self.max_lead_time = max_lead_time
        # first (perishing) -- middle (fresh) -- last (today's)
        self.inventory = np.zeros(product_lifetime + 1)
        # first (old) -- middle (new) -- last (today's)
        self.pending_orders = np.zeros(max_lead_time + 1)
        self.arrival_prob = arrival_prob

    def step(self, order, demand):

        # dealing with orders
        self.pending_orders[-1] = order
        order_arrive_indicator = np.random.binomial(1, self.arrival_prob)
        self.inventory[-1] += np.sum(order_arrive_indicator * self.pending_orders)
        self.pending_orders = self.pending_orders * (1 - order_arrive_indicator)

        self.pending_orders[:-1] = self.pending_orders[1:]
        self.pending_orders[-1] = 0 

        # dealing with inventory
        lost_sales = max(demand - np.sum(self.inventory), 0)
        remaining_demand = demand
        for i in range(self.product_lifetime + 1):
            consumed_demand = min(self.inventory[i], remaining_demand)
            self.inventory[i] -= consumed_demand
            remaining_demand -= consumed_demand

        perished_quantity = self.inventory[0]
        self.inventory[:-1] = self.inventory[1:]
        self.inventory[-1] = 0

        # calculate reward term
        info = dict(
            order = order,
            demand = demand,
            sold_quantity = demand - lost_sales,
            total_inventory = np.sum(self.inventory),
            intransit_quantity = np.sum(self.pending_orders),
            lost_sales = lost_sales,
            place_order = (order > 0) * 1,
            perished_quantity = perished_quantity,
            inventory = str(self.inventory),
            pending_orders = str(self.pending_orders),
        )

        return info

    def get_state(self):

        return np.concatenate([self.inventory[:-1], self.pending_orders[:-1]])


class VectorEnv(object):
    def __init__(self, n, env_func, parallel_init=False, **kwargs):

        print('[{}] Wait for initializing environments'.format(str(datetime.datetime.now())))
        self.pool = Pool(NUM_CORES)
        if parallel_init:
            init_func = lambda x: env_func(**kwargs)
            self.envs = self.pool.map(init_func, list(range(n)))
        else:
            self.envs = tuple(env_func(**kwargs) for _ in range(n))
        self.return_state_format = 'list'
        print('[{}] Finish initializing environments'.format(str(datetime.datetime.now())))

    def set_return_state_format(self, fmt):
        self.return_state_format = fmt

    def seed(self, seeds):

        seed_func = lambda args: args[0].seed(args[1])
        self.pool.map(seed_func, list(zip(self.envs, seeds)))

    def reset(self):

        reset_func = lambda env: env.reset()
        states = self.pool.map(reset_func, self.envs)
        if self.return_state_format == 'array':
            states = np.array(states)
        return states

    def step(self, actions):

        def step_func(args):
            env, a = args
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()
            return observation, reward, done, info

        res = self.pool.map(step_func, list(zip(self.envs, actions)))
        states, rewards, dones, infos = list(zip(*res))
        if self.return_state_format == 'array':
            states = np.array(states)
        return states, rewards, dones, infos

    def __del__(self):
        self.close()

    def close(self):
        self.pool.close()


def make_env(config, etype):
    if config.name == 'SingleSKUEnv':
        # Chenyi provides this configuration
        assert config.demand_type == 'uniform'
        return SingleSKUEnv(
            num_product = config.num_product,
            product_lifetime = config.product_lifetime,       
            max_lead_time = config.max_lead_time,       
            arrival_prob = config.arrival_prob,  
            max_demand = config.max_demand,
            holding_cost = config.holding_cost,           
            lost_sale_cost = config.lost_sale_cost,         
            fixed_order_cost = config.fixed_order_cost,       
            perish_cost = config.perish_cost,
            profit = config.profit,
            time_limit = config.time_limit,
            action_space = spaces.Discrete(config.action_space_size),
            demand_func = lambda: np.random.randint(config.max_demand, size=(1,)),
            budget = config.budget
            )
    elif config.name == 'MultipleSKUEnv':
        assert config.demand_type == 'uniform'
        np.random.seed(config.seed)
        n = config.num_product
        return MultipleSKUEnv(
            parallel = config.parallel,
            num_product = config.num_product,
            product_lifetime = config.product_lifetime * n,       
            max_lead_time = config.max_lead_time * n,       
            arrival_prob = config.arrival_prob,  
            max_demand = config.max_demand * n,
            holding_cost = \
                np.random.uniform(config.holding_cost[0], 
                    config.holding_cost[1], size=n),           
            lost_sale_cost = config.lost_sale_cost * n,         
            fixed_order_cost = \
                np.random.uniform(config.fixed_order_cost[0], 
                    config.fixed_order_cost[1], size=n),    
            perish_cost = config.perish_cost * n,
            profit = config.profit * n,
            time_limit = config.time_limit,
            action_space = spaces.Discrete(config.action_space_size),
            demand_func = lambda: np.random.randint(config.max_demand, size=(n,)),
            budget = config.budget
            )
    else:
        raise NotImplementedError


def _update_single(product, action, demand, skuno,
    holding_cost, lost_sale_cost, fixed_order_cost, perish_cost):

    info = product.step(action, demand)
    info.update({'SKUno': skuno})

    reward = 0
    reward -= holding_cost * info['total_inventory']
    reward -= lost_sale_cost * info['lost_sales']
    reward -= fixed_order_cost * info['place_order']
    reward -= perish_cost * info['perished_quantity']

    state = product.get_state()

    return info, reward, state


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'default.yaml'))
    args = parser.parse_args()

    # Read config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = DefaultMunch.fromDict(config)
    env = make_env(config.env, 'train')
    print(env.reset())
    done = False
    while not done:
        _, _, done, _ = env.step(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    env.record.to_csv('multiple.csv')

