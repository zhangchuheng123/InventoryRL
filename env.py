import numpy as np
import pandas as pd
from gym import spaces


class SingleSKUEnv(object):
    
    MAX_TIME_LIMIT = 200
    
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
        action_space = spaces.Discrete(10),
        demand_func = lambda: np.random.randint(3, size=(1,))):
    
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
            action_space=action_space,
            demand_func=demand_func
        )
        self.counter = 0
        self._record = []

    def seed(self, s):
        np.random.seed(s)

    def close(self):
        pass 
        
    def reset(self):
        self.counter = 0
        self._record = []
        return self.env.reset()[0, :]
    
    def step(self, action):
        self.counter += 1
        states, rewards, done, info = self.env.step(np.array([action]))
        if self.counter >= self.MAX_TIME_LIMIT:
            done = True
        self._record.append(info[0])
        return states[0, :], rewards[0], done, info
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        dim = self.env.product_lifetime[0] + self.env.max_lead_time[0]
        return spaces.Box(np.zeros((dim, )), np.inf * np.ones((dim, ))) 

    @property
    def record(self):
        return pd.DataFrame(self._record)
    

class InventoryEnv(object):
    
    def __init__(self,
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
        self.action_space = action_space
        self.demand_func = demand_func

        self.products = [None] * self.num_product
    
    def reset(self):
        for i in range(self.num_product):
            self.products[i] = \
                Product(self.product_lifetime[i], self.max_lead_time[i], self.arrival_prob)
        states = [self.products[i].get_state() for i in range(self.num_product)]
        return np.array(states)
    
    def step(self, action):

        demands = self.demand_func()
        rewards = np.zeros(self.num_product)
        states = []
        info = []
        for i in range(self.num_product):
            info.append(self.products[i].step(action[i], demands[i]))
            # rewards[i] += self.profit[i] * info[-1]['sold_quantity']
            rewards[i] -= self.holding_cost[i] * info[-1]['total_inventory']
            rewards[i] -= self.lost_sale_cost[i] * info[-1]['lost_sales']
            rewards[i] -= self.fixed_order_cost[i] * info[-1]['place_order']
            rewards[i] -= self.perish_cost[i] * info[-1]['perished_inventory']
            states.append(self.products[i].get_state())

        done = False

        return np.array(states), rewards, done, info


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
        for i in range(self.product_lifetime):
            consumed_demand = min(self.inventory[i], remaining_demand)
            self.inventory[i] -= consumed_demand
            remaining_demand -= consumed_demand

        perished_inventory = self.inventory[0]
        self.inventory[:-1] = self.inventory[1:]
        self.inventory[-1] = 0

        # calculate reward term
        info = dict(
            order = order,
            demand = demand,
            sold_quantity = demand - lost_sales,
            total_inventory = np.sum(self.inventory),
            lost_sales = lost_sales,
            place_order = (order > 0) * 1,
            perished_inventory = perished_inventory,
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


def make_env(name, etype):
    if name == 'SingleSKUEnv':
        if etype == 'train':
            return SingleSKUEnv()
        elif etype == 'valid':
            return SingleSKUEnv()
    else:
        raise NotImplementedError