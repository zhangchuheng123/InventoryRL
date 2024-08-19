from pathos.multiprocessing import ProcessingPool
from classical import ClassicalAgent
from sac_discrete import SacdAgent
from munch import DefaultMunch
from functools import partial
from itertools import product
from tqdm import trange
import numpy as np
import argparse
import yaml
import pdb
import os 


def single_run(hop, config):

    holding_cost, fixed_order_cost, perish_cost = hop

    config = config.copy()

    config.env.holding_cost = [holding_cost, holding_cost]
    config.env.fixed_order_cost = [fixed_order_cost, fixed_order_cost]
    config.env.perish_cost = [perish_cost, perish_cost]

    config.config_name = f'h{holding_cost}_o{fixed_order_cost}_p{perish_cost}_ss'
    config.method = "ss_policy"
    agent = ClassicalAgent(config=config)
    agent.evaluate()

    # config.config_name = f'h{holding_cost}_o{fixed_order_cost}_p{perish_cost}_ours'
    # config.budget_estimation_episodes = 20
    # config.seed = 8000
    # agent = SacdAgent(config=config)
    # agent.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'default.yaml'))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config = DefaultMunch.fromDict(config)

    exp_list = product([0.5, 1.0, 1.5], [2, 3, 4], [8, 10, 12])

    single_run_partial = partial(single_run, config=config)
    pool = ProcessingPool(4)
    pool.map(single_run_partial, exp_list)
