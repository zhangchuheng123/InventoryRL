from pathos.multiprocessing import ProcessingPool
from classical import ClassicalAgent
from sac_discrete import SacdAgent
from munch import DefaultMunch
from functools import partial
from tqdm import trange
import numpy as np
import argparse
import yaml
import pdb
import os 


def single_run(P_plus_L, config):

    config = config.copy()

    product_lifetime = P_plus_L // 2 
    max_lead_time = P_plus_L - product_lifetime
    arrival_prob = list(np.linspace(0, 1, max_lead_time + 2)[1:])

    config.env.product_lifetime = [product_lifetime]
    config.env.max_lead_time = [max_lead_time]
    config.env.arrival_prob = arrival_prob

    config.config_name = "ppl{}_ss".format(P_plus_L)
    config.method = "ss_policy"
    agent = ClassicalAgent(config=config)
    agent.evaluate()

    config.config_name = "ppl{}_sacd_ours".format(P_plus_L)
    config.budget_estimation_episodes = 20
    agent = SacdAgent(config=config)
    agent.run()

    config.config_name = "ppl{}_sacd_bench".format(P_plus_L)
    config.budget_estimation_episodes = 1000
    agent = SacdAgent(config=config)
    agent.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'default.yaml'))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config = DefaultMunch.fromDict(config)

    single_run_partial = partial(single_run, config=config)
    pool = ProcessingPool(4)
    pool.map(single_run_partial, range(2, 14, 2))
