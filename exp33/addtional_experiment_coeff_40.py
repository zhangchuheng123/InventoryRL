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


def single_run(init_coeff, config):

    config = config.copy()

    config.config_name = "coeff{}_sacd_ours_N100b40".format(init_coeff)
    config.seed = 8000
    config.env.lost_sale_cost = [init_coeff]
    config.env.target_budget_discount = 40

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

    pool = ProcessingPool(3)
    pool.map(single_run_partial, [0, 3, 6])
