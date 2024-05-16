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


def single_run(args, config):

    num_layers, hidden_size = args

    config = config.copy()

    config.algo.hidden_size = hidden_size
    config.algo.num_layers = num_layers

    config.config_name = "decQ_l{}_h{}_sacd_ours".format(num_layers, hidden_size)
    config.seed = 8000

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
    pool = ProcessingPool(5)
    # pool.map(single_run_partial, [(3, 32), (3, 64), (3, 128), (2, 64), (4, 64)])
    # pool.map(single_run_partial, [(1, 64), (2, 64), (3, 64)])
    pool.map(single_run_partial, [(3, 64), (2, 64), (1, 64), (0, 64), (-1, 64)])

    # single_run((-1, 64), config)
