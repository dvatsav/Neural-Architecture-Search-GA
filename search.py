import sys
import numpy as np
import logging

from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2

from nasnet import NeuralArchitectureSearch
from utils import *
from env import *

import argparse

parser = argparse.ArgumentParser(description='Neural Architecture Search')
parser.add_argument("-m", "--max_blocks", help="max basic blocks", action="store_true", type=int, default=3)
parser.add_argument("-c", "--max_conv", help="max conv layers per block", action="store_true", type=int, default=2)
parser.add_argument("-e", "--epochs", help="epochs", action="store_true", type=int, default=1)
parser.add_argument("-p", "--pop_size", help="population size", action="store_true", type=int, default=5)
parser.add_argument("-o", "--n_offsprings", help="population size", action="store_true", type=int, default=1)
parser.add_argument("-d", "--dataset", help="dataset mnist/cifar", action="store_true", type=str, default="mnist")


args = parser.parse_args()

np.random.seed(42)

random_seed = 2
torch.manual_seed(random_seed)

inp_size = 28
inp_channels = 1
if args.dataset == "cifar":
	inp_size = 32
	inp_channels = 3


n_obj = 2
max_blocks = args.max_blocks
max_convs_per_block = args.max_convs_per_block
activations = ['sigmoid', 'ReLU']
epochs = args.epochs
n_gen = 50
n_vars_block = 1 + 1 + 2 * max_convs_per_block
n_var = int(max_blocks * (n_vars_block)) 
lower_bound = np.zeros(n_var, dtype=np.int)
upper_bound = np.ones(n_var, dtype=np.int)
for i in range(0, n_var-1, n_vars_block):
	for j in range(i + 1 + max_convs_per_block + 1, i + 1 + 1 + 2*max_convs_per_block, 1):
		lower_bound[j] = 1
		upper_bound[j] = 7

problem = NeuralArchitectureSearch(n_var=n_var, n_obj=n_obj, lb=lower_bound,
								ub=upper_bound, max_blocks=max_blocks, 
								max_convs_per_block=max_convs_per_block,
								epochs=epochs, args=args)
algorithm = NSGA2(pop_size=args.pop_size, eliminate_duplicates=True, n_offsprings=args.n_offsprings)
result = minimize(problem, algorithm, termination=('n_gen', 5))
logging.info("Best_Genome_%s"%(str(result.X)))
best_model = problem.best_model
writer.add_graph(best_model.to(device), torch.randn(32, inp_channels, inp_size, inp_size).cuda())
writer.close()