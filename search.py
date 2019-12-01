import sys
import numpy as np
import logging

from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2

from nasnet import NeuralArchitectureSearch
from utils import *
from env import *


np.random.seed(42)

n_obj = 2
max_blocks = 5
max_convs_per_block = 3
activations = ['sigmoid', 'ReLU']
epochs = 3
n_gen = 50
n_vars_block = 1 + 1 + 2 * max_convs_per_block
n_var = int(max_blocks * (n_vars_block)) 
lower_bound = np.zeros(n_var, dtype=np.int)
upper_bound = np.ones(n_var, dtype=np.int)
for i in range(0, n_var-1, n_vars_block):
	for j in range(i + 1 + max_convs_per_block + 1, i + 1 + 1 + 2*max_convs_per_block, 1):
		lower_bound[j] = 16
		upper_bound[j] = 1024

problem = NeuralArchitectureSearch(n_var=n_var, n_obj=n_obj, lb=lower_bound,
								ub=upper_bound, max_blocks=max_blocks, 
								max_convs_per_block=max_convs_per_block,
								epochs=epochs)
algorithm = NSGA2(pop_size=5, eliminate_duplicates=True, n_offsprings=1)
result = minimize(problem, algorithm, termination=('n_gen', 5))
writer.close()