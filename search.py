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

random_seed = 2
torch.manual_seed(random_seed)

n_obj = 2
max_blocks = 3
max_convs_per_block = 2
activations = ['sigmoid', 'ReLU']
epochs = 1
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
								epochs=epochs)
algorithm = NSGA2(pop_size=5, eliminate_duplicates=True, n_offsprings=1)
result = minimize(problem, algorithm, termination=('n_gen', 5))
logging.info("Best Genome _ %s"%(str(result.X)))
performances = problem.model_performances
best_performance_model = sorted(performances.items(), key=lambda kv:kv[1]['test_accuracy'], reverse=True)[0]
print (best_performance_model)
best_model = 0
for model in best_performance_model:
	best_model = model
writer.add_graph(best_model, torch.randn(32, 1, 28, 28))
writer.close()