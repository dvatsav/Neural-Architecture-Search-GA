from pymop.problem import Problem
import numpy as np
from utils import decode_genome
from network import NeuralNetwork
from env import *
from evaluator import evaluate

class NeuralArchitectureSearch(Problem):
	def __init__(self, args, n_var=21, n_obj=1, n_constr=0, lb=None, ub=None, 
					max_blocks=11, max_convs_per_block=2, epochs=30):
		self.lb = lb
		self.ub = ub
		super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, 
					xl=self.lb, xu=self.ub, type_var=np.int)
		self.max_blocks = max_blocks
		self.n_var = n_var
		self.max_convs_per_block = max_convs_per_block
		self.n_vars_block = 1 + 1 + 2 * max_convs_per_block
		self.epochs = epochs
		self.n_obj = n_obj
		self.model_count = 0
		self.model_performances = {}
		self.best_model = 0
		self.best_performance = -np.inf
		self.args = args

	def _evaluate(self, x, out, *args, **kwargs):
		num_genomes = x.shape[0]
		objectives = np.full((num_genomes, self.n_obj), np.nan)
		x = (np.around(x)).astype(int)
		for i in range(num_genomes):
			self.model_count += 1
			blocks = decode_genome(x[i], self.n_var, self.max_convs_per_block, self.n_vars_block)
			print (blocks)
			performance = {
				'test accuracy': 0,
				'num parameters': np.inf
			}
			if len(blocks) > 0:
				model = NeuralNetwork(blocks=blocks, in_channels=1, num_outputs=10)
				print (model)
				model = model.to(device)
				performance = evaluate(self.args, model, epochs=self.epochs, model_name="Model_%d"%self.model_count)
				self.model_performances[model] = performance
				print (performance)
				if performance['test accuracy'] > self.best_performance:
					self.best_performance = performance['test accuracy']
					self.best_model = model
			objectives[i, 0] = 100 - performance['test accuracy']
			objectives[i, 1] = performance['num parameters']

		out["F"] = objectives