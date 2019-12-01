import torch
import torch.nn as nn
import numpy as np

def get_round_channel(channel):
	channels = {1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024}
	return channels[channel]

def decode_genome(genome, n_var, max_convs_per_block, n_vars_block):
	blocks = []
	genome_len = n_var
	for i in range(0, genome_len, n_vars_block):
		if genome[i] == 1:
			block = {'activation':0, 'out_channels':[]}
			if genome[i+1] == 0:
				block['activation'] = nn.ReLU
			else:
				block['activation'] = nn.Sigmoid

			for j in range(i + 1 + max_convs_per_block + 1, i + 1 + 1 + 2*max_convs_per_block, 1):
				if genome[j-max_convs_per_block] == 1:
					block['out_channels'].append(get_round_channel(genome[j]))
			if len(block['out_channels']) > 0:
				blocks.append(block)
	return blocks