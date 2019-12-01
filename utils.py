import torch
import torch.nn as nn
import numpy as np

def get_round_channel(channel):
	if channel >= 16 and channel < 32:
		if int(np.abs(channel - 16)) < int(np.abs(channel-32)):
			return 16
		else:
			return 32
	elif channel >= 32 and channel < 64:
		if int(np.abs(channel - 32)) < int(np.abs(channel-64)):
			return 32
		else:
			return 64
	elif channel >= 64 and channel < 128:
		if int(np.abs(channel - 64)) < int(np.abs(channel-128)):
			return 64
		else:
			return 128
	elif channel >= 128 and channel < 256:
		if int(np.abs(channel - 128)) < int(np.abs(channel-256)):
			return 128
		else:
			return 256
	elif channel >= 256 and channel < 512:
		if int(np.abs(channel - 256)) < int(np.abs(channel-512)):
			return 256
		else:
			return 512
	else:
		if int(np.abs(channel - 512)) < int(np.abs(channel-1024)):
			return 512
		else:
			return 1024

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