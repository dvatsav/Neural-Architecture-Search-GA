import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
	def __init__(self, activation, in_channels, out_channels, inp_size=28):
		super(BasicBlock, self).__init__()
		self.in_channels = in_channels
		self.inp_size = inp_size
		self.n_convs = len(out_channels)
		self.convs = self._make_convs(out_channels)
		self.activation = activation()

	def _make_convs(self, out_channels):
		layers = []
		for i in range(self.n_convs):
			if self.inp_size > 6:
				layers.append(nn.Conv2d(self.in_channels, out_channels[i], 
													  kernel_size=3, bias=True))
				self.in_channels = out_channels[i]
				self.inp_size -= 2
		return nn.Sequential(*layers)

	def forward(self, x):
		for i in range(len(self.convs)):
			x = self.activation(self.convs[i](x))
		return x

class NeuralNetwork(nn.Module):
	def __init__(self, blocks, in_channels, num_outputs):
		super(NeuralNetwork, self).__init__()
		self.blocks = blocks
		self.n_layers = len(self.blocks)
		self.layers, out_channels, out_size = self._make_layers(self.blocks, in_channels)
		self.dropout = nn.Dropout2d(0.5, inplace=True)
		self.maxpool = nn.MaxPool2d(2)
		if out_size % 2 != 0:
			self.maxpool = nn.MaxPool2d(kernel=2, stride=1)
			out_size = out_size - 2 + 1
		else:
			out_size = int(((out_size-2)/2)+1)
		
		self.fc1 = nn.Linear(out_channels*out_size*out_size, 128)
		self.fc2 = nn.Linear(128, num_outputs)
		

	def _make_layers(self, blocks, in_channels, inp_size=28):
		layers = []
		for i in range(self.n_layers):
			if inp_size > 6:
				block = blocks[i]
				bblock = BasicBlock(block['activation'], in_channels, block['out_channels'], inp_size)
				layers.append(bblock)
				inp_size = bblock.inp_size
				in_channels=block['out_channels'][len(block['out_channels'])-1]
		return nn.Sequential(*layers), in_channels, inp_size

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)
		x = self.maxpool(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = F.softmax(x, dim=1)
		return x