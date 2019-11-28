import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class BasicBlock(nn.Module):
	def __init__(self, activation, in_channels, out_channels, last=False):
		super(BasicBlock, self).__init__()
		self.activation = activation()
		self.in_channels = in_channels
		self.n_convs = len(out_channels)
		self.convs = self._make_convs(out_channels)

	def _make_convs(self, out_channels):
		layers = []
		for i in range(self.n_convs):
			layers.append(nn.Conv2d(self.in_channels, out_channels[i], 
												  kernel_size=1, bias=True))
			self.in_channels = out_channels[i]
		return nn.Sequential(*layers)

	def forward(self, x):
		for i in range(self.n_convs):
			x = self.activation(self.convs[i](x))
		return x

class NeuralNetwork(nn.Module):
	def __init__(self, blocks, in_channels, num_outputs):
		super(NeuralNetwork, self).__init__()
		self.blocks = blocks
		self.n_layers = len(self.blocks)
		self.layers, out_channels = self._make_layers(self.blocks, in_channels)
		self.fc = nn.Linear(out_channels * 14 * 14, num_outputs)
		self.avgpool = nn.AvgPool2d(2)

	def _make_layers(self, blocks, in_channels):
		layers = []
		for i in range(self.n_layers):
			block = blocks[i]
			layers.append(BasicBlock(block['activation'], in_channels, 
								   block['out_channels']))
			in_channels=block['out_channels'][len(block['out_channels'])-1]
		return nn.Sequential(*layers), in_channels

	def forward(self, x):
		for i in range(self.n_layers):
			x = self.layers[i](x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x