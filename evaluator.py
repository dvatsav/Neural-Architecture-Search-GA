import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from env import *

batch_size_train = 32
batch_size_test = 32
learning_rate = 0.01
momentum = 0.5
log_interval = 10

def setup_data():
	mnist_train = torchvision.datasets.MNIST('files/', train=True, download=True, transform=torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
								(0.1307,), (0.3081,))
							]))
	mnist_test = torchvision.datasets.MNIST('files/', train=False, download=True,
								transform=torchvision.transforms.Compose([
									torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize(
									(0.1307,), (0.3081,))
								]))
	
	cifar_train = torchvision.datasets.CIFAR10('files/', train=True, download=True, transform=torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
								(0.1307,), (0.3081,))
							]))

	cifar_test = torchvision.datasets.CIFAR10('files/', train=False, download=True, transform=torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
								(0.1307,), (0.3081,))
							]))
	
	cifar_train = torch.utils.data.Subset(cifar_train, list(range(10000)))
	mnist_train = torch.utils.data.Subset(mnist_train, list(range(10000)))
	
	train_loader_mnist = torch.utils.data.DataLoader(mnist_train,
								batch_size=batch_size_train, shuffle=True)

	test_loader_mnist = torch.utils.data.DataLoader(mnist_test,
								batch_size=batch_size_test, shuffle=True)

	train_loader_cifar = torch.utils.data.DataLoader(cifar_train,
								batch_size=batch_size_train, shuffle=True)

	test_loader_cifar = torch.utils.data.DataLoader(cifar_test,
								batch_size=batch_size_test, shuffle=True)

	return train_loader_mnist, test_loader_mnist, train_loader_cifar, test_loader_cifar

def train(model, epoch, optimizer, criterion, train_losses, train_counter, train_loader):
	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		data = data.to(device)
		target = target.to(device)
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
				(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
	return train_losses

def test(model, criterion, test_losses, test_counter, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data = data.to(device)
			target = target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item()
			pred = output.argmax(1, keepdim=True)
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	test_accuracy = 100. * correct / len(test_loader.dataset)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	return test_accuracy

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)



train_loader_mnist, test_loader_mnist, train_loader_cifar, test_loader_cifar = setup_data()

def evaluate(args, model, epochs, model_name):
	optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()
	
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = []

	train_loader = None
	test_loader = None
	if args.dataset == "mnist":
		train_loader = train_loader_mnist
		test_loader = test_loader_mnist
	else:
		train_loader = train_loader_cifar
		test_loader = test_loader_cifar

	for i in range(epochs):
		train_losses = train(model, i+1, optimizer, criterion, train_losses, train_counter, train_loader)
		for j in range(len(train_losses)):
			writer.add_scalar('loss/%s'%model_name, train_losses[j], j)
	accuracy = test(model, criterion, test_losses, test_counter, test_loader)
	parameters = count_parameters(model)
	
	performance = {
			'test accuracy': accuracy.item(),
			'num parameters': parameters
	}


	logging.info("%s\n%s\n\n%s\n"%(model_name, str(model), str(performance)))
	return performance

