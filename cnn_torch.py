# Convolutional Neural Network for classifying MNIST images
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
		self.conv2 = nn.Conv2d(16,20, kernel_size=6)
		self.fc1 = nn.Linear(320, 80)
		self.fc2 = nn.Linear(80, 10)
	
	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return F.log_softmax(x)


model = ConvNet()
model.cuda() # run on GPU
LEARNING_RATE = 0.1

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True)

def train(epoch):
	for batch_idx, (data,target) in enumerate(train_loader):
		data, target = Variable(data.cuda()), Variable(target.cuda())
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target) # cross entropy loss
		loss.backward() # backpropagate
		optimizer.step()
		if batch_idx%10000 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                	epoch, batch_idx * len(data), len(train_loader.dataset),
                	100. * batch_idx / len(train_loader), loss.data[0]))

def test():
	test_loss = 0
	correct = 0
	for data, target in test_loader:	
		data, target = Variable(data.cuda()), Variable(target.cuda())
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss = test_loss/len(test_loader.dataset)
	print "Test Accuracy : ", correct/len(test_loader.dataset)

for epoch in range(1, 10):
	train(epoch)
	test()
