import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.vgg import VGG
import attacks

use_cuda = torch.cuda.is_available()


def load_cifar():
	"""
	Load and normalize the training and test data for CIFAR10
	"""
	print('==> Preparing data..')
	transform_train = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
	return trainloader, testloader


def train(model, optimizer, criterion, trainloader, attacker, num_epochs=25):
	"""
	Train the model with the optimizer and criterion for num_epochs epochs on data trainloader.
	attacker is an object that produces adversial inputs given regular inputs.
	""" 
	for epoch in range(num_epochs):
		running_loss = 0.0
		total, correct, correct_adv = 0.0, 0.0, 0.0
		
		for i, data in enumerate(trainloader):
			inputs, labels = data
			inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
			labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

			optimizer.zero_grad()

			y_hat = model(inputs)
			loss = criterion(y_hat, labels)
			loss.backward()

			_, predicted = torch.max(y_hat.data, 1)
			total += labels.size(0)
			correct += predicted.eq(labels.data).sum()

			# perturb
			_, predicted = torch.max(model(attacker.attack(inputs, labels, model)).data, 1)
			correct_adv = predicted.eq(labels.data).sum() 

			# print statistics
			running_loss = loss.data[0]

			if (i+1) % 100 == 0:
				print '[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100), correct/total, correct_adv/total
				running_loss = 0.0

			optimizer.step()

	return correct/total, correct_adv/total


def test(model, criterion, testloader, attacker):
	"""
	Test the model with the data from testloader.
	attacker is an object that produces adversial inputs given regular inputs.
	"""
	correct, correct_adv, total = 0.0, 0.0, 0.0

	for data in testloader:
		inputs, labels = data
		inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
		labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

		y_hat = model(inputs)
		loss = criterion(y_hat, labels)
		loss.backward()

		_, predicted = torch.max(y_hat.data, 1)
		total += labels.size(0)
		correct += predicted.eq(labels.data).sum() 

		# perturb
		_, predicted = torch.max(model(attacker.attack(inputs, labels, model)).data, 1)
		correct_adv = predicted.eq(labels.data).sum()

	return correct/total, correct_adv/total


if __name__ == "__main__":
	trainloader, testloader = load_cifar()
	model = VGG('VGG16')

	if use_cuda:
		model.cuda()
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True 

	attacker = attacks.FGSM(epsilon=0.25)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
	train_acc, train_adv_acc = train(model, optimizer, criterion, trainloader, attacker, num_epochs=50)
	test_acc, test_adv_acc = test(model, criterion, testloader, attacker)

	print 'Train accuracy of the network on the 10000 test images:', train_acc, train_adv_acc
	print 'Test accuracy of the network on the 10000 test images:', test_acc, test_adv_acc
