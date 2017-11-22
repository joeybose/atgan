import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.vgg import VGG
from models.lenet import LeNet
import models.resnet as resnet
import models.densenet as densenet
import attacks
import numpy as np


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
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
	return trainloader, testloader


def train(model, optimizer, criterion, trainloader, attacker=None, num_epochs=25, freq=10):
	"""
	Train the model with the optimizer and criterion for num_epochs epochs on data trainloader.
	attacker is an object that produces adversial inputs given regular inputs.
	Return the accuracy on the normal inputs and on the perturbed inputs.

	To save time, only perturb inputs on the last epoch, at the frequency freq.
	""" 
	for epoch in range(num_epochs):
		running_loss = 0.0
		total, correct, correct_adv, total_adv  = 0.0, 0.0, 0.0, 1.0
		
		for i, data in enumerate(trainloader):
			inputs, labels = data
			inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
			labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

			y_hat = model(inputs)
			loss = criterion(y_hat, labels)			

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			_, predicted = torch.max(y_hat.data, 1)
			total += labels.size(0)
			correct += predicted.eq(labels.data).sum()

			# print statistics
			running_loss = loss.data[0]

			if attacker:
				# only perturb inputs on the last epoch, to save time
				# if (i+1) % freq == 0: #  and (epoch == num_epochs - 1):
				adv_inputs, adv_labels, num_unperturbed = attacker.attack(inputs, labels, model, optimizer)
				correct_adv += num_unperturbed
				total_adv += labels.size(0)

			if (i+1) % freq == 0:
				print '[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 2), correct/total, correct_adv/total_adv
				running_loss = 0.0

	return correct/total, correct_adv/total_adv 


def test(model, criterion, testloader, attacker):
	"""
	Test the model with the data from testloader.
	attacker is an object that produces adversial inputs given regular inputs.
	Return the accuracy on the normal inputs and the unperturbed inputs.
	"""
	correct, correct_adv, total = 0.0, 0.0, 0.0

	for data in testloader:
		inputs, labels = data
		inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
		labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

		y_hat = model(inputs)
		loss = criterion(y_hat, labels)
		loss.backward()

		predicted = torch.max(y_hat.data, 1)[1]
		correct += predicted.eq(labels.data).sum() 

		adv_inputs, adv_labels, num_unperturbed = attacker.attack(inputs, labels, model)
		correct_adv += num_unperturbed

		total += labels.size(0)

	return correct/total, correct_adv/total


if __name__ == "__main__":
	trainloader, testloader = load_cifar()
	model = VGG('VGG16')
	model2 = resnet.ResNet50() # densenet.densenet_cifar() # resnet.ResNet50() # LeNet() 

	if use_cuda:
		model.cuda()
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

		model2 = model2.cuda()
		model2 = torch.nn.DataParallel(model2, device_ids=range(torch.cuda.device_count()))

		cudnn.benchmark = True 

	# use default hyperparams for best results!
	# attacker = attacks.FGSM()
	# attacker = attacks.CarliniWagner(verbose=True)
	attacker = attacks.DCGAN(train_adv=False)

	criterion = nn.CrossEntropyLoss()
	
	# train first model adversarially
	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
	train_acc, train_adv_acc = train(model, optimizer, criterion, trainloader, attacker, num_epochs=50)
	test_acc, test_adv_acc = test(model, criterion, testloader, attacker)
	attacker.save('gan_generator.pth')

	"""
	# train second model normally
	# attacker.load('gan_generator.pth')
	optimizer2 = optim.SGD(model2.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
	train_acc, train_adv_acc = train(model2, optimizer2, criterion, trainloader, num_epochs=50)
	test_acc, test_adv_acc = test(model2, criterion, testloader, attacker)

	print 'Train accuracy of the network on the 10000 test images:', train_acc, train_adv_acc
        print 'Test accuracy of the network on the 10000 test images:', test_acc, test_adv_acc
	"""
