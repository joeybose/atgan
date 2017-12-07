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
import models.alexnet as alexnet
import models.googlenet as googlenet
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
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
	return trainloader, testloader


def train(model, optimizer, criterion, trainloader, architecture, attacker=None, num_epochs=50, freq=10, early_stopping=True):
	"""
	Train the model with the optimizer and criterion for num_epochs epochs on data trainloader.
	attacker is an object that produces adversial inputs given regular inputs.
	Return the accuracy on the normal inputs and on the perturbed inputs.

	To save time, only perturb inputs on the last epoch, at the frequency freq.
	"""
	for epoch in range(num_epochs):
		running_loss = 0.0
		total, correct, correct_adv, total_adv  = 0.0, 0.0, 0.0, 1.0
                early_stop_param = 0.05

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
			running_loss += loss.data[0]

			if attacker:
				# only perturb inputs on the last epoch, to save time
				# if (i+1) % freq == 0: #  and (epoch == num_epochs - 1):
				adv_inputs, adv_labels, num_unperturbed = attacker.attack(inputs, labels, model, optimizer)
				correct_adv += num_unperturbed
				total_adv += labels.size(0)

			if (i+1) % freq == 0:
                            print '[%s: %d, %5d] loss: %.4f' % (architecture,epoch + 1, i + 1, running_loss / (i+1)),\
                                    correct/total, correct_adv/total_adv

		if early_stopping:
			if running_loss / (i+1) < early_stop_param:
				print("Early Stopping !!!!!!!!!!")
				break

		running_loss = 0.0

	return correct/total, correct_adv/total_adv


def test(model, criterion, testloader, attacker, name):
	"""
	Test the model with the data from testloader.
	attacker is an object that produces adversial inputs given regular inputs.
	Return the accuracy on the normal inputs and the unperturbed inputs.
	"""
	correct, correct_adv, total = 0.0, 0.0, 0.0
        epsilon = 1.0
	for data in testloader:
		inputs, labels = data
		inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
		labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

		y_hat = model(inputs)
		loss = criterion(y_hat, labels)
		loss.backward()

		predicted = torch.max(y_hat.data, 1)[1]
		correct += predicted.eq(labels.data).sum()

		_, adv_labels, num_unperturbed = attacker.attack(inputs, labels, model)
	        # adv_inputs  = attacker.perturb(inputs,epsilon=epsilon)
		correct_adv += num_unperturbed

		total += labels.size(0)
		print "\n", correct_adv/total

        # fake = adv_inputs
        # samples_name = 'images/'+name+'_samples.png'
        # vutils.save_image(fake.data,samples_name)

        print('Test Acc Acc: %.4f | Test Attacked Acc; %.4f' % (100.*correct/total, 100.*correct_adv/total))
	return correct/total, correct_adv/total

def prep(model):
	if model and use_cuda:
		model.cuda()
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True

	return model


if __name__ == "__main__":
	trainloader, testloader = load_cifar()
	criterion = nn.CrossEntropyLoss()

	architectures = [
#		(VGG, 'VGG16', 100),
#		(resnet.ResNet18, 'res18', 25),
#		(densenet.densenet_cifar, 'dense121', 500),
#		(alexnet.AlexNet, 'alex', 500),
#		(googlenet.GoogLeNet, 'googlenet', 500),
		(LeNet, 'lenet', 500)
	]

	do_train = True

	"""
	for init_func, name, epochs in architectures:
		for tr_adv in [False, True][:1]:
			print name, tr_adv
			model = prep(init_func())
			attacker = attacks.DCGAN(train_adv=tr_adv)

                     	if do_train:
			    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                            train_acc, train_adv_acc = train(model, optimizer, criterion, trainloader, name, attacker, num_epochs=epochs)
			    test_acc, test_adv_acc = test(model, criterion,testloader, attacker, name)

                        # pdb.set_trace()
			suffix = '_AT' if tr_adv else ''
			attacker.save('saved/{0}{1}_attacker_0.01.pth'.format(name, suffix))
			torch.save(model.state_dict(), 'saved/{0}{1}.pth'.format(name, suffix))
	"""

	model = prep(VGG('VGG16'))
	model.load_state_dict(torch.load('saved/VGG16.pth'))

	# use default hyperparams for best results!
	# attacker = attacks.FGSM()
	attacker = attacks.CarliniWagner(verbose=True)
	# attacker = attacks.DCGAN(train_adv=False)

	criterion = nn.CrossEntropyLoss()

	# train second model normally
	# attacker.load('VGG_attack_0.005.pth')

	test_acc, test_adv_acc = test(model, criterion, testloader, attacker, 'VGG16')
	print 'Test accuracy of the network on the 10000 test images:', test_acc, test_adv_acc
