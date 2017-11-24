import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import attacks
import numpy as np
import example
from models.vgg import VGG
from models.lenet import LeNet

use_cuda = torch.cuda.is_available()

attacker = attacks.DCGAN(train_adv=False)
attacker.load('saved/VGG16_attacker_0.005.pth')

model = VGG('VGG16')
model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.load_state_dict(torch.load('saved/VGG16.pth'))

criterion = nn.CrossEntropyLoss()
trainloader, testloader = example.load_cifar()

for inputs, labels in testloader:
	inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
	adv_inputs = attacker.perturb(inputs)
	vutils.save_image(adv_inputs.data, 'images/VGG_gen_adv.png')
	break

