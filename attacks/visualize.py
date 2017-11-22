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

use_cuda = torch.cuda.is_available()

attacker = attacks.DCGAN(train_adv=False)
attacker.load('gan_generator.pth')

trainloader, testloader = example.load_cifar()

for data, labels in testloader:
	inputs, _ = data
	inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
	adv_inputs = attacker.perturb(inputs)

	vutils.save_image(adv_inputs.data, 'images/VGG_gen_adv.png', normalize=True)
	break


