from __future__ import print_function
import pdb
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
nc = 3
batchSize=2048
imageSize=32
nz=100
ngf=48
ndf=48
niter=1000
lr=0.001
beta1=0.5
outf = './results_naive'

dataset = dset.CIFAR10(root='../data/', download=True,
                       transform=transforms.Compose([
                           transforms.Scale(imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                                 shuffle=True,
                                                 num_workers=int(2))
transform = transforms.Compose([transforms.ToTensor(),\
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

testset = dset.CIFAR10(root='../data/', train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=False,num_workers=2)
ngpu = int(1)
nz = int(nz)
ngf = int(ngf)
ndf = int(ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 48 x 32 x 32
            nn.Conv2d(ngf, ngf, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 3 x 32 x 32
            nn.Conv2d(ngf, nc, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
#netG.apply(weights_init)
print(netG)

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*2, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*2, 1, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 10, 1, 1, 1, bias=False),
            nn.AvgPool2d((8,8)),
            nn.Softmax()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.squeeze()


train_disc = False
cuda = True
netD = _netD(ngpu)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.CrossEntropyLoss()
#scheduler = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[150,250,350], gamma=0.1)
if cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()

def trainD():
    for epoch in range(niter):
        for i, batch in enumerate(dataloader):
            data, classes = batch[0], batch[1]

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data
            batch_size = real_cpu.size(0)
            real_cpu = real_cpu.cuda()
            #input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(real_cpu)
            inputv.requires_grad = True
            labelv = Variable(classes).cuda()
            output = netD(inputv)
            errD = criterion(output, labelv)
            errD.backward(retain_graph=True)
            optimizerD.step()
            _, predicted = torch.max(output.data, 1)
            correct = predicted.eq(labelv.data).cpu().sum()
            print('[%d/%d][%d/%d] Loss_D: %.4f | Acc: %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.data[0],100.*correct/batch_size))
    torch.save(netD.state_dict(), 'results_naive/netD_1000.pth')

def testD(model,testloader):
    correct, correct_adv, total = 0.0, 0.0, 0.0
    netG.load_state_dict(torch.load('results/netG_epoch_999.pth'))
    for data in testloader:
        inputs, labels = data
        inputs = Variable((inputs.cuda() if cuda else inputs), requires_grad=True)
        labels = Variable((labels.cuda() if cuda else labels), requires_grad=False)
        y_hat = model(inputs)
        loss = criterion(y_hat, labels)
        loss.backward()
        _, predicted = torch.max(y_hat.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum()
        fake = netG(Variable(inputs.grad.data))
        perturbed = inputs + fake
        y_fake = model(perturbed)
        _, predicted_fake = torch.max(y_fake.data, 1)
        correct_adv += predicted_fake.eq(labels.data).sum()
    print('Test Acc Acc: %.4f | Test Attacked Acc; %.4f'
          % (100.*correct/total, 100.*correct_adv/total))
# print labels
classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if train_disc:
    netD.apply(weights_init)
    print(netD)
    trainD()
else:
    transform = transforms.Compose([transforms.ToTensor(),\
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

    testset = dset.CIFAR10(root='./data', train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=2048,shuffle=False,num_workers=2)
    netD.load_state_dict(torch.load('results/netD_epoch_999.pth'))
    print(netD)
    testD(netD,testloader)
