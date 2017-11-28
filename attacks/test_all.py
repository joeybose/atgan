import example
#reload(example)
import torch
import torchvision
import torchvision.utils as vutils
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
import argparse
from collections import OrderedDict

use_cuda = torch.cuda.is_available()

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

        if total >= 1000:
            break

    return correct/total, correct_adv/total


def test_all(classifier_name, path_to_classifier_weights, path_to_CATN):
    criterion = nn.CrossEntropyLoss()
    _, testloader = example.load_cifar()

    architectures = {
    'VGG16': VGG,
    'res18': resnet.ResNet18,
    'dense121': densenet.densenet_cifar,
    'alex': alexnet.AlexNet,
    'googlenet': googlenet.GoogLeNet,
    'lenet': LeNet
    }
  
    model = example.prep(architectures[classifier_name]())
    model.load_state_dict(torch.load(path_to_classifier_weights))
    attacker_fgsm = attacks.FGSM()
    test_acc, fgsm_test_adv_acc = test(model, criterion, testloader, attacker_fgsm)

    #print(test_acc)
    #print(fgsm_test_adv_acc)

    attacker_cw = attacks.CarliniWagner(verbose=False)
    test_acc, cw_test_adv_acc = test(model, criterion, testloader, attacker_cw)

   # print(test_acc)
   # print(cw_test_adv_acc)

    attacker_catn = attacks.DCGAN(train_adv=False)
    attacker_catn.load(path_to_CATN)
    
    test_acc, catn_test_adv_acc = test(model, criterion, testloader, attacker_catn)

    #print(catn_test_adv_acc)
    return test_acc, fgsm_test_adv_acc, cw_test_adv_acc, catn_test_adv_acc


if __name__ == "__main__":
    #trainloader, testloader = example.load_cifar()
#     attacker_catn = attacks.DCGAN(train_adv=False)
#     attacker_catn.load('./saved/VGG16_attacker_0.005.pth')
    parser = argparse.ArgumentParser()
    parser.add_argument("classifier_name", help="pick one of 'VGG16','res18','dense121','alex','googlenet','lenet'")
    parser.add_argument("path_to_classifier_weights", help="path to trained classifier weights, eg. saved/VGG16.pth")
    parser.add_argument("path_to_attacker_weights", help="path to CATN generator trained weights, eg. saved/VGG16_attacker_0.005.pth")
    args = parser.parse_args()
    test_acc, fgsm_test_adv_acc, cw_test_adv_acc, catn_test_adv_acc = test_all(args.classifier_name,args.path_to_classifier_weights, args.path_to_attacker_weights)
    print "Unperturbed test accuracy: ", test_acc*100.0
    print "FGSM attacked test accuracy: ", fgsm_test_adv_acc*100.0
    print "CarliniWagner attacked test accuracy: ", cw_test_adv_acc*100.0
    print "CATN attacked test accuracy: ", catn_test_adv_acc*100.0