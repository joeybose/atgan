import example
import time
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

        #iaf total >= 1228:
        #    break

    return correct/total, correct_adv/total


def test_all(classifier_name, path_to_classifier_weights, path_to_CATN, fgsm=False, cw=False):
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

    if (fgsm == True):
        attacker_fgsm = attacks.FGSM()
        timestart1 = time.time()
        test_acc, fgsm_test_adv_acc = test(model, criterion, testloader, attacker_fgsm)
        timeend1 = time.time()
    else:
        fgsm_test_adv_acc = None
    #print(test_acc)
    #print(fgsm_test_adv_acc)
    #print "fgsm time: ", (timeend1-timestart1)
    if (cw == True):
        attacker_cw = attacks.CarliniWagner(verbose=False)
        timestart2 = time.time()
        test_acc, cw_test_adv_acc = test(model, criterion, testloader, attacker_cw)
        timeend2 = time.time()
    else:
        cw_test_adv_acc = None

   # print(test_acc)
   # print(cw_test_adv_acc)

    attacker_catn = attacks.DCGAN(train_adv=False)
    attacker_catn.load(path_to_CATN)

    timestart3 = time.time()
    test_acc, catn_test_adv_acc = test(model, criterion, testloader, attacker_catn)
    timeend3 = time.time()
    #print(catn_test_adv_acc)
    if (fgsm):
        print "fgsm time: ", (timeend1-timestart1)
    if(cw):
        print "cw time: ", (timeend2-timestart2)
    print "gatn time: " ,(timeend3 - timestart3)
    return test_acc, fgsm_test_adv_acc, cw_test_adv_acc, catn_test_adv_acc


if __name__ == "__main__":
    #trainloader, testloader = example.load_cifar()
 #   attacker_catn = attacks.DCGAN(train_adv=False)
 #   attacker_catn.load('1_GPU_saved_models/res18_joey_attacker_0.005.pth')
 #   print("loaded")

    parser = argparse.ArgumentParser()
    parser.add_argument("classifier_name", help="pick one of 'VGG16','res18','dense121','alex','googlenet','lenet'")
    parser.add_argument("path_to_classifier_weights", help="path to trained classifier weights, eg. saved/VGG16.pth")
    parser.add_argument("path_to_attacker_weights", help="path to GATN generator trained weights, eg. saved/VGG16_attacker_0.005.pth")
    parser.add_argument("--fgsm", default=False, help="True if want to generate FGSM attacks")
    parser.add_argument("--cw", default=False, help="True if want to generate CW attacks")
    args = parser.parse_args()

    architectures = {
    'VGG16': ['1_GPU_saved_models/VGG16.pth','1_GPU_saved_models/VGG16_attacker_0.005.pth'],
    #'res16': ['1_GPU_saved_models/res18_nodrop_joey.pth','saved/res16_attacker_0.005.pth'],
    'res18':
    ['1_GPU_saved_models/res18_nodrop_joey.pth','1_GPU_saved_models/res18_nodrop_joey_attacker_0.005.pth'],
    'dense121':
    ['1_GPU_saved_models/dense121_nodrop_joey.pth','1_GPU_saved_models/dense121_nodrop_joey_attacker_0.005.pth'],
    'alex':
    ['1_GPU_saved_models/alex_nodrop_joey.pth','1_GPU_saved_models/alex_nodrop_joey_attacker_0.005.pth'],
    'googlenet':
    ['1_GPU_saved_models/googlenet_nodrop_joey.pth','1_GPU_saved_models/googlenet_nodrop_joey_attacker_0.005.pth'],
    'lenet':
    ['1_GPU_saved_models/lenet_nodrop_joey.pth','1_GPU_saved_models/lenet_nodrop_joey_attacker_0.005.pth']
    }

    architectures_AT = {
    'VGG16': ['1_GPU_saved_models/VGG16_AT.pth','1_GPU_saved_models/VGG16_AT_attacker_0.005.pth'],
    #'res16': ['1_GPU_saved_models/res18_nodrop_joey.pth','saved/res16_attacker_0.005.pth'],
    'res18':
    ['1_GPU_saved_models/res18_AT_nodrop_joey.pth','1_GPU_saved_models/res18_AT_nodrop_joey_attacker_0.005.pth'],
    'dense121':
    ['1_GPU_saved_models/dense121_AT_nodrop_joey.pth','1_GPU_saved_models/dense121_AT_nodrop_joey_attacker_0.005.pth'],
    'alex':
    ['1_GPU_saved_models/alex_AT_nodrop_joey.pth','1_GPU_saved_models/alex_AT_nodrop_joey_attacker_0.005.pth'],
    'googlenet':
    ['1_GPU_saved_models/googlenet_AT_nodrop_joey.pth','1_GPU_saved_models/googlenet_AT_nodrop_joey_attacker_0.005.pth'],
    'lenet':
    ['1_GPU_saved_models/lenet_AT_nodrop_joey.pth','1_GPU_saved_models/lenet_AT_nodrop_joey_attacker_0.005.pth']
    }

    test_acc, fgsm_test_adv_acc, cw_test_adv_acc, catn_test_adv_acc = test_all(args.classifier_name,args.path_to_classifier_weights,
        args.path_to_attacker_weights,fgsm=args.fgsm,cw=args.cw)
    print "Unperturbed test accuracy: ", test_acc * 100.0
    if(args.fgsm):
        print "FGSM attacked test accuracy: ", fgsm_test_adv_acc*100.0
    if(args.cw):
        print "CarliniWagner attacked test accuracy: ", cw_test_adv_acc*100.0
    print "GATN attacked test accuracy: ", catn_test_adv_acc*100.0
'''
    for m in architectures.keys():
        for n in architectures.keys():
            if (m==n):
                test_acc,_,_, catn_test_adv_acc = \
                test_all(n,architectures2[n][0], architectures2[m][1],
                        fgsm=False, cw=False)
                print "Classifier (AT): " +  n + ", Generator (AT): " + m
                #print "Classifier: " + m + "Adv trained"
                print "Unperturbed test accuracy: ", test_acc*100.0
                #print "FGSM attacked test accuracy: ", fgsm_test_adv_acc*100.0
                #print "CarliniWagner attacked test accuracy: ", cw_test_adv_acc*100.0
                print "CATN attacked test accuracy: ", catn_test_adv_acc*100.0
'''
