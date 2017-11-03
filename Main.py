
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn




cuda_run = torch.cuda.is_available()


train_batch_size = 50
test_batch_size = 50
num_workers = 2


def DATA_PREPROS(train_batch_size, test_batch_size, num_workers):

    print('DATA PREPROS')
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

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes



trainloader, testloader, classes = DATA_PREPROS(train_batch_size, test_batch_size, num_workers)


################ DEFINE NETWORK


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        pass
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x


net = Net()

if cuda_run:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

################# END ~ DEFINE NETWORK


################# LOSS FUNCTION AND OPTIMIZER

    pass


#e.g. criterion = nn.CrossEntropyLoss()
#e.g. optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


################# END ~ LOSS FUNCTION AND OPTIMIZER


################# TRAIN THE NETWORK

def train(epoch_num):
 for epoch in range(epoch_num):  # loop over the dataset multiple times
    print('epoch: %d \n' % epoch)
    net.train()
    train_loss = 0
    for batch_num, (inputs, targets) in enumerate(trainloader):
        if cuda_run:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        output = net(inputs)


        pass

train(1)

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data
#
#         # wrap them in Variable
#         inputs, labels = Variable(inputs), Variable(labels)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.data[0]
#         if i % 2000 == 2:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

#################### END ~ TRAIN THE NETWORK


#################### TEST THE NETWORK
def test(epoch):
    net.eval()
    test_loss = 0
    for batch_num, (inputs, targets) in enumerate(testloader):
        if cuda_run:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output = net(inputs)

        pass
        # correct = 0
        # total = 0
        # for data in testloader:
        #     images, labels = data
        #     outputs = net(Variable(images))
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum()
        #
        # print('Accuracy of the network on the 10000 test images: %d %%' % (
        #     100 * correct / total))

test(1)
#################### END ~ TEST THE NETWORK