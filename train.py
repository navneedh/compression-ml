import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from cifar_models import *
from utils.dataset import *
from utils.image_utils import *
from cifar_models.vgg import *
device = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib

BATCH_SIZE = 128
EPOCHS = 40

transform = transforms.Compose(
            [
                        transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ]
            )

train_dset = CifarDataset("_bls_033", train = True, transform = transform)
test_dset = CifarDataset("_bls_033", train = False, transform = transform)

train_loader = DataLoader(train_dset, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(test_dset, batch_size = BATCH_SIZE, shuffle = False)

net = VGG('VGG11').to(device)

def train(trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    accuracy = []
    losses = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return correct/total, train_loss/(50000/BATCH_SIZE)

def test(testloader):
    softmax_output = nn.Softmax()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    metrics = []
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct/total, test_loss/(10000/BATCH_SIZE)


train_acc, train_loss, test_acc, test_loss = [], [], [], []
for epoch in tqdm(range(EPOCHS)):
    train_a, train_l = train(train_loader)
    test_a, test_l = test(test_loader)

    train_acc.append(train_a)
    train_loss.append(train_l)
    test_acc.append(test_a)
    test_loss.append(test_l)

print(test_acc)