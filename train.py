# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.networkmodel import NetworkModel
from model.resnet import ResNet
from model.resnet import BasicBlock
import torch.optim as optim
from CTData import dataloader
import time
import numpy as np

if __name__ == "__main__":

    resnet18_model = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet18_model.load_state_dict(torch.load("./weights/resnet18-5c106cde.pth"))
    model = NetworkModel(pretrained_net=resnet18_model, n_class=1)

    fcn_model = model.cuda()
    criterion = nn.BCELoss().cuda()
    # optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    optimizer = optim.Adam(fcn_model.parameters(), lr=1e-4)

    for epo in range(20):
        index = 0
        epo_loss = 0

        for item in dataloader:

            # time_end = time.time()
            # print('totally cost', time_end - time_start)
            # time_start = time.time()

            index += 1
            patient = item['patient']
            label = item['label']

            patient = (torch.autograd.Variable(patient.squeeze(0))).cuda()
            label = (torch.autograd.Variable(label)).cuda()

            optimizer.zero_grad()
            logits = model(patient)

            compute_loss = criterion(torch.sigmoid(logits), (label.unsqueeze(1)).float())

            compute_loss.backward()
            iter_loss = compute_loss.item()
            epo_loss += iter_loss
            optimizer.step()

            # if np.mod(index, 1000) == 1:
            #     print('epoch {}, {}/{}, loss is {}'.format(epo, index, len(dataloader), iter_loss))

        print('epoch ' + str(epo+1) + ' loss = %f' % (epo_loss / len(dataloader)))

        if np.mod(epo+1, 1) == 0:
            torch.save(fcn_model, 'checkpoints/ct_model_{}.pt'.format(epo+1))
            print('saveing checkpoints/ct_model_{}.pt'.format(epo+1))
