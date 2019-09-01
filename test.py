import os
import cv2
import random
import numpy as np
from torch.autograd import Variable
import pandas as pd
import torch
import torch.nn as nn
from model.networkmodel import NetworkModel
from model.resnet import ResNet
from model.resnet import BasicBlock
import torch.optim as optim
from CTData import dataloader

for epoch in range(8, 9, 1):

    fcn_model = torch.load('checkpoints/ct_model_' + str(epoch) + '.pt')

    if torch.cuda.is_available():
        fcn_model.cuda()
    fcn_model.eval()

    test_img_dir = "../data/train2_dataset_process/"

    t = 0
    f = 0
    for patient_name in os.listdir(test_img_dir):
        patient = []
        for ct_name in os.listdir(test_img_dir + patient_name + "/"):
            ct = cv2.imread(test_img_dir + patient_name + "/" + ct_name)
            ct = ct - [104.00699, 116.66877, 122.67892]
            ct = cv2.resize(ct, (128, 128))
            ct = ct.swapaxes(0, 2).swapaxes(1, 2)
            patient.append(ct)

        patient = torch.FloatTensor(patient)

        label_csv = pd.read_csv("../data/train2_label.csv")
        gt = label_csv[label_csv['id'].isin([patient_name])]['ret']
        label = torch.FloatTensor(int(gt))

        patient = (torch.autograd.Variable(patient)).cuda()
        label = (torch.autograd.Variable(label)).cuda()

        output = fcn_model(patient)
        score = torch.sigmoid(output)
        pred = score.data.cpu().numpy()[0]

        if pred[0] >=0.5 :
            pred = 1
        else :
            pred = 0

        if pred == int(gt):
            t = t + 1
        else:
            f = f + 1

    print(t / (t + f))
    print(t)
    print(f)