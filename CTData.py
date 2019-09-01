from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pandas as pd
import numpy as np
import gzip

import torch
import torchvision.models.vgg

class CTDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.train_img_dir = "./data/train_dataset_process/"

    def __len__(self):
        return len(os.listdir(self.train_img_dir))

    def __getitem__(self, idx):
        patient_name = os.listdir(self.train_img_dir)[idx]

        patient = []
        for ct_name in os.listdir(self.train_img_dir + patient_name + "/"):
            ct = cv2.imread(self.train_img_dir + patient_name + "/" + ct_name)
            ct = ct - [104.00699, 116.66877, 122.67892]
            ct = cv2.resize(ct, (128, 128))
            ct = ct.swapaxes(0, 2).swapaxes(1, 2)
            patient.append(ct)

        patient = torch.FloatTensor(patient)

        label_csv = pd.read_csv("./data/train_label.csv")
        label = label_csv[label_csv['id'].isin([patient_name])]['ret']
        if int(label) == 1:
            label = True
        else:
            label = False

        item = {'patient': patient, 'label': label}
        return item


ct_data = CTDataset()
dataloader = DataLoader(ct_data, batch_size=1, shuffle=True, num_workers=8)


if __name__ =='__main__':
    for batch in dataloader:
        break
