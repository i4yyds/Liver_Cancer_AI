import os
import shutil

train_dataset = os.listdir("./train2_dataset/")
train_dataset_process = os.listdir("./train2_dataset_process/")
a = 0
for i in train_dataset:
    if i not in train_dataset_process:
        if os.path.exists('./a/' + str(i)):
            os.makedirs('./a/' + str(i))
        shutil.copytree("./train2_dataset/" + str(i), './a/' + str(i))

