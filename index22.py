import os

data_dir_in = "./a_process/"

for i in os.listdir(data_dir_in):
    count = 0
    for j in os.listdir(data_dir_in + i + "/"):
        count = count + 1
    if count < 1:
        print(i)