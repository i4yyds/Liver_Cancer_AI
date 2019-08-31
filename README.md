# Liver_Cancer_AI

DataFountain2019肝癌影像AI诊断-开源项目

比赛链接：https://www.datafountain.cn/competitions/335

## CT 数据
因为 trainset1 中 7361240E-5A4D-4A1F-A9B4-203E92A0E10B 存在问题，所以将此条数据删除。本项目使用来自 trainset1 的 3599 个训练样本，使用来自trainset2 的 4000 个测试样本。
## CT 数据预处理
CT 数据预处理是个棘手的问题，本项目采用正态分布函数对每个病人的 CT图 采样，每个病人采样 20张 CT图。
## 建模
受机器性能的限制，本项目只使用 ResNet18 作为特征提取器，输入大小设置为 128*128 。
## 训练
python train.py
## 测试
python test.py
