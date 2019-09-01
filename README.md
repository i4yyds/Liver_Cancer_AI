# Liver_Cancer_AI

DataFountain2019肝癌影像AI诊断-开源项目

比赛链接：https://www.datafountain.cn/competitions/335

## CT 数据
因为 trainset1 中 7361240E-5A4D-4A1F-A9B4-203E92A0E10B 存在问题，trainset2 中 83483964-89C3-4BDE-8DE5-48CB04B48B7D、FE6792D2-752E-44BA-BC5F-264F289B4DF9也存在问题，所以将这些数据删除。本项目使用来自 trainset1 的 3599 个训练样本，使用来自trainset2 的 3972 个测试样本。
## CT 数据预处理
CT 数据预处理是个棘手的问题，本项目采用正态分布函数对每个病人的 CT图 采样，每个病人采样 20张 CT图。
## 建模
本项目使用 ResNet18 作为特征提取器，输入大小设置为 128*128 。提取特征后，利用通道注意力机制为特征加权，接着使用卷积融合特征。最后经全连接层输出预测结果。受由于硬件条件的限制，特征提取器，图像输入尺寸和全连接层数均做了限制。读者可以再增加类似CBAM中的空间注意力机制进一步提高精度。类似方法可以得到0.98的AC精度！！！
## 训练
python train.py
## 测试
python test.py
## 参考以下论文能进一步提高精度
http://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf


# 求star，谢谢！
