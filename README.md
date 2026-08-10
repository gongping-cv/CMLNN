# CMLNN
Code for our paper: "Cross-Modal Control of Temporal Integration for Video Action Recognition: A CMLNN Framework"

1. 关于预训练权重
我们CMLNN架构中Backbone使用TimeSformer，需在项目下建立“./data/weights/timesformer_k400”目录，程序第一次运行时自动下载官方config.json 和 pytorch_model.bin到该目录下。

2. 关于数据集的视频文件
（1）本项目的处理方式是先将数据集中的视频文件转换成.npy，并对应生成optical-flow的.npy文件。根据官方训练集/测试集/验证集的划分，生成对应的list.txt。
（2）以UCF101数据集为例，训练时程序加载的“train_RGB_npy_Split01_list.txt”逐行罗列了所有的训练RGB.npy文件路径，例如：
train/0_ApplyEyeMakeup_g11_c01_RGB.npy
train/0_ApplyEyeMakeup_g11_c02_RGB.npy
train/0_ApplyEyeMakeup_g11_c03_RGB.npy
train/0_ApplyEyeMakeup_g11_c04_RGB.npy
train/1_ApplyLipstick_g11_c01_RGB.npy
train/1_ApplyLipstick_g11_c02_RGB.npy
train/1_ApplyLipstick_g11_c03_RGB.npy
train/1_ApplyLipstick_g11_c04_RGB.npy
……
（3）test_RGB_npy_Split01_list.txt文件类似，
test/0_ApplyEyeMakeup_g01_c01_RGB.npy
test/0_ApplyEyeMakeup_g01_c02_RGB.npy
test/0_ApplyEyeMakeup_g01_c03_RGB.npy
test/1_ApplyLipstick_g01_c01_RGB.npy
test/1_ApplyLipstick_g01_c02_RGB.npy
test/1_ApplyLipstick_g01_c03_RGB.npy
……
test/0_ApplyEyeMakeup_g01_c06_RGB.npy
（4）flow.npy通过replace对应的RGB.npy路径加载（程序中有完整的加载逻辑）
（5）可以根据实际情况改成不同的数据集或不同形式的数据文件。

