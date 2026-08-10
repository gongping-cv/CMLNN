# CMLNN
Code for our paper: "Cross-Modal Control of Temporal Integration for Video Action Recognition: A CMLNN Framework"

0. 本项目实验环境python 3.8，基本安装包括但不限于：
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117  （根据实际情况修改）
pip install transformers
pip install opencv-contrib-python-headless


1. 关于预训练权重
我们CMLNN架构中Backbone使用TimeSformer，需在项目下建立“./data/weights/timesformer_k400”目录，程序第一次运行时自动下载官方config.json 和 pytorch_model.bin到该目录下。


2. 关于数据集的视频文件
（1）本项目的处理方式是先将数据集中的视频文件转换成.npy，并对应生成optical-flow的.npy文件。根据官方训练集/测试集/验证集的划分，生成对应的list.txt。
（2）以UCF101数据集为例，训练时程序加载的“train_RGB_npy_Split01_list.txt”逐行罗列了所有的训练RGB.npy文件路径。
（3）测试时加载的test_RGB_npy_Split01_list.txt文件类似。
（4）flow.npy通过replace对应的RGB.npy路径加载（程序中有完整的加载逻辑）
（5）可以根据实际情况改成不同的数据集或不同形式的数据文件。

