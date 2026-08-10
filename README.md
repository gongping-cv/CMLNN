# Code for our paper: "Cross-Modal Control of Temporal Integration for Video Action Recognition: A CMLNN Framework"

1. Environment Setup

This project is developed with Python 3.8. The essential packages include (but are not limited to):

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117

pip install transformers

pip install opencv-contrib-python-headless

Please adjust the PyTorch and CUDA versions according to your actual environment.

2. Pre-trained Weights

Our CMLNN architecture adopts TimeSformer as the backbone.

Create the directory ./data/weights/timesformer_k400 under the project root.

The program will automatically download the official pre-trained weights, including config.json and pytorch_model.bin, to this directory upon the first run.

3. Dataset Preparation

(1) Video Preprocessing

The pipeline of this project is as follows: convert the original video files of the dataset into .npy format, and simultaneously generate the corresponding optical-flow .npy files (detailed optical-flow extraction parameters are provided in the paper).

(2) Generating File Lists

According to the official training/validation/testing splits, generate the corresponding .txt list files.

For example, in the dataset_npy_keepall/ucf101_processed/ directory:

train_RGB_npy_Split01_list.txt – the training list for Split 1 of the UCF101 dataset.

test_RGB_npy_Split01_list.txt – the corresponding test list.

(3) Data Loading During Training

During training, the program reads train_RGB_npy_Split01_list.txt and loads the input data based on the listed RGB .npy file paths.

For optical-flow data, the program automatically replaces the RGB paths with the corresponding _FLOW.npy paths for loading.

(4) Adapting to Other Datasets

You can modify the code to use different datasets or different data file formats according to your needs, as long as the directory structure and list file format remain consistent.
