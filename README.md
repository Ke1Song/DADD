# DADD
DADD - Official PyTorch Implementation
![image](https://user-images.githubusercontent.com/33835813/231756431-07cff8ef-8f33-41b1-93d9-9702d0bf9249.png)
[IJCNN 2023] Non-exemplar Class-incremental Learning via Dual Augmentation and Dual Distillation
Ke Song, Quan Xia, Zhaoyong Qiu

Usage
We run the code with torch version: 1.11.0+cu113, python version: 3.9.7\\
Train CIFAR100\\
cd CIFAR
python main.py
Train Tiny-ImageNet\\
cd Tiny-ImageNet
python main_tiny.py
Train ImageNet-Subset\\
cd ImageNet-Subset
python main_PASS_imagenet.py
