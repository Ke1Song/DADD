import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
# import resource
import numpy as np
import sklearn.metrics
from ResNet_imagenet import resnet18_ImageNet
from DADD import DADD
from data_manager_imagenet import *
import torch

parser = argparse.ArgumentParser(description='Class Incremental Learning via Dual Augmentation and Dual Augmentation')
parser.add_argument('--epochs', default=141, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='ImageNet100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--semanAug_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_check_DADD/', type=str, help='save files directory')
args = parser.parse_args()


def main():
    data_manager = DataManager()
    cuda_index = "cuda:" + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    print(device)
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + 'x' + str(task_size)
    feature_extractor = resnet18_ImageNet()
    model = DADD(args, file_name, feature_extractor, task_size, device)
    model.checkpoint()
    class_set = list(range(args.total_nc))

    for i in range(1, args.task_num + 1):
        print('task %d' % i)
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i)
        model.train(i, old_class=old_class)
        model.afterTrain()

    ####### Test ######
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("############# Test for each Task #############")
    acc_all = []
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % class_index
        model = torch.load(filename, map_location=cuda_index)
        model.eval()
        acc_up2now = []
        for i in range(current_task + 1):
            if i == 0:
                classes = class_set[:args.fg_nc]
            else:
                classes = class_set[(args.fg_nc + (i - 1) * task_size):(args.fg_nc + i * task_size)]
            testfolder = data_manager.get_dataset(test_transform, index=classes, train=False)
            test_loader = torch.utils.data.DataLoader(
                testfolder, batch_size=100,
                shuffle=False,
                drop_last=True, num_workers=4)
            correct, total = 0.0, 0.0
            for setp, data in enumerate(test_loader):
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()
                with torch.no_grad():
                    outputs = model(imgs)
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num - current_task) * [0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    # print(acc_all)
    a = np.array(acc_all)
    result = []
    for i in range(args.task_num + 1):
        if i == 0:
            result.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result.append(100 * res)
    print(50 * '#')
    print('Forgetting result:')
    print(result)
    print("############# Test for up2now Tasks #############")
    average_acc = 0
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % class_index
        model = torch.load(filename, map_location=cuda_index)
        model.to(device)
        model.eval()

        classes = class_set[:(args.fg_nc + current_task * task_size)]
        testfolder = data_manager.get_dataset(test_transform, index=classes, train=False)
        test_loader = torch.utils.data.DataLoader(
            testfolder, batch_size=100,
            shuffle=False,
            drop_last=False, num_workers=4)
        correct, total = 0.0, 0.0
        for setp, data in enumerate(test_loader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print(accuracy)
        average_acc += accuracy
    print('average acc: ')
    print(average_acc / (args.task_num + 1))


if __name__ == "__main__":
    main()
