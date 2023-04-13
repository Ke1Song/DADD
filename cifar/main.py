import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
import numpy as np
from DADD import DADD
from ResNet import resnet18_cbam
from iCIFAR100 import iCIFAR100


parser = argparse.ArgumentParser(description='Class Incremental Learning via Dual Augmentation')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--seman_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_check_DADD/', type=str, help='save files directory')
args = parser.parse_args()



def main():
    cuda_index = 'cuda:0'
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + 'x' + str(task_size)
    feature_extractor = resnet18_cbam()
    model = DADD(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))
    # model.checkpoint()

    for i in range(args.task_num + 1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i)
        model.train(i, old_class=old_class)
        model.afterTrain()

    ####### Test ######
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    print("############# Test for each Task #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    acc_all = []
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % class_index
        model = torch.load(filename, map_location=cuda_index)
        model.eval()
        acc_up2now = []
        for i in range(current_task + 1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]  # 50-55-60-...-100
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size,
                                     num_workers=8
                                     )
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):  # setp是下标 后面三个是属性
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():  # 在梯度不更新的前提下
                    outputs = model(imgs)
                predicts = torch.max(outputs, dim=1)[1]  # 一列里面最大的值,加上[1]就是预测值
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
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    average_acc = 0
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % class_index
        model = torch.load(filename, map_location=cuda_index)
        model.to(device)
        model.eval()
        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=8
                                 )
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
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
