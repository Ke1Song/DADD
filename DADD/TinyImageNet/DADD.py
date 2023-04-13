import random
import torch
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork_classaug import network
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import cv2
from torch.autograd import Variable
import numpy as np
import os
import sys
from data_manager_tiny import *


class DADD:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.args = args
        self.numclass = self.args.fg_nc
        self.task_size = task_size
        self.augnumclass = 4 * self.numclass + int(self.numclass * (self.numclass - 1) / 2)
        self.file_name = file_name
        self.model = network(self.augnumclass, feature_extractor)
        self.cov = None
        self.prototype = None
        self.class_label = None
        self.old_model = None
        self.device = device
        self.radius = 0
        self.data_manager = DataManager()
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        class_set = list(range(200))
        if current_task == 0:
            classes = class_set[:self.numclass]
        else:
            classes = class_set[self.numclass - self.task_size:self.numclass]
        trainfolder = self.data_manager.get_dataset(self.train_transform, index=classes, train=True)
        testfolder = self.data_manager.get_dataset(self.test_transform, index=class_set[:self.numclass], train=False)

        self.train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=self.args.batch_size,
                                                        shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(testfolder, batch_size=self.args.batch_size,
                                                       shuffle=False, drop_last=False, num_workers=8)
        if current_task > 0:
            old_class = self.numclass - self.task_size
            self.augnumclass = 4 * self.numclass + int(self.task_size * (self.task_size - 1) / 2)
            self.model.Incremental_learning(old_class, self.augnumclass)
        self.model.train()
        self.model.to(self.device)

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        accuracy = 0
        for epoch in range(self.args.epochs):
            for i, data in enumerate(self.train_loader):
                images, target = data
                images, target = images.to(self.device), target.to(self.device)
                opt.zero_grad()
                loss = self._compute_loss(images, target, old_class)
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()
            if epoch % 10 == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d,accuracy:%.5f' % (epoch, accuracy))
        self.protoSave(self.model, self.train_loader, current_task)
        return accuracy

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, data in enumerate(testloader):
            imgs, labels = data
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = outputs[:, :self.numclass]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cuda() == labels.cuda()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def attention_distillation_loss(self, attention_map1, attention_map2):
        """Calculates the attention distillation loss"""
        attention_map1 = torch.norm(attention_map1, p=2, dim=0)
        attention_map2 = torch.norm(attention_map2, p=2, dim=0)
        return torch.norm(attention_map2 - attention_map1, p=1, dim=1).sum().mean()

    def _compute_loss(self, imgs, target, old_class=0):
        imgs, target = imgs.to(self.device), target.to(self.device)
        imgs, target = self.classAug(imgs, target)
        output = self.model(imgs)
        loss_cls = nn.CrossEntropyLoss()(output / self.args.temp, target)
        if self.old_model == None:
            return loss_cls
        else:
            with GradCAM(self.model, self.model.feature.layer4) as gradcam:
                with GradCAM(self.old_model, self.old_model.feature.layer4) as gradcam_old:
                    attmap_old = gradcam_old(imgs)
                    attmap = gradcam(imgs)
                    attmap = torch.from_numpy(attmap)
                    attmap_old = torch.from_numpy(attmap_old)
                    attention_loss = self.attention_distillation_loss(attmap_old, attmap)
                    feature = self.model.feature(imgs)
                    feature_old = self.old_model.feature(imgs)
                    loss_kd = torch.dist(feature, feature_old, 2)
                    proto_aug = []
                    proto_aug_label = []
                    index = list(range(old_class))
                    for _ in range(self.args.batch_size):
                        np.random.shuffle(index)
                        temp = self.prototype[index[0]] + random.uniform(0.0, self.radius) * np.random.normal(0, 1, 512)
                        proto_aug.append(temp)
                        proto_aug_label.append(self.class_label[index[0]])
                    proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).to(self.device)
                    proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device).long().cuda()
                    soft_feat_aug = self.model.fc(proto_aug)
                    soft_feat_aug = soft_feat_aug[:, :self.numclass]

                    ratio = 2.5
                    isda_aug_proto_aug = self.semanAug(proto_aug, soft_feat_aug, proto_aug_label, ratio)
                    loss_semanAug = nn.CrossEntropyLoss()(isda_aug_proto_aug / self.args.temp, proto_aug_label)
                    return loss_cls + self.args.seman_weight * loss_semanAug + self.args.kd_weight * loss_kd + \
                           self.args.ad_weight * attention_loss

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = path + '%d_model.pkl' % self.numclass
        self.model.saveOption(self.numclass)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()
        self.numclass += self.task_size

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                images, target = data
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]
        prototype = []
        cov = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            cov_class = np.cov(feature_classwise.T)
            cov.append(cov_class)
            if current_task == 0:
                cov1 = np.cov(feature_classwise.T)
                radius.append(np.trace(cov1) / feature_dim)
        if current_task == 0:
            self.cov = np.concatenate(cov, axis=0).reshape([-1, 512, 512])
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.cov = np.concatenate((cov, self.cov), axis=0)
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

    def semanAug(self, features, y, labels, ratio):
        N = features.size(0)
        C = self.numclass
        A = features.size(1)
        weight_m = list(self.model.fc.parameters())[0]
        weight_m = weight_m[:self.numclass, :]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV = self.cov
        labels = labels.cpu()
        CV_temp = torch.from_numpy(CV[labels]).to(self.device)
        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp.float()), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(self.device).expand(N, C, C)).sum(2).view(N, C)
        aug_result = y + 0.5 * sigma2
        return aug_result

    def classAug(self, x, y, mix_times=2):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            index = torch.randperm(batch_size).to(self.device)
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = random.uniform(0.3, 0.7)
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        mix_target = torch.Tensor(mix_target).cuda()
        rot_x = torch.stack([torch.rot90(x, k, (2, 3)) for k in range(1, 4)], 1).view(-1, 3, 64, 64)
        rot_y = torch.stack([self.numclass * k + y for k in range(1, 4)], 1).view(-1)
        for item in rot_x:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        y = torch.cat((y, rot_y), 0)
        y = torch.cat((y, mix_target), 0)
        y = y.long()
        return x, y

    def generate_label(self, y_a, y_b):
        if self.old_model == None:
            y_a, y_b = y_a, y_b
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = ((2 * self.numclass - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        else:
            y_a = y_a - (self.numclass - self.task_size)
            y_b = y_b - (self.numclass - self.task_size)
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = int(((2 * self.task_size - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
        return label_index + 4 * self.numclass


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cuda().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cuda().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(input_tensor)

        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
