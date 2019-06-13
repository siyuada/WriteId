# -*- coding: utf-8 -*- 
# @Time : 2019/5/28 21:46 
# @Author : Siyu Huang
# @Project : WriterID 
"""
@File : CNN-10.py 
function:
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
import tensorboardX
import torch.onnx

import math

vgg11 = models.vgg11_bn(pretrained=True)
'''
for name,para in vgg11.named_parameters():
    print(name, para.size())
torch.save(vgg11, 'vgg11_pre.pkl')
'''

print(os.getcwd())
batch_size = 64
num_classes = 10 # 分类
num_epochs = 60
learning_rate = 0.001
tr_plt = 23
te_plt = 7

#train_transform = transform.Compose([transform.RandomHorizontalFlip(),transform.RandomVerticalFlip(), transform.RandomCrop(128),transform.Resize((224,224)),transform.ToTensor()])  # 数据预处理
train_transform = transform.Compose([transform.RandomRotation(30), transform.RandomCrop(180), transform.Resize((224,224)),transform.ToTensor()])  # 数据预处理
test_transform = transform.Compose([transform.RandomRotation(30), transform.RandomCrop(180),transform.Resize((224,224)),transform.ToTensor()])

train_csv = 'dataset/image_data/Data10/Train.csv'
test_csv = 'dataset/image_data/Data10/Validation.csv'
dataset_root = 'dataset/image_data/Data10'

class WriteID_img(Dataset):
    def __init__(self, root, csv_file, transforms=None, train=True, test=False):
        #csv:Train/0/图片
        # 获取所有数据的地址
        self.dataset_root = root
        # dataset/image_data/Data10
        self.test = test
        self.train = train
        self.transforms = transforms
        if self.train:
            self.path = pd.read_csv(csv_file)
        else:
            self.path = pd.read_csv(csv_file)

    def __getitem__(self, index):
        item = self.path.iloc[index,:]
        img_path = os.path.join(self.dataset_root, item['imgs_path'])
        label = int(item['label']-1)
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.path)

print('Data Preparing ...')
train_dataset = WriteID_img(dataset_root, train_csv, train=True, transforms=train_transform)
test_dataset = WriteID_img(dataset_root, test_csv, train=False, test=True, transforms=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Model Loading ...')
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')
'''
def feature_layer():
    layers = []
    vgg16 = torch.load('vgg11_pre.pth').features
    for name, layer in vgg16._modules.items():
        if isinstance(layer, nn.ReLU):#nn.Conv2d):
            layers += [layer, nn.Dropout2d(0.5)]
        else:
            layers += [layer]
    #print(layers)
    features = nn.Sequential(*layers)
    #print(features)
    return features


def class_layer():
    layers = []
    vgg16 = torch.load('vgg11_pre.pth').classifier
    for name, layer in vgg16._modules.items():
        
        #print(name, layer)
        if name == '6':
            fc_feature = vgg16[6].in_features
            layers += [nn.Linear(fc_feature, num_classes), nn.Dropout(0.5)]
        else:
            layers += [layer]
        
        layers += [layer]
    #print(layers)
    classifier = nn.Sequential(*layers)
    return classifier

class vggg(nn.Module):
    def __init__(self):
        super(vggg, self).__init__()
        self.vgg16test = torch.load('vgg11_pre.pkl')
        self.features = self.vgg16test.features #feature_layer()
        #self.classifier = class_layer()
        self.classifier = nn.Sequential(  # 分类器结构
            # fc6
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # fc7
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # fc8
            nn.Linear(4096, num_classes))
        # 初始化权重
        self._initialize_weights()


    def forward(self, x):
        #print('aaa', x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1) #展成向量
        #print('bbb', x.shape)
        x = self.classifier(x)
        #print(x.shape)

        return x

    def _initialize_weights(self):
        fc_feature = self.vgg16test.classifier[0].in_features
        self.classifier[0] = nn.Linear(fc_feature, 4096)
        fc_feature = self.vgg16test.classifier[3].in_features
        self.classifier[3] = nn.Linear(fc_feature, 4096)
        fc_feature = self.vgg16test.classifier[6].in_features
        self.classifier[6] = nn.Linear(fc_feature, num_classes)
'''
# 在VGG16基础上修改网络
#vgg16 = vggg()

# 简单参数修改
#fc_feature = vgg16.classifier[6].in_features
#vgg16.classifier[6] = nn.Linear(fc_feature, num_classes)


fc_feature = vgg11.classifier[6].in_features
vgg11.classifier[6] = nn.Linear(fc_feature, num_classes)
vgg11 = vgg11.to(device)

vgg11.classifier[2] = nn.Dropout2d(0.5)
vgg11.classifier[5] = nn.Dropout2d(0.5)


criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(vgg16.parameters(), lr=learning_rate, weight_decay=0.00005)
#optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
weight_p, bias_p = [],[]
for name, p in vgg11.named_parameters():
  if 'bias' in name:
     bias_p += [p]
  else:
     weight_p += [p]
# 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
optimizer = torch.optim.SGD([
          {'params': weight_p, 'weight_decay':5e-5},
          {'params': bias_p, 'weight_decay':0}
          ], lr=5e-4, momentum=0.9)


def train(epoch):
    # Train10 the Model
    train_acc_list = []
    train_loss_list = []
    train_loss = 0
    train_acc = 0
    running_loss = 0.0
    vgg11.train()
    i = 0
    for data in tqdm(train_loader, 0):
        i = i + 1
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = vgg11(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        running_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == labels).sum().item()
        acc = float(num_correct) / imgs.shape[0]
        train_acc += acc

        if (i % tr_plt == 0):
            train_acc_list.append(acc)
            train_loss_list.append(loss)

    return train_acc, train_loss, train_acc_list, train_loss_list


def test(epoch):
    vgg11.eval()
    with torch.no_grad():
        test_acc_list = []
        test_loss_list = []
        test_loss = 0
        test_acc = 0
        i=0
        for data in tqdm(test_loader,0):
            i=i+1

            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = vgg11(imgs)
            loss = criterion(out,labels)
            test_loss+=loss.item()
            _,pred = out.max(1)
            num_correct = (pred==labels).sum().item()
            acc = float(num_correct)/imgs.shape[0]
            test_acc+=acc

            if (i % te_plt == 0):
                test_acc_list.append(acc)
                test_loss_list.append(loss)
    return test_acc,test_loss,test_acc_list,test_loss_list

def main():
    if os.path.exists('./log'):
        pass
    else:
        os.mkdir('./log')
    if os.path.exists('./save'):
        pass
    else:
        os.mkdir('./save')
    #writer = SummaryWriter('./log')
    save_path = './save/'

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    print('Start training ...')
    for epoch in range(num_epochs):
        train_acc,train_loss,acc_list,loss_list = train(epoch)
        test_acc, test_loss, tacc_list, tloss_list = test(epoch)

        # 一个epoch完毕将其更新至tensorboard
        #writer.add_scalar('Train10 Loss', train_loss / len(train_loader), epoch)
        #writer.add_scalar('Train10 Acc', train_acc / len(train_loader), epoch)
        #writer.add_scalar('eval Loss', test_loss / len(test_loader), epoch)
        #writer.add_scalar('eval acc', test_acc / len(test_loader), epoch)

        train_acc_list = train_acc_list + acc_list
        train_loss_list = train_loss_list + loss_list
        test_acc_list = test_acc_list + tacc_list
        test_loss_list = test_loss_list + tloss_list

        print('epoch: {}, Train10 Loss: {:.6f}, Train10 Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format(epoch+1, train_loss / len(train_loader), train_acc / len(train_loader), test_loss / len(test_loader), test_acc / len(test_loader)))

    print('Model saving ...')
    # Save the Model
    torch.save(vgg11, 'vgg-11-2.pth')

    # 画图
    fig = plt.figure(figsize=(14.40, 9))
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(len(train_loss_list)), np.array(train_loss_list), label='train_loss')
    ax1.plot(np.arange(len(train_acc_list)), np.array(train_acc_list), label='train_acc')
    ax1.plot(np.arange(len(test_loss_list)), np.array(test_loss_list), label='test_loss')
    ax1.plot(np.arange(len(test_acc_list)), np.array(test_acc_list), label='test_acc')
    ax1.legend(loc='best')
    ax1.set_xlabel('train,batch-size=1000, LSTM layer=2')
    ax1.set_title("Learning Curve")
    ax1.set_ylabel('Loss/Accuracy')
    ax1.grid()
    plt.savefig('./CNN-10vgg11-2.jpg')

if __name__ == '__main__':
    main()

