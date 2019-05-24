# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transform
from torch.autograd import Variable
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import Dataset
import numpy as np
import glob

# Hyper Parameters
sequence_length = 100  # 序列长度
input_size = 3  # 输入数据的特征维度
hidden_size = 800  # 隐藏层的size
num_layers = 5  # 有多少层

num_classes = 10 # 分类
batch_size = 1200
num_epochs = 70
learning_rate = 0.001
train_num = 1000
test_num = 600

tr_plt = 9
te_plt = 4

def findFiles(path): return glob.glob(path)

class WriteID(Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        # 获取所有数据的地址
        self.test = test
        self.train = train
        self.transforms = transforms
        if self.train:
            root_ = root + 'Train10/'
            path = [findFiles(root_+str(i)+'/*.npy') for i in os.listdir(root_)]
            self.path = np.reshape(np.array(path), (num_classes * train_num))
        else:
            root_ = root + 'Test10/'
            path = [findFiles(root_ + str(i) + '/*.npy') for i in os.listdir(root_)]
            self.path = np.reshape(np.array(path), (num_classes * test_num))


    def __getitem__(self, index):
        data_path = self.path[index]
        data = np.load(data_path)
        # data_path = './dataset/Task1/Train10/7/000704.npy'
        label = int(data_path.split('/')[-2])
        data = torch.FloatTensor(data) #array
        return data, label


    def __len__(self):
        return len(self.path)

train_dataset = WriteID(root='./dataset/Task1/', train=True)

test_dataset = WriteID(root='./dataset/Task1/', train=False, test=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=2,shuffle=False)


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out

def weights_init(m): # TODO
    classname = m.__class__.__name__
    #nn.init.xavier_uniform_(m.weight.data)
    nn.init.xavier_normal_(m.weight.data, gain=0.01)
    #nn.init.kaiming_uniform_(m.weight.data)
    #nn.init.kaiming_nromal_(m.weight.data)
    nn.init.constant_(m.bias.data,0.0)


rnn = RNN(input_size, hidden_size, num_layers, num_classes)
#rnn.apply(weights_init) # 权重初始化！
rnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

def train(epoch):
    # Train10 the Model
    train_acc_list = []
    train_loss_list = []
    train_loss = 0
    train_acc = 0
    running_loss = 0.0
    rnn.train()
    i = 0
    for data in tqdm(train_loader, 0):
        i = i + 1
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        out = rnn(imgs)
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
    with torch.no_grad():
        test_acc_list = []
        test_loss_list = []
        test_loss = 0
        test_acc = 0
        i=0
        rnn.eval()
        for imgs,labels in tqdm(test_loader):
            i=i+1
            imgs = imgs.view(-1,sequence_length, input_size).cuda()
            labels = labels.cuda()
            out = rnn(imgs)
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
    writer = SummaryWriter('./log')
    save_path = './save/'

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    for epoch in range(num_epochs):
        train_acc,train_loss,acc_list,loss_list = train(epoch)
        test_acc, test_loss, tacc_list, tloss_list = test(epoch)

        # 一个epoch完毕将其更新至tensorboard
        writer.add_scalar('Train10 Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Train10 Acc', train_acc / len(train_loader), epoch)
        writer.add_scalar('eval Loss', test_loss / len(test_loader), epoch)
        writer.add_scalar('eval acc', test_acc / len(test_loader), epoch)

        train_acc_list = train_acc_list + acc_list
        train_loss_list = train_loss_list + loss_list
        test_acc_list = test_acc_list + tacc_list
        test_loss_list = test_loss_list + tloss_list

        print('epoch: {}, Train10 Loss: {:.6f}, Train10 Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format(epoch+1, train_loss / len(train_loader), train_acc / len(train_loader), test_loss / len(test_loader), test_acc / len(test_loader)))

# 画图
    fig = plt.figure(figsize=(14.40, 9))
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(len(train_loss_list)), np.array(train_loss_list), label='train_loss')
    ax1.plot(np.arange(len(train_acc_list)), np.array(train_acc_list), label='train_acc')
    ax1.plot(np.arange(len(test_loss_list)), np.array(test_loss_list), label='test_loss')
    ax1.plot(np.arange(len(test_acc_list)), np.array(test_acc_list), label='test_acc')
    ax1.legend(loc='best')
    ax1.set_xlabel('100 iterations(train,batch-size=128)')
    ax1.set_title("Learning Curve")
    ax1.set_ylabel('Loss/Accuracy')
    ax1.grid()
    plt.savefig('./1.jpg')

    # Save the Model
    torch.save(rnn.state_dict(), 'rnn.pkl')

if __name__ == '__main__':
    main()