# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transform
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data.dataset import Dataset
import numpy as np
import glob
import pandas as pd

import torch.onnx


# Hyper Parameters
sequence_length = 100  # 序列长度
input_size = 3  # 输入数据的特征维度
hidden_size = 200  # 隐藏层的size
num_layers = 3  # 有多少层
# 层数过多，可能导致过深，梯度消失，学不到任何有用输出! layer=10基本就没有效果了

num_classes = 10  # 分类
batch_size = 800
num_epochs = 50
learning_rate = 0.001
train_num = 1000
test_num = 600

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


use_bi = True
if use_bi:
    direction = 2
else:
    direction = 1

tr_plt = 5 # For plot
te_plt = 2


def findFiles(path): return glob.glob(path)


class WriteID(Dataset):
    def __init__(self, csv_file, transforms=None, train=True, test=False):
        # 获取所有数据的地址
        self.test = test
        self.train = train
        self.transforms = transforms
        self.path = pd.read_csv(csv_file)

    def __getitem__(self, index):
        item = self.path.iloc[index, :]
        data = np.load(item['RHS_path'])
        # data_path = './dataset/Task1/Train10/7/000704.npy'
        label = int(item['label'])
        length = int(item['len'])  # 为了解码长度
        # data = torch.FloatTensor(data) #array
        return data, label, length

    def __len__(self):
        return len(self.path)


print('Data preparing ...')
train_dataset = WriteID(csv_file='/data/data_98HKWr05Uo7j/WriteId/dataset/csv_data/1new_100RHSTrain.csv', train=True)

test_dataset = WriteID(csv_file='/data/data_98HKWr05Uo7j/WriteId/dataset/csv_data/1new_100RHSTest.csv', train=False,
                       test=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=use_bi)
        # 双向要*2
        self.fc = nn.Linear(hidden_size * direction, num_classes)

    def forward(self, x, seq_lengths):
        # 对序列长度解析

        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = (torch.zeros(num_layers * direction, x.size(0), hidden_size)).to(device)  # 双向*2
        c0 = (torch.zeros(num_layers * direction, x.size(0), hidden_size)).to(device)

        pack = nn.utils.rnn.pack_padded_sequence(x.float(), seq_lengths, batch_first=True)

        # Forward propagate RNN
        # out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)
        out, (h_out, c_out) = self.lstm(pack, (h0, c0))

        # unpack
        all_output, all_length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        row_indices = torch.arange(0, all_output.size(0)).long()  # 取对应序列长度的输出，row为每一个样本，col为其对应真实长度
        col_indices = all_length - 1
        last_output = all_output[row_indices, col_indices, :]#.unsqueeze(1)
        #last_output = all_output[row_indices, col_indices, :]  # batch 800

        # Decode hidden state of last time step
        # out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        out = self.fc(last_output)#.squeeze()
        return out


# 双向两层
def weights_init(m):
    classname = m.__class__.__name__
    for layer in m.modules():
        if isinstance(layer, nn.LSTM):
            nn.init.normal_(layer.weight_ih_l0.data, 0, 0.01)
            nn.init.normal_(layer.weight_hh_l0.data, 0, 0.01)
            nn.init.normal_(layer.weight_ih_l1.data, 0, 0.01)
            nn.init.normal_(layer.weight_hh_l1.data, 0, 0.01)
            # nn.init.kaiming_uniform_(m.weight.data)
            # nn.init.kaiming_nromal_(m.weight.data)
            nn.init.constant_(layer.bias_ih_l0.data, 5.0)
            nn.init.constant_(layer.bias_ih_l0_reverse.data, 5.0)
            nn.init.constant_(layer.bias_hh_l0.data, 0.0)
            nn.init.constant_(layer.bias_hh_l0_reverse.data, 0.0)
            nn.init.constant_(layer.bias_ih_l1.data, 5.0)
            nn.init.constant_(layer.bias_ih_l1_reverse.data, 5.0)
            nn.init.constant_(layer.bias_hh_l1.data, 0.0)
            nn.init.constant_(layer.bias_hh_l1_reverse.data, 0.0)


print('Model Loading ...')
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')

rnn = RNN(input_size, hidden_size, num_layers, num_classes)
print(get_parameter_number(rnn))

#rnn.load_state_dict(torch.load('/data/data_98HKWr05Uo7j/WriteId/save/10rnn2-bi-new100.pkl'))
rnn.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0.0005)

# 在模型基础上微调时修改学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


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

        imgs, labels, length = data

        seq_lengths = np.array(length)
        idxs = np.argsort(seq_lengths, axis=0)[::-1]  # [::-1]倒序 从大的长度到小的
        labels = labels[idxs.tolist()]
        seq_lengths = seq_lengths[idxs]
        imgs = imgs[idxs.tolist(), :, :]

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = rnn(imgs, seq_lengths)
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
        i = 0
        rnn.eval()
        for imgs, labels, length in tqdm(test_loader):
            i = i + 1
            seq_lengths = np.array(length)
            idxs = np.argsort(seq_lengths, axis=0)[::-1]  # [::-1]倒序 从大的长度到小的
            labels = labels[idxs.tolist()]
            seq_lengths = seq_lengths[idxs]
            imgs = imgs[idxs.tolist(), :, :]

            imgs = imgs.to(device)
            labels = labels.to(device)
            out = rnn(imgs, seq_lengths)

            loss = criterion(out, labels)
            test_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == labels).sum().item()
            acc = float(num_correct) / imgs.shape[0]
            test_acc += acc

            if (i % te_plt == 0):
                test_acc_list.append(acc)
                test_loss_list.append(loss)


    return test_acc, test_loss, test_acc_list, test_loss_list


def main():
    if os.path.exists('./log'):
        pass
    else:
        os.mkdir('./log')
    if os.path.exists('./save'):
        pass
    else:
        os.mkdir('./save')
    # writer = SummaryWriter('./log')
    save_path = './save/'

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    print('Start training ...')
    for epoch in range(num_epochs):
        scheduler.step()
        print(scheduler.get_lr())
        train_acc, train_loss, acc_list, loss_list = train(epoch)
        test_acc, test_loss, tacc_list, tloss_list = test(epoch)

        # 一个epoch完毕将其更新至tensorboard
        # writer.add_scalar('Train10 Loss', train_loss / len(train_loader), epoch)
        # writer.add_scalar('Train10 Acc', train_acc / len(train_loader), epoch)
        # writer.add_scalar('eval Loss', test_loss / len(test_loader), epoch)
        # writer.add_scalar('eval acc', test_acc / len(test_loader), epoch)

        train_acc_list = train_acc_list + acc_list
        train_loss_list = train_loss_list + loss_list
        test_acc_list = test_acc_list + tacc_list
        test_loss_list = test_loss_list + tloss_list

        print('epoch: {}, Train10 Loss: {:.6f}, Train10 Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format(
            epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader), test_loss / len(test_loader),
            test_acc / len(test_loader)))

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
    plt.savefig(save_path+'10rnnbi2new100.jpg')

    print('Model saving ...')
    # Save the Model
    torch.save(rnn.state_dict(), save_path+'10rnnbi2new100.pkl')


if __name__ == '__main__':
    main()
