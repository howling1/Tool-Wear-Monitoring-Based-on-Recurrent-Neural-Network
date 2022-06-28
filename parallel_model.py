import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class Net(nn.Module):
    def __init__(self, filter_num, filter_size, window_size, padding, blstm_unit_num, blstm_layer_num):
        super(Net, self).__init__()
        # 定义CNN
        self.conv = nn.Conv1d(1, filter_num, filter_size, padding=padding, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool1d(window_size)

        # 定义Bi-LSTM
        self.bl = nn.LSTM(input_size=1, hidden_size=blstm_unit_num, num_layers=blstm_layer_num, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.dropout1 = nn.Dropout(0.2)

        # 定义全连接网络
        self.fc = nn.Linear(360, 512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # 输出层
        self.outlayer = nn.Linear(512, 2)
        self.softmax = nn.Softmax()

        self.padding = padding
        self.filter_size = filter_size
        self.window_size = window_size
        self.filter_num = filter_num
        self.blstm_unit_num = blstm_unit_num

    def forward(self, x, hidden):
        # CNN运算
        x1 = self.conv(x)
        x1 = self.sigmoid(x1)
        x1 = self.pool(x1)
        x1 = x1.view(-1, 260)

        # Bi—LSTM运算
        x2 = x.view(-1, 130, 1)
        hhh1 = hidden[0]
        x2, (h_n, c_n) = self.bl(x2, hhh1)
        x2 = self.dropout1(x2)
        x2 = x2[:, -1, :]

        # 全连接网络运算
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # 输出
        x = self.outlayer(x)
        x = self.softmax(x)
        return x

    def initHidden(self, batch_size):
        # 对隐含层单元变量全部初始化为0
        out = []
        hidden1 = torch.zeros(4, batch_size, self.blstm_unit_num)
        cell1 = torch.zeros(4, batch_size, self.blstm_unit_num)
        out.append((hidden1, cell1))
        return out


def create_loader(batch_size, train_data, test_data, valid_data):
    """建立数据加载器"""
    x_train = torch.from_numpy(np.array(train_data)[:, 0:-1].astype(np.float32))
    x_test = torch.from_numpy(np.array(test_data)[:, 0:-1].astype(np.float32))
    x_valid = torch.from_numpy(np.array(valid_data)[:, 0:-1].astype(np.float32))

    x_train = x_train.view(-1, 1, 130)
    x_test = x_test.view(-1, 1, 130)
    x_valid = x_valid.view(-1, 1, 130)

    y_train = torch.from_numpy(np.array(pd.get_dummies(
        train_data.iloc[:, -1])).astype(np.float32))
    y_test = torch.from_numpy(np.array(pd.get_dummies(
        test_data.iloc[:, -1])).astype(np.float32))
    y_valid = torch.from_numpy(np.array(pd.get_dummies(
        valid_data.iloc[:, -1])).astype(np.float32))

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, valid_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        tg = torch.max(targets.data, 1)[1].view(-1, 1)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss * tg + \
                 (1 - self.alpha) * (1 - pt) ** self.gamma * BCE_loss * (1 - tg)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def cal_accuracy(predictions, labels):
    """计算识别正确个数与总识别数"""
    pred = torch.max(predictions.data, 1)[1]
    lab = torch.max(labels.data, 1)[1]
    rights = pred.eq(lab.data.view_as(pred)).sum()
    return int(rights), len(labels)


def drawer(valid_loss, valid_acc, train_loss, train_acc):
    """绘制训练过程"""
    fig = plt.figure(figsize=(20, 8), dpi=80)
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss change')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.title('accuracy change')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    batch_size = 25
    num_classes = 2
    filter_num = 10
    filter_size = 3
    padding = 1
    window_size = 5
    blstm_unit_num = 50
    blstm_layer_num = 2
    epochs = 30

    # load data
    is_generative = input('选择使用生成的平衡数据还是原非平衡数据?原非平衡数据则输入0,生成的平衡数据则输入1:')
    dataset_choise = "generative" if is_generative == '1' else "original"
    train_data = pd.read_csv('./data/{} training data.csv'.format(dataset_choise))
    test_data = pd.read_csv('./data/{} test data.csv'.format(dataset_choise))
    valid_data = pd.read_csv('./data/{} valid data.csv'.format(dataset_choise))

    train_loader, test_loader, valid_loader = create_loader(batch_size, train_data, test_data, valid_data)
    net = Net(filter_num, filter_size, window_size, padding, blstm_unit_num, blstm_layer_num)
    net.apply(weights_init)
    criterion = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # train model
    for epoch in range(epochs):
        train_rights = 0
        train_num = 0
        train_loss_sum = 0
        for batch, data in enumerate(train_loader):
            x, y = Variable(data[0], requires_grad=True), Variable(data[1])
            net.train()
            init_hidden = net.initHidden(len(data[0]))  # 初始化LSTM的隐单元变量
            outputs = net(x, init_hidden)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rights, num = cal_accuracy(outputs, y)
            train_rights += rights
            train_num += num
            train_loss_sum += float(loss) * len(data[0])

        train_accuracy = train_rights / train_num
        train_loss = train_loss_sum / train_num
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if 0 == 0:
            # Run over the check set and calculate the classification accuracy
            # on the check set
            valid_rights = 0
            valid_num = 0
            valid_loss_sum = 0
            net.eval()
            for batch, data in enumerate(valid_loader):
                init_hidden = net.initHidden(len(data[0]))
                x, y = Variable(data[0]), Variable(data[1])
                outputs = net(x, init_hidden)
                loss = criterion(outputs, y)
                rights, num = cal_accuracy(outputs, y)
                valid_rights += rights
                valid_num += num
                valid_loss_sum += float(loss) * len(data[0])

            valid_accuracy = valid_rights / valid_num
            valid_loss = valid_loss_sum / valid_num
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(
                '第{}轮, 训练Loss:{:.10f}, 训练准确度:{:.4f},校验Loss:{:.10f}, 校验准确度:{:.4f}'.format(
                    epoch,
                    train_loss,
                    train_accuracy,
                    valid_loss,
                    valid_accuracy))

    # Run on the test set and calculate the total accuracy
    net.eval()
    test_rights = 0
    test_num = 0
    i = 0

    for batch, data in enumerate(test_loader):
        init_hidden = net.initHidden(len(data[0]))
        x, y = Variable(data[0]), Variable(data[1])
        output = net(x, init_hidden)
        rights, num = cal_accuracy(output, y)
        test_rights += rights
        test_num += num
        if i == 0:
            test_true = y.detach().numpy()
            test_pre = output.detach().numpy()
        else:
            test_true = np.vstack((test_true, y.detach().numpy()))
            test_pre = np.vstack((test_pre, output.detach().numpy()))
        i += 1

    test_accuracy = test_rights / test_num
    print('测试集准确率:', test_accuracy)
    print("LSTM模型的AUC指标：", roc_auc_score(test_true, test_pre))
    drawer(valid_losses, valid_accuracies, train_losses, train_accuracies)

    # save the model
    torch.save(net, 'model/并行模型.pkl')

