import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

use_cuda = torch.cuda.is_available()

class CNNClassifier_custom(nn.Module):
    def __init__(self,input_channel):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier_custom, self).__init__()
        conv1 = nn.Conv1d(in_channels=input_channel, out_channels=50, kernel_size=3)
        # activation ReLU
        pool1 = nn.MaxPool1d(2)
        conv2 = nn.Conv1d(in_channels=50, out_channels=70, kernel_size=3)
        # activation ReLU
        pool2 = nn.MaxPool1d(2)
        conv3 = nn.Conv1d(in_channels=70, out_channels=70, kernel_size=3)
        # activation ReLU
        pool3 = nn.MaxPool1d(2)


        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.BatchNorm1d(50, affine=True),
            pool1,
            conv2,
            nn.ReLU(),
            nn.BatchNorm1d(70, affine=True),
            pool2,
            conv3,
            nn.ReLU(),
            nn.BatchNorm1d(70, affine=True),
            pool3,
        )

        self.generator = nn.Linear(17360,2)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.activation = self.activation.cuda()
            self.generator = self.generator.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        out = torch.flatten(out, 1)
        y = self.activation(self.generator(out))
        return y

class DNNClassifier_custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(328,256)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.generator = nn.Linear(128*2,2)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

        if use_cuda:
            self.emb = self.emb.cuda()
            self.rnn = self.rnn.cuda()
            self.generator = self.generator.cuda()
            self.activation = self.activation.cuda()

    def forward(self,x):
        # |x| = (batch_size, length)
        x = x.long()
        x = x.reshape(x.shape[0],x.shape[2])
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:,-1,:]))
        # |y| = (batch_size, n_classes)
        return y

def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)

if __name__ == '__main__':
    #cnn = CNNClassifier_custom(input_channel=14)
    cnn = DNNClassifier_custom()

    summary(cnn, (1,328))