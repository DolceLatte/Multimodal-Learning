import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

use_cuda = torch.cuda.is_available()

class CNNClassifier_custom(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier_custom, self).__init__()
        conv1 = nn.Conv1d(in_channels=17, out_channels=50, kernel_size=3)
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
            nn.BatchNorm1d(50, affine=True),
            nn.ReLU(),
            pool1,
            conv2,
            nn.BatchNorm1d(70, affine=True),
            nn.ReLU(),
            pool2,
            conv3,
            nn.BatchNorm1d(70, affine=True),
            nn.ReLU(),
            pool3,
        )

        fc1 = nn.Linear(31360, 1000)
        fc2 = nn.Linear(1000, 300)
        fc3 = nn.Linear(300, 9)
        fc4 = nn.Linear(9, 2)

        self.fc_module = nn.Sequential(
            fc1,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            fc2,
            nn.BatchNorm1d(300),
            nn.ReLU(),
            fc3,
            nn.BatchNorm1d(9),
            nn.ReLU(),
            fc4
        )

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        # make linear
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d

        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)
#
# class CNNClassifier_custom(nn.Module):
#     def __init__(self):
#         # 항상 torch.nn.Module을 상속받고 시작
#         super(CNNClassifier_custom, self).__init__()
#         conv1 = nn.Conv1d(in_channels=17, out_channels=50, kernel_size=3)
#         # activation ReLU
#         pool1 = nn.MaxPool1d(2)
#         conv2 = nn.Conv1d(in_channels=50, out_channels=70, kernel_size=3)
#         # activation ReLU
#         pool2 = nn.MaxPool1d(2)
#         conv3 = nn.Conv1d(in_channels=70, out_channels=70, kernel_size=3)
#         # activation ReLU
#         pool3 = nn.MaxPool1d(2)
#         conv4 = nn.Conv1d(in_channels=70, out_channels=50, kernel_size=3)
#         # activation ReLU
#         pool4 = nn.MaxPool1d(2)
#
#         self.conv_module = nn.Sequential(
#             conv1,
#             nn.BatchNorm1d(50, affine=True),
#             nn.ReLU(),
#             pool1,
#             conv2,
#             nn.BatchNorm1d(70, affine=True),
#             nn.ReLU(),
#             pool2,
#             conv3,
#             nn.BatchNorm1d(70, affine=True),
#             nn.ReLU(),
#             pool3,
#             conv4,
#             nn.BatchNorm1d(50, affine=True),
#             nn.ReLU(),
#             pool4,
#         )
#
#         fc1 = nn.Linear(11150, 5000)
#         fc2 = nn.Linear(5000, 300)
#         fc3 = nn.Linear(300, 9)
#         fc4 = nn.Linear(9, 2)
#
#         self.fc_module = nn.Sequential(
#             fc1,
#             nn.BatchNorm1d(5000),
#             nn.ReLU(),
#             fc2,
#             nn.BatchNorm1d(300),
#             nn.ReLU(),
#             fc3,
#             nn.BatchNorm1d(9),
#             nn.ReLU(),
#             fc4
#         )
#
#         # gpu로 할당
#         if use_cuda:
#             self.conv_module = self.conv_module.cuda()
#             self.fc_module = self.fc_module.cuda()
#
#     def forward(self, x):
#         out = self.conv_module(x)
#         # make linear
#         dim = 1
#         for d in out.size()[1:]:
#             dim = dim * d
#         out = out.view(-1, dim)
#         out = self.fc_module(out)
#         return F.softmax(out, dim=1)


def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)

if __name__ == '__main__':
    cnn = CNNClassifier_custom()
    summary(cnn, (17,3600))