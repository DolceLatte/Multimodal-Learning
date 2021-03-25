import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
use_cuda = torch.cuda.is_available()

# class DNNClassifier_custom(nn.Module):
#     def __init__(self):
#         # 항상 torch.nn.Module을 상속받고 시작
#         super(DNNClassifier_custom, self).__init__()
#
#         self.emb = nn.Sequential(
#             nn.Embedding(num_embeddings=256, embedding_dim=16),
#             #nn.ReLU()
#             #nn.Tanh()
#         )
#
#         fc1 = nn.Linear(in_features=5248, out_features=5000)
#         fc2 = nn.Linear(in_features=5000, out_features=3000)
#         fc3 = nn.Linear(in_features=3000, out_features=2200)
#         fc4 = nn.Linear(in_features=2200, out_features=2)
#
#         self.fc_module = nn.Sequential(
#             fc1,
#             nn.BatchNorm1d(5000),
#             #nn.Tanh(),
#             fc2,
#             nn.BatchNorm1d(3000),
#             #nn.Tanh(),
#             fc3,
#             nn.BatchNorm1d(2200),
#             #nn.Tanh(),
#             fc4,
#             nn.BatchNorm1d(2),
#             #nn.Tanh(),
#         )
#
#         # gpu로 할당
#         if use_cuda:
#             self.emb = self.emb.cuda()
#             self.fc_module = self.fc_module.cuda()
#
#     def forward(self, x):
#         x = x.cuda().long()
#         x = x.view(x.size(0), -1)
#         out = self.emb(x)
#         # make linear
#         dim = 1
#         for d in out.size()[1:]:
#             dim = dim * d
#
#         out = out.view(-1, dim)
#         out = self.fc_module(out)
#         return F.softmax(out, dim=1)
#
# class DNNClassifier_custom(nn.Module):
#     def __init__(self):
#         # 항상 torch.nn.Module을 상속받고 시작
#         super(DNNClassifier_custom, self).__init__()
#
#         self.emb = nn.Sequential(
#             nn.Embedding(num_embeddings=256, embedding_dim=16),
#             nn.ReLU()
#         )
#
#         conv1 = nn.Conv1d(in_channels=5249, out_channels=5000, kernel_size=3)
#         # activation ReLU
#         pool1 = nn.MaxPool1d(2)
#         conv2 = nn.Conv1d(in_channels=5000, out_channels=2624, kernel_size=3)
#         # activation ReLU
#         pool2 = nn.MaxPool1d(2)
#         conv3 = nn.Conv1d(in_channels=2624, out_channels=2000, kernel_size=3)
#         # activation ReLU
#         pool3 = nn.MaxPool1d(2)
#
#         self.conv_module = nn.Sequential(
#             conv1,
#             nn.BatchNorm1d(5000, affine=True),
#             nn.ReLU(),
#             pool1,
#             conv2,
#             nn.BatchNorm1d(2624, affine=True),
#             nn.ReLU(),
#             pool2,
#             conv3,
#             nn.BatchNorm1d(2000, affine=True),
#             nn.ReLU(),
#             pool3,
#         )
#
#         self.fc_module = nn.Sequential(
#             nn.Linear(31360, 1000),
#             nn.BatchNorm1d(2),
#             nn.ReLU(),
#         )
#
#         # gpu로 할당
#         if use_cuda:
#             self.emb = self.emb.cuda()
#             self.fc_module = self.fc_module.cuda()
#             self.conv_module = self.conv_module.cuda()
#
#     def forward(self, x):
#         x = x.cuda().long()
#         x = x.view(x.size(0), -1)
#         out = self.emb(x)
#
#         print(out.shape)
#
#         out = self.conv_module(out)
#
#         dim = 1
#         for d in out.size()[1:]:
#             dim = dim * d
#         out = out.view(-1, dim)
#         print(dim)
#         out = self.fc_module(out)
#         return F.softmax(out, dim=1)


class DNNClassifier_custom(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
         super(DNNClassifier_custom, self).__init__()
         fc1 = nn.Linear(in_features=328, out_features=1000)
         fc2 = nn.Linear(in_features=1000, out_features=500)
         fc3 = nn.Linear(in_features=500, out_features=2)
         #fc4 = nn.Linear(in_features=100, out_features=2)

         self.fc_module = nn.Sequential(
             fc1,
             #nn.BatchNorm1d(1000),
             #nn.ReLU(),
             fc2,
             #nn.BatchNorm1d(500),
             #nn.ReLU(),
             fc3,
             #nn.BatchNorm1d(100),
             #nn.ReLU(),
             #fc4
         )

         # gpu로 할당
         if use_cuda:
             self.fc_module = self.fc_module.cuda()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc_module(x)
        return out


def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)

if __name__ == '__main__':
    dnn = DNNClassifier_custom()
    summary(dnn,(328,))

