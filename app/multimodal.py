import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from modelsummary import summary
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

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        # make linear
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        return out

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
        y = x.reshape(x.shape[0],-1)
        # |y| = (batch_size, n_classes)
        return y

class GateClassifier(nn.Module):
    def __init__(self,param_cnn,param_dnn):
        super(GateClassifier, self).__init__()
        self.cnn = CNNClassifier_custom(input_channel=14)
        self.dnn = DNNClassifier_custom()

        self.cnn.load_state_dict(state_dict=param_cnn, strict=False)
        self.dnn.load_state_dict(state_dict=param_dnn, strict=False)

        self.cnn.cuda()
        self.dnn.cuda()

        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()
        # self.W_cnn = nn.Linear(in_features=2200, out_features=2300)
        self.W_cnn = nn.Linear(in_features=17360, out_features=5000)
        self.W_dnn = nn.Linear(in_features=83968, out_features=5000)
        self.W_combined_inp = nn.Linear(in_features=10000, out_features=5000)
        self.generator = nn.Linear(5000, 2)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.layer_out = nn.LogSoftmax(dim=-1)

        if use_cuda:
            self.activation = self.activation.cuda()
            self.gate_activation = self.gate_activation.cuda()
            self.W_cnn = self.W_cnn.cuda()
            self.W_dnn = self.W_dnn.cuda()
            self.W_combined_inp = self.W_combined_inp.cuda()
            self.generator = self.generator.cuda()
            self.layer_out = self.layer_out.cuda()


    def forward(self, dnn_inp, cnn_inp):
        cnn_out = self.cnn(cnn_inp)
        dnn_out = self.dnn(dnn_inp)

        if use_cuda:
            cnn_out = cnn_out.cuda()
            dnn_out = dnn_out.cuda()

        w_cnn_out = self.W_cnn(cnn_out)
        w_dnn_out = self.W_dnn(dnn_out)

        del cnn_out
        del dnn_out

        combined_inp = torch.cat((w_cnn_out, w_dnn_out), 1)
        combined_inp = self.W_combined_inp(combined_inp)

        z = self.gate_activation(combined_inp)
        z_hat = 1 - z

        del combined_inp

        cnn_h = self.activation(w_cnn_out)
        dnn_h = self.activation(w_dnn_out)

        del w_cnn_out
        del w_dnn_out

        h = (cnn_h * z) + (dnn_h * z_hat)
        y = self.layer_out(self.generator(h))
        return y

class FusionClassifier(nn.Module):
    def __init__(self,param_cnn,param_dnn):
        super(FusionClassifier, self).__init__()
        self.cnn = CNNClassifier_custom(input_channel=14)
        self.dnn = DNNClassifier_custom()

        self.cnn.load_state_dict(state_dict=param_cnn, strict=False)
        self.dnn.load_state_dict(state_dict=param_dnn, strict=False)

        self.cnn.cuda()
        self.dnn.cuda()

        # self.W_cnn = nn.Linear(in_features=2200, out_features=2300)
        self.W_cnn = nn.Linear(in_features=17360, out_features=5000)
        self.W_dnn = nn.Linear(in_features=83968, out_features=5000)

        layer_in = nn.Linear(in_features=10000, out_features=5000)

        layer_out = nn.Linear(5000, 2)

        self.fc_module = nn.Sequential(
            layer_in,
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            layer_out
        )
        self.activation = nn.LogSoftmax(dim=-1)
        if use_cuda:
            self.W_cnn = self.W_cnn.cuda()
            self.W_dnn = self.W_dnn.cuda()
            self.fc_module = self.fc_module.cuda()
            self.activation = self.activation.cuda()

    def forward(self, dnn_inp, cnn_inp):
        cnn_out = self.cnn(cnn_inp)
        dnn_out = self.dnn(dnn_inp)

        if use_cuda:
            cnn_out = cnn_out.cuda()
            dnn_out = dnn_out.cuda()

        w_cnn_out = self.W_cnn(cnn_out)
        w_dnn_out = self.W_dnn(dnn_out)

        combined_inp = torch.cat((w_cnn_out, w_dnn_out), 1)

        out = self.activation(self.fc_module(combined_inp))
        return out

if __name__ == '__main__':
    a = torch.zeros((64, 328,), dtype=torch.float32).cuda()
    b = torch.zeros((64, 14, 2000)).cuda().cuda()
    cnn = GateClassifier()
    summary(cnn, a, b)