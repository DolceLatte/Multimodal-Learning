from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelsummary import summary

use_cuda = torch.cuda.is_available()


class CNN(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNN, self).__init__()
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

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        out = out.view(out.size(0), -1)
        return out


class DNN(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(DNN, self).__init__()
        fc1 = nn.Linear(in_features=328, out_features=1000)
        fc2 = nn.Linear(in_features=1000, out_features=500)
        # fc3 = nn.Linear(in_features=500, out_features=2)
        self.fc_module = nn.Sequential(
            fc1,
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            fc2,
            # nn.BatchNorm1d(500),
            # nn.ReLU(),
            # fc3,
            # nn.BatchNorm1d(100),
            # nn.ReLU(),
            # fc4
        )

        # gpu로 할당
        if use_cuda:
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc_module(x)
        return out

class CDNNClassifier(nn.Module):
    def __init__(self, param_cnn=None, param_dnn=None):
        super(CDNNClassifier, self).__init__()
        self.cnn = CNN()
        self.dnn = DNN()

        #self.cnn.load_state_dict(state_dict=param_cnn, strict=False)
        #self.dnn.load_state_dict(state_dict=param_dnn, strict=False)

        self.cnn.cuda()
        self.dnn.cuda()

        #for p in self.dnn.parameters():
            #p.requires_grad = False

        #for p in self.cnn.parameters():
            #p.requires_grad = False

        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()

        # self.W_cnn = nn.Linear(in_features=2200, out_features=2300)
        self.W_cnn = nn.Linear(in_features=31360, out_features=5000)
        self.W_dnn = nn.Linear(in_features=500, out_features=5000)

        self.W_combined_inp = nn.Linear(in_features=10000, out_features=5000)
        self.B = nn.BatchNorm1d(5000, affine=True)

        layer_out1 = nn.Linear(5000, 6000)
        #layer_out1 = nn.Linear(5000, 2)
        layer_out2 = nn.Linear(6000, 2000)
        layer_out3 = nn.Linear(2000, 300)
        layer_out4 = nn.Linear(300, 2)

        self.fc_module = nn.Sequential(
            layer_out1,
            nn.LayerNorm(normalized_shape=6000,eps=1e-5,elementwise_affine=True),
            #nn.BatchNorm1d(6000, affine=True),
            ##nn.Tanh(),
            nn.LeakyReLU(),
            layer_out2,
            nn.LayerNorm(normalized_shape=2000, eps=1e-5, elementwise_affine=True),
            #nn.BatchNorm1d(2000, affine=True),
            #nn.Tanh(),
            nn.LeakyReLU(),
            layer_out3,
            nn.LayerNorm(normalized_shape=300, eps=1e-5, elementwise_affine=True),
            #nn.BatchNorm1d(300, affine=True),
            #nn.Tanh(),
            nn.LeakyReLU(),
            layer_out4,
        )

        if use_cuda:
            self.activation = self.activation.cuda()
            self.gate_activation = self.gate_activation.cuda()
            self.W_cnn = self.W_cnn.cuda()
            self.W_dnn = self.W_dnn.cuda()
            self.W_combined_inp = self.W_combined_inp.cuda()
            self.fc_module = self.fc_module.cuda()
            self.B = self.B.cuda()

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

        w_cnn_out = self.B(w_cnn_out)
        w_dnn_out = self.B(w_dnn_out)

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
        # h = (cnn_h * z) + (dnn_h * z)

        out = self.fc_module(h)
        return F.softmax(out), z


def print_grad(parameters):
    for p in parameters:
        print(p)
        return

if __name__ == '__main__':
    torch.cuda.empty_cache()
    cnn = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/cnn.pth")
    #cnnOpt = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/cnnOpt.pth")

    dnn = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/dnn_noactivate.pth")
    cnn_dnn = CDNNClassifier(cnn,dnn)

    a = torch.zeros((64,328,),dtype=torch.float32).cuda()
    b = torch.zeros((64,17,3600)).cuda().cuda()

    print_grad(cnn_dnn.parameters())

    summary(cnn_dnn,a,b)