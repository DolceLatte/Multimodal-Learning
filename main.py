import app
import torch
import app.Classifier as Classifier
import app.DNN_Classifier as DNN_Classifier
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

use_cuda = torch.cuda.is_available()

def main():
    torch.cuda.empty_cache()
    torch.cuda.get_device_name(0)

    _ , X_train, y_train = app.load_pefile_data("./label/trainedLabel.csv", "/dataset/dataset_17",w=17,MaxChunkLen=3600)
    _ , X_test, y_test = app.load_pefile_data("./label/testLabel.csv", "/dataset/dataset_17",w=17,MaxChunkLen=3600)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    trn_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(testset, batch_size=64, shuffle=True)

    cnn = Classifier.CNNClassifier_custom()
    Classifier.weight_init(cnn)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    num_epochs = 10
    num_batches = len(trn_loader)

    trn_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i, data in enumerate(trn_loader):
            x, label = data
            if use_cuda:
                x = x.cuda()
                label = label.cuda()
            x = x.transpose(1, 2)
            # grad init
            optimizer.zero_grad()
            # forward propagation
            model_output = cnn(x)
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
            # del (memory issue)
            del loss
            del model_output

            # 학습과정 출력
            if (i + 1) % 100 == 0:
                # every 100 mini-batches
                with torch.no_grad():  # very very very very important!!!
                    val_loss = 0.0
                    for j, val in enumerate(val_loader):
                        val_x, val_label = val
                        if use_cuda:
                            val_x = val_x.cuda()
                            val_label = val_label.cuda()
                        val_x = val_x.transpose(1, 2)
                        val_output = cnn(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss
                del val_output
                del v_loss

                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)
                ))

                trn_loss_list.append(trn_loss / 100)
                val_loss_list.append(val_loss / len(val_loader))
                trn_loss = 0.0

    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            if use_cuda:
                val_x = val_x.cuda()
                val_label = val_label.cuda()
            val_x = val_x.transpose(1, 2)
            val_output = cnn(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))
    torch.save(cnn.state_dict(), "./model/cnn_11150.pth")


def DNN_main():
    torch.cuda.empty_cache()
    torch.cuda.get_device_name(0)

    _ , X_train, y_train = app.load_pefile_data("./label/trainedLabel_e.csv", "/dataset/dataset_e", w=1, MaxChunkLen=328)
    _ , X_test, y_test = app.load_pefile_data("./label/testLabel_e.csv", "/dataset/dataset_e", w=1, MaxChunkLen=328)

    X_train = np.array(X_train)
    X_train = X_train.reshape(len(X_train),(328))

    X_test = np.array(X_test)
    X_test = X_test.reshape(len(X_test), (328))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    trn_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(testset, batch_size=64, shuffle=True)

    dnn = DNN_Classifier.DNNClassifier_custom()
    DNN_Classifier.weight_init(dnn)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3

    optimizer = optim.Adam(dnn.parameters(), lr=learning_rate)

    num_epochs = 10
    num_batches = len(trn_loader)

    trn_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i, data in enumerate(trn_loader):
            x, label = data
            if use_cuda:
                x = x.cuda()
                label = label.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            model_output = dnn(x)
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
            # del (memory issue)
            del loss
            del model_output

            # 학습과정 출력
            if (i + 1) % 100 == 0:
                # every 100 mini-batches
                with torch.no_grad():  # very very very very important!!!
                    val_loss = 0.0
                    for j, val in enumerate(val_loader):
                        val_x, val_label = val
                        if use_cuda:
                            val_x = val_x.cuda()
                            val_label = val_label.cuda()
                        val_output = dnn(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss
                del val_output
                del v_loss

                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)
                ))

                trn_loss_list.append(trn_loss / 100)
                val_loss_list.append(val_loss / len(val_loader))
                trn_loss = 0.0

    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            if use_cuda:
                val_x = val_x.cuda()
                val_label = val_label.cuda()
            val_output = dnn(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))

    #torch.save(dnn.state_dict(), "./model/dnn_noactivate.pth",)

if __name__ == '__main__':
    DNN_main()
    main()