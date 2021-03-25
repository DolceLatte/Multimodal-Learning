import app
import torch
import app.Multimodal_Classifier as Classifier
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

use_cuda = torch.cuda.is_available()

def Shuffle(l,index):
    return l[index]

def load_multimodal_data():
    l, X_train, y_train = app.load_pefile_data("./label/trainedLabel.csv", "/dataset/dataset_17", w=17,
                                               MaxChunkLen=3600)
    l, X_test, y_test = app.load_pefile_data("./label/testLabel.csv", "/dataset/dataset_17", w=17, MaxChunkLen=3600)

    e_l, e_X_train, e_y_train = app.load_pefile_data("./label/trainedLabel.csv", "/dataset/dataset_e_named", w=1,
                                                     MaxChunkLen=328)
    e_l, e_X_test, e_y_test = app.load_pefile_data("./label/testLabel.csv", "/dataset/dataset_e_named", w=1,
                                                   MaxChunkLen=328)

    e_X_train = np.array(e_X_train)
    e_X_train = e_X_train.reshape(len(e_X_train), (328))

    e_X_test = np.array(e_X_test)
    e_X_test = e_X_test.reshape(len(e_X_test), (328))

    index = np.arange(X_train.shape[0])
    np.random.shuffle(index)

    test_index = np.arange(X_test.shape[0])
    np.random.shuffle(test_index)

    X_train = Shuffle(X_train,index)
    y_train = Shuffle(y_train,index)
    X_test = Shuffle(X_test,test_index)
    y_test = Shuffle(y_test,test_index)

    e_X_train = Shuffle(e_X_train, index)
    e_y_train = Shuffle(e_y_train, index)
    e_X_test = Shuffle(e_X_test, test_index)
    e_y_test = Shuffle(e_y_test, test_index)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    e_X_train = torch.tensor(e_X_train, dtype=torch.float32)
    e_y_train = torch.tensor(e_y_train, dtype=torch.int64)
    e_X_test = torch.tensor(e_X_test, dtype=torch.float32)
    e_y_test = torch.tensor(e_y_test, dtype=torch.int64)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    e_trainset = TensorDataset(e_X_train, e_y_train)
    e_testset = TensorDataset(e_X_test, e_y_test)

    trn_loader = DataLoader(trainset, batch_size=64, shuffle=False)
    val_loader = DataLoader(testset, batch_size=64, shuffle=False)

    e_trn_loader = DataLoader(e_trainset, batch_size=64, shuffle=False)
    e_val_loader = DataLoader(e_testset, batch_size=64, shuffle=False)

    return trn_loader , val_loader ,e_trn_loader , e_val_loader

def main():
    trn_loader , val_loader ,e_trn_loader , e_val_loader = load_multimodal_data()

    #cnn = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/cnn.pth")
    cnnOpt = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/cnnOpt.pth")

    dnn = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/dnn.pth")
    dnn_2200 = torch.load("C:/Users/김정우/PycharmProjects/Malware_Classification/model/dnn_2200.pth")

    cnn_dnn = Classifier.CDNNClassifier(cnnOpt, dnn)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(cnn_dnn.parameters(), lr=learning_rate, weight_decay = 1e-7)

    num_epochs = 12
    num_batches = len(trn_loader)

    trn_loss_list = []
    val_loss_list = []

    cnn_dnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i, (data, e_data) in enumerate(zip(trn_loader, e_trn_loader)):
            x, label = data
            e_x, e_label = e_data

            if use_cuda:
                x = x.cuda()
                e_x = e_x.cuda()
                label = label.cuda()

            x = x.transpose(1, 2)
            optimizer.zero_grad()
            model_output = cnn_dnn(e_x, x)
            loss = criterion(model_output, label)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()
            del loss
            del model_output

            # 학습과정 출력
            if (i + 1) % 236 == 0:
                # every 100 mini-batches
                with torch.no_grad():
                    val_loss = 0.0
                    for j, (val, e_val) in enumerate(zip(val_loader, e_val_loader)):
                        val_x, val_label = val
                        e_val_x, _ = e_val
                        if use_cuda:
                            val_x = val_x.cuda()
                            e_val_x = e_val_x.cuda()
                            val_label = val_label.cuda()
                        val_x = val_x.transpose(1, 2)

                        val_output = cnn_dnn(e_val_x, val_x)
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

    cnn_dnn.eval()
    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, (val, e_val) in enumerate(zip(val_loader, e_val_loader)):
            val_x, val_label = val
            e_val_x, _ = e_val
            if use_cuda:
                val_x = val_x.cuda()
                e_val_x = e_val_x.cuda()
                val_label = val_label.cuda()
            val_x = val_x.transpose(1, 2)
            val_output = cnn_dnn(e_val_x, val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))
    #torch.save(cnn_dnn.state_dict(), "./model/multimodal_Cnn_dnn_smallHiddnV.pth", )

if __name__ == '__main__':
    main()
