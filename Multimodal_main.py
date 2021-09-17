import app
import app.multimodal as Classifier
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
use_cuda = torch.cuda.is_available()

'''
'.header': 1, '.text': 2, '.data': 3, '.idata': 4, '.edata': 5, '.pdata': 6,
'.rsrc': 7, '.reloc': 8, '.rdata': 9, '.sdata': 10, '.xdata': 11,
'.tls': 12, 'Undefined': 13
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    index = np.arange(22059)
    np.random.shuffle(index)
    test_index = np.arange(9459)
    np.random.shuffle(test_index)

    e_X_train, e_y_train = app.load_pefile_data(fileListPath="dataset/trainedLabel.csv",
                                            folder="data/raw_header", w=1, MaxChunkLen=328,ind=index)
    e_X_test, e_y_test = app.load_pefile_data(fileListPath="dataset/testLabel.csv",
                                            folder="data/raw_header", w=1, MaxChunkLen=328,ind=test_index)

    print("Xtrain's shape : ",e_X_train.shape)
    print("Xtest's shape : ", e_X_test.shape)

    X_train, y_train = app.load_pefile_data(fileListPath="dataset/trainedLabel.csv",
                                            folder="data/structural_entropy_14section_sectionmapping", w=14, MaxChunkLen=2000,ind=index)
    X_test, y_test = app.load_pefile_data(fileListPath="dataset/testLabel.csv",
                                          folder="data/structural_entropy_14section_sectionmapping", w=14, MaxChunkLen=2000,ind=test_index)

    print("Xtrain's shape : ", X_train.shape)
    print("Xtest's shape : ", X_test.shape)

    trn_loader = app.getLoader(X_train, y_train)
    val_loader = app.getLoader(X_test, y_test)

    e_trn_loader = app.getLoader(e_X_train, e_y_train)
    e_val_loader = app.getLoader(e_X_test, e_y_test)

    cnn_weight = torch.load("model/cnn.pt")
    dnn_weight = torch.load("model/rnn.pt")

    #cnn_dnn = Classifier.GateClassifier(param_cnn=cnn_weight,param_dnn=dnn_weight)
    cnn_dnn = Classifier.FusionClassifier(param_cnn=cnn_weight,param_dnn=dnn_weight)
    criterion = nn.NLLLoss()
    learning_rate = 1e-5
    optimizer = optim.Adam(cnn_dnn.parameters(), lr=learning_rate)

    num_epochs = 10
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
            optimizer.zero_grad()
            model_output = cnn_dnn(e_x, x)
            loss = criterion(model_output, label)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()
            del loss
            del model_output

            # 학습과정 출력
            if (i + 1) % 86 == 0:
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
            val_output = cnn_dnn(e_val_x, val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))
    torch.save(cnn_dnn.state_dict(),"model/Fusionmodel_rnn.pt")