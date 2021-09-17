import app
import app.Classifier as Classifier
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from collections import Counter
use_cuda = torch.cuda.is_available()
import numpy as np
'''
'.header': 1, '.text': 2, '.data': 3, '.idata': 4, '.edata': 5, '.pdata': 6,
'.rsrc': 7, '.reloc': 8, '.rdata': 9, '.sdata': 10, '.xdata': 11,
'.tls': 12, 'Undefined': 13
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_channel = 1
    test_index = np.arange(9459)
    np.random.shuffle(test_index)

    X_test, y_test = app.load_pefile_data(fileListPath="dataset/testLabel.csv",
                                            folder="data/raw_header", w=input_channel, MaxChunkLen=328, ind= test_index)

    print("Xtest's shape : ", X_test.shape)
    val_loader = app.getLoader(X_test,y_test)

    #cnn = Classifier.CNNClassifier_custom(input_channel=input_channel)
    cnn = Classifier.DNNClassifier_custom()
    cnn.load_state_dict(torch.load("model/dnn.pt"))

    cnn.eval()
    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            if use_cuda:
                val_x = val_x.cuda()
                val_label = val_label.cuda()
            val_output = cnn(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))