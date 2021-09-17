import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

# 데이터의 80%는 훈련데이터
trainRatio = 0.8

def load_pefile_data(fileListPath, folder , w ,MaxChunkLen,ind):
    unfoundedFiles = 0
    df = pd.read_csv(fileListPath, sep=',',header=None)
    # numpy array, 2D,
    malware_name_label = df.values

    #malware_name_label = df.head(3).values
    mixed_malware_name_label = malware_name_label

    dirTargetHaar2D = "/".join([os.getcwd(),folder])
    filesLen = len(mixed_malware_name_label)
    data_with_padding = np.zeros((filesLen, MaxChunkLen, w))
    y_label_number = np.zeros(filesLen)
    index = 0

    for entryIndex in tqdm(range(len(tqdm(mixed_malware_name_label)))):
        fetched_name_label = mixed_malware_name_label[entryIndex]
        name_with_extension = fetched_name_label[1]
        pathTargetHaar2D = os.path.join(dirTargetHaar2D, name_with_extension)
        try:
            df_haar = pd.read_csv(pathTargetHaar2D, sep=',',header=None,index_col=None)
            data_non_pad = df_haar.values[:,:w]

            if len(data_non_pad) < MaxChunkLen:
                tp = MaxChunkLen - len(data_non_pad)
                padArray = np.zeros((tp, w))
                data_non_pad = np.vstack((data_non_pad, padArray))

            else:
                data_non_pad = data_non_pad[:MaxChunkLen]
            data_with_padding[index] = data_non_pad
            y_label_number[index] = mixed_malware_name_label[entryIndex][2]
            index += 1

        except FileNotFoundError:
            print("File does not exist: " + name_with_extension)

            unfoundedFiles += 1

    if unfoundedFiles != 0:
        print('delete')
        data_with_padding = data_with_padding[: filesLen - unfoundedFiles]
        y_label_number = y_label_number[:filesLen - unfoundedFiles]

    x = data_with_padding.reshape(data_with_padding.shape[0],
                                  data_with_padding.shape[2],
                                  data_with_padding.shape[1])
    y = y_label_number
    print(x.shape)

    x = Shuffle(x,ind)
    y = Shuffle(y,ind)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

def t_sne(X,y):
    y = np.array(y)
    X = np.array(X)

    print(len(X))

    tsne = TSNE(random_state=0)
    digits_tsne = tsne.fit_transform(X)


    colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
              '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

    for i in range(len(X)):  # 0부터  digits.data까지 정수
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(y[i]), color=colors[y[i]],
                 fontdict={'weight': 'bold', 'size': 9}
                 )
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())  # 최소, 최대
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())  # 최소, 최대
    plt.xlabel('t-SNE 0')  # x축 이름
    plt.ylabel('t-SNE 1')  # y축 이름
    plt.show()  # 그래프 출력

def getLoader(X, y):
    set = TensorDataset(X, y)
    loader = DataLoader(set, batch_size=256, shuffle=True)
    return loader

def Shuffle(l,index):
    return l[index]

if __name__ == '__main__':
    '''
    X_train, y_train = load_pefile_data(fileListPath="../dataset/trainedLabel.csv",
                                            folder="../data/structural_entropy_14section", w=1, MaxChunkLen=100)
    X_test, y_test = load_pefile_data(fileListPath="../dataset/testLabel.csv",
                                           folder="../data/structural_entropy_14section", w=1, MaxChunkLen=100)
    '''
    X_train, y_train = load_pefile_data(fileListPath="../dataset/trainedLabel.csv",
                                        folder="../data/raw_header", w=1, MaxChunkLen=256)

    X_test, y_test = load_pefile_data(fileListPath="../dataset/testLabel.csv",
                                      folder="../data/raw_header", w=1, MaxChunkLen=256)