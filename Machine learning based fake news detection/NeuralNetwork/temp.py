import time
import torch
import numpy as np
import pandas as pd
import jieba
from sklearn.metrics import classification_report

from train_eval import train, init_network
from importlib import import_module
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame


def cutword(text):  # 去停用词的分词函数
    init_cutwordlist = list(jieba.cut(text))
    final_cutword = " "
    for word in init_cutwordlist:
        if word not in stopwords and word!=' ':
            final_cutword += word + " "
    return final_cutword


def fun1(x):
    return " ".join(x)


if __name__ == '__main__':


    f = open(r"D:\codeing\pycoding\DeepLearning\Dataset\data\sgns.sogounews.bigram-char", encoding="UTF-8")
    t=f.readline().split()
    n, dimension = int(t[0]), int(t[1])
    print(dimension)


    chinesewordvec = f.readlines()
    chinesewordvec = [i.split() for i in chinesewordvec]
    vectorsmap = []
    wordtoindex = indextoword = {}
    for i in range(n):
        vectorsmap.append(list(map(float, chinesewordvec[i][len(chinesewordvec[i]) - dimension:])))
        wordtoindex[chinesewordvec[i][0]]=i
        indextoword[i] = chinesewordvec[i][0]
    f.close()
    print(123)
    arry2=[len(auto.split()) for auto in x_train1["new"].tolist()]
    print(np.median(arry2))   # 14
    print(np.mean(arry2))     # 24
    print(123)
