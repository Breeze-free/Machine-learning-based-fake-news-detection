import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本特征抽取
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("./train.news.csv")
data2 = pd.read_csv("./test.news.csv")
N = 20


df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
print(df)
lst1 = df.values.tolist()
print(lst1)


def cutword(text):
    return " ".join(list(jieba.cut(text)))





if __name__ == "__main__":
    #tfidf(data, data2)
    print(123)
