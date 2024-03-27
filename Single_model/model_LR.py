from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("./export_data/mix/value.csv")

# lb = LabelEncoder()
# for index, value in enumerate(df.dtypes):
#     if value == object:
#         lb.fit(df[df.columns[index]].drop_duplicates())
#         df[df.columns[index]] = lb.transform(df[df.columns[index]])


# 選擇需要欄位
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

# 切出訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=22)


# 載入模型
LR = linear_model.LogisticRegression()
LR.fit(X_train, y_train.values.ravel())


# 預測
LR_predict = LR.predict(X_test)

# 模型效能指標
LR_acc = accuracy_score(y_test, LR_predict)
LR_f1 = f1_score(y_test, LR_predict)

print("-------LR-------")
print("acc={}".format(LR_acc))
print("f1={}".format(LR_f1))
