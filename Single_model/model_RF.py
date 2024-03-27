from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = {"part": [2],
     "distance": [16.970563],
     "limit": [71.046151],
     "chanege_distance_between": [0.0],
     "acceleration": [12.083046]}


w = pd.DataFrame(x)
print(w)

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
    X, y, test_size=0.3, random_state=42)


# 建立 Random Forest Classifier 模型
RF = RandomForestClassifier(n_estimators=10, random_state=42)
# 使用訓練資料訓練模型
RF.fit(X_train, y_train.values.ravel())


# 使用訓練資料預測分類
RF_predict = RF.predict(X_test)

# 模型效能指標
RF_acc = accuracy_score(y_test, RF_predict)
RF_f1 = f1_score(y_test, RF_predict)

print("-------DecsionTree-------")
print("acc={}".format(RF_acc))
print("f1={}".format(RF_f1))
