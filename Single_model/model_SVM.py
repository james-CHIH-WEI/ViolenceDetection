from sklearn import svm
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pre_train_model(data_folder):
    model_path = "./export_data/" + data_folder + "/train/SVM_v2.m"

    if not os.path.isfile(model_path):
        df = pd.read_csv("./export_data/" + data_folder +
                         "/train/new_value_v2.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X)

        SVM = svm.SVC(C=1, kernel='poly', gamma='auto', max_iter=100000)
        SVM.fit(X, y.values.ravel())

        joblib.dump(SVM, model_path)
    else:
        SVM = joblib.load(model_path)
    return SVM


SVM = pre_train_model("train_test_data_balance_2")


df = pd.read_csv(
    "./export_data/train_test_data_balance_2/train/new_value_v2.csv")

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


# scaler = StandardScaler()
# X_train = scaler.fit(X_train).transform(X_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)


# 載入模型
SVM = svm.SVC(C=1, kernel='poly', gamma='auto', max_iter=100000)
SVM.fit(X_train, y_train.values.ravel())


# result_test = {
#     "center_distance": [0],
#     "center_chanege": [100.289703],
#     "wrist_distance": [-14.071133],
#     "wrist_acceleration": [45.566804],
#     "ankle_distance": [23.0],
#     "ankle_acceleration": [0.0],
# }
# result_test_df = pd.DataFrame(result_test)


result_test = [
    0,
    100.289703,
    -14.071133,
    45.566804,
    23.0,
    0.0,
]
result_test = np.array(result_test)
# result_test = result_test.reshape(-1, 1)

# 預測
SVM_predict = SVM.predict(result_test)
print(SVM_predict)

# # 模型效能指標
# SVM_acc = accuracy_score(y_test, SVM_predict)
# SVM_f1 = f1_score(y_test, SVM_predict)

# print("-------SVM-------")
# print("acc={}".format(SVM_acc))
# print("f1={}".format(SVM_f1))


# lg_conf_matrix = confusion_matrix(y_test, SVM_predict)
# print(lg_conf_matrix)
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.heatmap(lg_conf_matrix, annot=True, fmt="g")
# plt.show()
