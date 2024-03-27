from eli5.sklearn import PermutationImportance
import eli5
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./export_data/train_test_data_balance_2/train/value.csv")

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
    X, y, test_size=0.9, random_state=42)


model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=2)
model.fit(X_train, y_train.values.ravel())

# 預測
predict = model.predict(X_test)

# 模型效能指標
acc = accuracy_score(y_test, predict)
f1 = f1_score(y_test, predict)

print("-------DecsionTree-------")
print("acc={}".format(acc))
print("f1={}".format(f1))


lg_conf_matrix = confusion_matrix(y_test, predict)
print(lg_conf_matrix)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(lg_conf_matrix, annot=True, fmt="g")
plt.show()


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
# eli5.show_weights(perm, feature_names=X_test.columns.tolist())
print(eli5.format_as_text(eli5.explain_weights(perm)))
