import csv
import random
from sklearn.tree import DecisionTreeClassifier
import joblib

with open('data_word_aug.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)

# 将第一行作为特征名，后面的行作为数据
features = data[0]  # 标签名
data_values = data[1:]  # 数据

random.seed(1)  # 初始化随机数种子
random.shuffle(data_values)  # 原始数据过于整洁，shuffle list做打乱处理

# 将数据转换为字典格式
data_dict = {}
for feature_index, feature_name in enumerate(features):
    feature_values = {}
    for row in data_values:
        feature_value = row[feature_index]
        if feature_value not in feature_values:
            feature_values[feature_value] = len(feature_values)
    data_dict[feature_name] = feature_values

# print(data_dict)

data_encoded = []
for d in data_values:
    encoded = [data_dict[k][v] for k, v in zip(features, d)]
    data_encoded.append(encoded)

# 划分训练集和测试集 50 : 50分
train_data = data_encoded[:50]
test_data = data_encoded[50:]

# 训练模型
X_train = [d[:-1] for d in train_data]  # 特征
y_train = [d[-1] for d in train_data]  # 标签
# print(X_train)
# print(y_train)
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)

joblib.dump(clf, 'melon_detect_1.pkl')

# 预测测试集
print("###### 通过测试集测试数据")
num_correct = 0
for d in test_data:
    X_test = [d[:-1]]
    predicted_label = clf.predict(X_test)
    if predicted_label == d[-1]:
        num_correct += 1

    print("    [{0}]原始西瓜属性: {1}\n        原始数据: {2}, 决策树算法预测: {3}"
          .format("√" if d[6] == predicted_label else "×",
                  d,
                  "好瓜" if d[6] == 0 else "坏瓜",
                  "好瓜" if predicted_label == 0 else "坏瓜"))

print('###### 模型训练结果\n    预测模型精确度: ', num_correct / len(test_data))
