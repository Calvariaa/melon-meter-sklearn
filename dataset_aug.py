import csv
import random

with open('data_word.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)

# 将第一行作为特征名，后面的行作为数据
features = data[0]  # 标签名
data_values = data[1:]  # 数据

random.seed(1)  # 初始化随机数种子
random.shuffle(data_values)  # 原始数据过于整洁，shuffle list做打乱处理

# print(features)
# print(data_values)

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

# 将数据转化为int类型枚举
data_encoded = []
for d in data_values:
    encoded = [data_dict[k][v] for k, v in zip(features, d)]
    data_encoded.append(encoded)

# 划分训练集和测试集
# train_data = data_encoded[:4]
# test_data = data_encoded[4:]

train_data = data_encoded
test_data = data_encoded

# 计算先验概率
print("###### 统计训练集中好瓜与坏瓜数量")
total = len(train_data)
num_good = sum([d[-1] for d in train_data])
print("    好瓜: ", num_good)

num_bad = total - num_good
print("    坏瓜: ", num_bad)

# 没有对数据shuffle的时候发现的问题，全是好或者坏瓜会导致计算先验概率过程被0除
if num_good == 0 or num_bad == 0:
    print("请先确保训练集的好瓜和坏瓜数量均大于0\n训练与预测终止")
    exit(1)

p_good = num_good / total
p_bad = num_bad / total

# 计算条件概率
num_features = len(train_data[0]) - 1
p_good_features = [{} for _ in range(num_features)]
p_bad_features = [{} for _ in range(num_features)]

for d in train_data:
    for i in range(num_features):
        feature_value = d[i]
        label = d[-1]
        if feature_value not in p_good_features[i]:
            p_good_features[i][feature_value] = 0
            p_bad_features[i][feature_value] = 0
        if label == 0:
            p_good_features[i][feature_value] += 1
        else:
            p_bad_features[i][feature_value] += 1

for i in range(num_features):
    for key in p_good_features[i]:
        p_good_features[i][key] = p_good_features[i][key] / num_good
    for key in p_bad_features[i]:
        p_bad_features[i][key] = p_bad_features[i][key] / num_bad

# print(p_good_features)
# print(p_bad_features)

# 预测测试集
print("###### 通过测试集测试数据")
num_correct = 0
for d in test_data:
    p_good_given_features = p_good
    p_bad_given_features = p_bad
    for i in range(num_features):
        feature_value = d[i]
        if feature_value in p_good_features[i]:
            p_good_given_features *= p_good_features[i][feature_value]
        if feature_value in p_bad_features[i]:
            p_bad_given_features *= p_bad_features[i][feature_value]
    if p_good_given_features > p_bad_given_features:
        predicted_label = 0
    else:
        predicted_label = 1
    if predicted_label == d[-1]:
        num_correct += 1

    print("    [{0}]原始西瓜属性: {1}\n        原始数据: {2}, 贝叶斯算法预测: {3}, \n        好瓜概率: {4}, 坏瓜概率: {5}"
          .format("√" if d[6] == predicted_label else "×",
                  d,
                  "好瓜" if d[6] == 0 else "坏瓜",
                  "好瓜" if predicted_label == 0 else "坏瓜",
                  p_good_given_features,
                  p_bad_given_features))

print('###### 训练结果\n    预测模型精确度: ', num_correct / len(test_data))

# 以预测数据随机生成50份好瓜样例
good_examples = []
for i in range(50):
    example = []
    for j in range(num_features):
        feature_values = list(data_dict[features[j]].keys())
        feature_prob = list(p_good_features[j].values())
        feature = random.choices(feature_values, weights=feature_prob)[0]
        example.append(feature)
    example.append(1)
    good_examples.append(example)

print(good_examples)


# 以预测数据随机生成50份坏瓜样例
bad_examples = []
for i in range(50):
    example = []
    for j in range(num_features):
        feature_values = list(data_dict[features[j]].keys())
        feature_prob = list(p_bad_features[j].values())
        feature = random.choices(feature_values, weights=feature_prob)[0]
        example.append(feature)
    example.append(0)
    bad_examples.append(example)

print(bad_examples)

import csv

# 将生成的好瓜样例和坏瓜样例写入CSV文件中
with open('data_word_aug.csv', mode='w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜'])
    for example in good_examples:
        writer.writerow(example)
    for example in bad_examples:
        writer.writerow(example)