import joblib

features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
data_dict = {'色泽': {'浅白': 0, '青绿': 1, '乌黑': 2}, '根蒂': {'硬挺': 0, '稍蜷': 1, '蜷缩': 2}, '敲声': {'清脆': 0, '浊响': 1, '沉闷': 2},
             '纹理': {'模糊': 0, '清晰': 1, '稍糊': 2}, '脐部': {'平坦': 0, '稍凹': 1, '凹陷': 2}, '触感': {'硬滑': 0, '软粘': 1},
             '好瓜': {'0': 0, '1': 1}}

clf = joblib.load('melon_detect_1.pkl')

data = []
for i in range(6):
    print("请输入第{}个属性[{}], 你有{}个选项，他们分别是{}："
          .format(i + 1, features[i], len(data_dict[features[i]]),
                  list(data_dict[features[i]].keys())))
    attr = input("  >>".format(i + 1))
    data.append(attr)

data_encoded = []
encoded = [data_dict[k][v] for k, v in zip(features, data)]
data_encoded.append(encoded)

# 预测
X_test = data_encoded
predicted_label = clf.predict(X_test)
print("[√]这个西瓜是{0}瓜".format("好" if predicted_label == 0 else "坏"))
