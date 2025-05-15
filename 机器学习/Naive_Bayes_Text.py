'''
请手动实现（不准调用任何现成的机器学习工具包中的朴素贝叶斯分类器）朴素贝叶斯分类器算法（包括概率平滑方法），并在ppt中列出的D14数据集进行训练和验证：
将D14数据集随机打乱后，取10个样例为训练集，另外4个测试集；输出测试结果。
'''


import random
from collections import defaultdict

# 数据集 D14
D14 = [
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Weak","PlayTennis":"No"},
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Strong","PlayTennis":"No"},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"High","Wind":"Weak","PlayTennis":"Yes"},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"High","Wind":"Weak","PlayTennis":"Yes"},
    {"Outlook":"Rain","Temperature":"Cool","Humidity":"Normal","Wind":"Weak","PlayTennis":"Yes"},
    {"Outlook":"Rain","Temperature":"Cool","Humidity":"Normal","Wind":"Strong","PlayTennis":"No"},
    {"Outlook":"Overcast","Temperature":"Cool","Humidity":"Normal","Wind":"Strong","PlayTennis":"Yes"},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"High","Wind":"Weak","PlayTennis":"No"},
    {"Outlook":"Sunny","Temperature":"Cool","Humidity":"Normal","Wind":"Weak","PlayTennis":"Yes"},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"Normal","Wind":"Weak","PlayTennis":"Yes"},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Wind":"Strong","PlayTennis":"Yes"},
    {"Outlook":"Overcast","Temperature":"Mild","Humidity":"High","Wind":"Strong","PlayTennis":"Yes"},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"Normal","Wind":"Weak","PlayTennis":"Yes"},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"High","Wind":"Strong","PlayTennis":"No"},
]


# 将数据集随机打乱
random.shuffle(D14)

# 分割数据集为训练集和测试集
train_set = D14[:10]
test_set = D14[10:14]



#手动实现朴素贝叶斯分类器算法（包括概率平滑方法）
def train_naive_bayes(data):
    # 初始化先验概率、条件概率和类别集合
    prior = defaultdict(float)
    cond_prob = defaultdict(lambda: defaultdict(float))
    classes = set()

    # 计算先验概率
    total_count = len(data)
    for instance in data:
        classes.add(instance["PlayTennis"])
        prior[instance["PlayTennis"]] += 1
    
    for cls in classes:
        prior[cls] /= total_count

    # 计算条件概率
    attr_values = defaultdict(set)
    for instance in data:
        for attr in instance.keys():
            if attr != "PlayTennis":
                attr_values[attr].add(instance[attr])
    
    for attr in attr_values.keys():
        for attr_value in attr_values[attr]:
            for cls in classes:
                # 计算特征值在每个类别下的条件概率，使用拉普拉斯平滑
                count = sum(1 for instance in data if instance[attr] == attr_value and instance["PlayTennis"] ==cls)
                cond_prob[attr][(attr_value, cls)] = (count + 1) / (prior[cls] * total_count + len(attr_values[attr]))

    
    return prior, cond_prob, classes

# 预测朴素贝叶斯分类器
def predict_naive_bayes(instance, prior, cond_prob, classes):
    # 初始化最大后验概率和预测类别
    max_prob = -1
    preb_cls = None
    for cls in classes:
        prob = prior[cls]
        for attr in instance.keys():
            if attr != "PlayTennis":
                # 计算特征对应类别的后验概率
                prob *= cond_prob[attr][(instance[attr], cls)]

        # 选择具有最大后验概率的类别作为预测结果
        if prob > max_prob:
            max_prob = prob
            pred_cls = cls
    
    return pred_cls


# 训练分类器
prior, cond_prob, classes = train_naive_bayes(train_set)

# 测试分类器
correct_count = 0
for instance in test_set:
    pred_cls = predict_naive_bayes(instance, prior, cond_prob, classes)
    if pred_cls == instance["PlayTennis"]:
        correct_count += 1

accuracy = correct_count / len(test_set)
print("测试集准确率为：", accuracy)



