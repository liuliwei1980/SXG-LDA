# -*- coding:utf-8 -*-
"""
作者：Tan
日期:2023年07月22日
"""
# 导入所需的库和模块
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.decomposition import FastICA, PCA, FactorAnalysis

# 加载训练集和测试集的特征数据和标签数据。
x_train = np.loadtxt('./data/features_train.txt')
y_train = np.loadtxt('./data/label_train.txt')
x_test = np.loadtxt('./data/features_test.txt')
y_test = np.loadtxt('./data/label_test.txt')
# x_train = np.loadtxt('./example/features_train.txt')
# y_train = np.loadtxt('./example/label_train.txt')
# x_test = np.loadtxt('./example/features_test.txt')
# y_test = np.loadtxt('./example/label_test.txt')

#######################################################################################
# 使用ICA进行降维
ica = FastICA(n_components=2, random_state=0)
x_ica = ica.fit_transform(x_train)
# 使用PCA进行降维
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_train)
# 使用因子分析(FA)进行降维
fa = FactorAnalysis(n_components=2)
x_fa = fa.fit_transform(x_train)
# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
x_tsne = tsne.fit_transform(x_train)
# 创建子图，用于可视化
plt.figure(figsize=(12, 12))
# 定义不同类别的标记样式
class0_marker = 'v'  # 三角形标记
class1_marker = 'p'  # 正方形标记
# 绘制ICA降维后的图，并根据类别标记不同颜色和形状
plt.subplot(221)
plt.scatter(x_ica[y_train == 0, 0], x_ica[y_train == 0, 1], c='blue', marker=class0_marker, label='relevant', s=15)
plt.scatter(x_ica[y_train == 1, 0], x_ica[y_train == 1, 1], c='black', marker=class1_marker, label='irrelevant', s=15)
plt.title('Independent component analysis (ICA)')
plt.legend()

# 绘制PCA降维后的图，并根据类别标记不同颜色和形状
plt.subplot(222)
plt.scatter(x_pca[y_train == 0, 0], x_pca[y_train == 0, 1], c='blue', marker=class0_marker, label='relevant', s=15)
plt.scatter(x_pca[y_train == 1, 0], x_pca[y_train == 1, 1], c='black', marker=class1_marker, label='irrelevant', s=15)
plt.title('Principal component analysis (PCA)')
plt.legend()

# 绘制FA降维后的图，并根据类别标记不同颜色和形状
plt.subplot(223)
plt.scatter(x_fa[y_train == 0, 0], x_fa[y_train == 0, 1], c='blue', marker=class0_marker, label='relevant', s=15)
plt.scatter(x_fa[y_train == 1, 0], x_fa[y_train == 1, 1], c='black', marker=class1_marker, label='irrelevant', s=15)
plt.title('Factor Analysis (FA)')
plt.legend()

# 绘制t-SNE降维后的图，并根据类别标记不同颜色和形状
plt.subplot(224)
plt.scatter(x_tsne[y_train == 0, 0], x_tsne[y_train == 0, 1], c='blue', marker=class0_marker, label='relevant', s=15)
plt.scatter(x_tsne[y_train == 1, 0], x_tsne[y_train == 1, 1], c='black', marker=class1_marker, label='irrelevant', s=15)
plt.title('T-distributed Stochastic Neighbor Embedding (t-SNE)')
plt.legend()
plt.show()
#######################################################################################


# 定义了5个基分类器(base_model1到base_model5)和一个元分类器(meta_model)。

# 基分类器使用随机森林(Random Forest)算法
base_model1 = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=100)
base_model2 = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=100)
base_model3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=100)
base_model4 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=100)
base_model5 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=100)

#################################################################################################
# # 基分类器使用KNN算法
# base_model1 = KNeighborsClassifier(n_neighbors=5)
# base_model2 = KNeighborsClassifier(n_neighbors=10)
# base_model3 = KNeighborsClassifier(n_neighbors=15)
# base_model4 = KNeighborsClassifier(n_neighbors=20)
# base_model5 = KNeighborsClassifier(n_neighbors=25)
#
# # 基分类器使用决策树(DecisionTreeClassifier)算法
# base_model1 = DecisionTreeClassifier(random_state=100)
# base_model2 = DecisionTreeClassifier(random_state=100)
# base_model3 = DecisionTreeClassifier(max_depth=5, random_state=100)
# base_model4 = DecisionTreeClassifier(max_depth=5, random_state=100)
# base_model5 = DecisionTreeClassifier(max_depth=10, random_state=100)

# # 基分类器使用GDBT算法
# base_model1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=100)
# base_model2 = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=100)
# base_model3 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=100)
# base_model4 = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=100)
# base_model5 = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=100)
#################################################################################################

# 定义SVM分类器对象
# meta_model = SVC(kernel='linear', C=1.0, probability=True, random_state=100)

#################################################################################################
# 元分类器使用随机森林
# meta_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=100)

# 元分类器使用GDBT
# meta_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=100)

# 元分类器使用LogisticRegression
# meta_model = LogisticRegression(solver='liblinear', random_state=100)

# 元分类器使用KNN
# meta_model = KNeighborsClassifier(n_neighbors=5)

# 元分类器使用lightgbm
# meta_model = lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=100)

# 元分类器使用XGBoost
meta_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=100)

#################################################################################################
# 创建一个StackingCVClassifier对象(stack)作为堆叠集成模型。
# 其中，classifiers参数指定了使用的基分类器列表，meta_classifier参数指定了元分类器，random_state参数用于控制随机数生成器的种子，
# use_probas参数指定是否使用分类器的预测概率进行堆叠，cv参数指定交叉验证的折数。
stack = StackingCVClassifier(
    classifiers=[base_model1, base_model2, base_model3, base_model4, base_model5],
    meta_classifier=meta_model, random_state=10, use_probas=True, cv=5)

# 定义变量和数组，包括n_folds(折数)、kf(KFold对象)、i(计数器)、acc_stack(用于保存每折交叉验证的准确率)、mcc(用于保存每折交叉验证的Matthews相关系数)、SN(用于保存每折交叉验证的灵敏度)。
n_folds = 10
kf = KFold(n_splits=10, shuffle=True, random_state=10)
i = 0
acc_stack = np.zeros(10)
mcc = np.zeros(10)
SN = np.zeros(10)
# 定义空列表，用于存储每折的ROC曲线数据
roc_curves = []
# 定义空列表，用于存储每折的AUC值
auc_values = []


# 使用KFold对象的split方法对训练集进行交叉验证，得到训练集的索引。
for train_index, test_index in kf.split(x_train):
    # 通过训练集的索引，使用stack.fit方法对堆叠集成模型进行训练。
    stack.fit(x_train[train_index], y_train[train_index])
    # 使用训练集中未被使用的部分数据(测试集)，使用stack.predict_proba方法得到预测概率。
    stack_pred = stack.predict_proba(x_train[test_index])
    # 提取预测概率中的第二列，即正例的概率。
    stack_predict = stack_pred[:, 1]
    # 使用stack.predict方法对测试集进行预测。
    stack_p = stack.predict(x_train[test_index])
    # print(stack_p)
    # 在循环内部定义空列表，用于存储每折的AUPR值
    aupr_values = []
    # 计算AUC值，并将结果保存到相应的数组中。
    auc_value = roc_auc_score(y_train[test_index], stack_predict)
    auc_values.append(auc_value)
    # 计算AUPR值，并将结果保存到相应的数组中。
    aupr_value = average_precision_score(y_train[test_index], stack_predict)
    aupr_values.append(aupr_value)
    # 计算准确率(accuracy_score)、ROC曲线的假正率和真正率(roc_curve)、灵敏度(recall_score)和Matthews相关系数(matthews_corrcoef)，并将结果保存到相应的数组中。
    acc_stack[i] = accuracy_score(y_train[test_index], stack_p)
    fpr, tpr, _ = roc_curve(y_train[test_index], stack_predict)
    roc_curves.append((fpr, tpr))
    SN[i] = recall_score(y_train[test_index], stack_p)
    mcc[i] = matthews_corrcoef(y_train[test_index], stack_p)
    print("Performance of the " + str(i + 1) + "-th fold lncRNA-disease associations")
    print('acc_test:', acc_stack[i])
    print('sn_test:', SN[i])
    print('mcc_test:', mcc[i])
    print('AUC:', auc_value)
    print("\n")

    # # 使用t-SNE对特征数据进行降维
    # tsne = TSNE(n_components=2, random_state=10)
    # x_train_embedded = tsne.fit_transform(x_train[test_index])  # 使用测试集数据进行降维
    #
    # # 绘制t-SNE降维后的数据散点图
    # plt.figure()
    # plt.scatter(x_train_embedded[:, 0], x_train_embedded[:, 1], c=y_train[test_index], cmap='coolwarm', marker="*", s=16)
    # plt.title('t-SNE Visualization')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # filename = f'tsne_visualization_fold_{i + 1}.png'
    # plt.savefig('./t-sne/'+filename)
    # plt.show()

    # 进行下一折交叉验证，更新计数器i。
    i = i + 1

# 计算平均AUC值
mean_auc = np.mean(auc_values)
print("Mean AUC:", mean_auc)
# 计算平均AUPR值
mean_aupr = np.mean(aupr_values)
print("Mean AUPR:", mean_aupr)
# 计算平均准确率（accuracy）
average_acc = np.mean(acc_stack)
print("Average Accuracy:", average_acc)
# 计算平均灵敏度（sensitivity）
average_sn = np.mean(SN)
print("Average Sensitivity:", average_sn)
# 计算平均Matthews相关系数（MCC）
average_mcc = np.mean(mcc)
print("Average MCC:", average_mcc)

# # 绘制每折的ROC曲线
# plt.figure()
#
# for i, (fpr, tpr) in enumerate(roc_curves):
#     plt.plot(fpr, tpr, label=f'Fold {i + 1}')
#
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.title('ROC Curves for Different Folds')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.grid(True)
# plt.show()


# # 使用完整的训练集对堆叠集成模型进行训练。
# stack.fit(x_train, y_train)
# # 使用堆叠集成模型对测试集进行预测。
# stack_pred = stack.predict_proba(x_test)
# stack_predict = stack_pred[:, 1]
# stack_p = stack.predict(x_test)
# # 计算测试集上的准确率、ROC曲线的假正率和真正率、灵敏度和Matthews相关系数，并打印结果。
# acc_stack_test = accuracy_score(y_test, stack_p)
# fpr, tpr, thresholdTest = roc_curve(y_test, stack_predict)
# mcc_test = matthews_corrcoef(y_test, stack_p)
# sn_test = recall_score(y_test, stack_p)
# print('The performance of lncRNA-disease associations：')
# print('acc_test:', acc_stack_test)
# print('sn_test:', sn_test)
# print('mcc_test:', mcc_test)

# # 初始化T-SNE模型
# tsne = TSNE(n_components=2, random_state=42)
#
# # 进行降维
# data_tsne = tsne.fit_transform(x_train)
#
# # 可视化降维结果
# plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=y_train, cmap='coolwarm', s=5.0, marker="*")
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.title('T-SNE Visualization')
# plt.savefig('./t-sne/tsne_visualization.png')
# plt.show()




