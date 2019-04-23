#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:50:17 2019

@author: Adam
"""

# 加载pkg文件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load数据
data = pd.read_csv('data_train.csv')
data = data.values

# 判断数据的横纵方向含义，row表示的是一个sample，colomn表示的是一个feature
n_samples, n_features = data.shape
# 样本的label总共有0，1，2三种
n_label = 3
# colomn中有两个不是feature，是label和value
print("n_label %d, \t n_samples %d, \t n_features %d"  % (n_label, n_samples, n_features - 2))

# 第21列是value变量，从矩阵中截取
ftl_value = np.zeros(n_samples)
ftl_value = data[:,20]

# 第64列是label变量，从矩阵中截取
labels = np.zeros(n_samples)
labels = data[:,63]

# 删除第21和第64列
features = np.delete(data, [20,63], axis=1)

# feature矩阵标准化
sc = StandardScaler()
features = sc.fit_transform(features)

# 现在我们有三个变量
# ftl_value.................表示客流的价值
# labdels...................表示客流的类型
# features..................表示客流的特点数据矩阵

# 接下来我们来预测label
# 由于我们已经知道label和数据，我们希望的是一个监督分类学习，因此我们进行分类算法
# 输入变量X和变量Y
X = features
y = labels

# 定义分类算法名称和函数
names = ['l2regression','LinearSVC','NonlinearSVC','KNeighbors','GPC','DecisionTree']
classifierSet = [LogisticRegression(penalty='l2',solver='saga', multi_class = 'auto', max_iter = 10000),
                 SVC(kernel='linear'),
                 SVC(kernel='poly', gamma = 'scale'),
                 KNeighborsClassifier(3),
                 GaussianProcessClassifier(),
                 DecisionTreeClassifier()]


# 输出各算法的准确度
print("Origional Data")
print("Name             Accuracy")
i = 0
for classifier in classifierSet:
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print("%s   %0.1f%%" % (names[i].ljust(16), accuracy * 100))
    i += 1
    
# 根据输出的结果，先采用LDA降维
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(X, y).transform(X)

plt.figure()
colors = ['#ffe5a0', '#8effa3','#a3b8ff']

for color, i in zip(colors, [0, 1, 2]):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], alpha=.8, color=color, edgecolors='k')
plt.title('LDA of VDF footfall dataset')

plt.show()

# 利用X_lda数据集重复分类步骤
print("Reduced-Demination Data")
print("Name             Accuracy")
i = 0
for classifier in classifierSet:
    classifier.fit(X_lda, y)
    y_pred = classifier.predict(X_lda)
    accuracy = accuracy_score(y, y_pred)
    print("%s   %0.1f%%" % (names[i].ljust(16), accuracy * 100))
    i += 1

# lda后分为训练集和预测集
X = StandardScaler().fit_transform(X_lda)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# 设置图像色调
cm = plt.cm.viridis
cm_bright = ListedColormap(['#ffe5a0', '#8effa3','#a3b8ff'])
                            
# 生成网格采样点
h =.02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# 设置画布大小
fig, axes = plt.subplots(nrows=1, ncols=len(classifierSet) + 1,figsize=(41, 4))

# 作第一张图
i = 0

# just plot the dataset first              
ax = axes[i]
ax.set_title("Input data")

# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# 作第二~end张图,并添加分数
for name, clf in zip(names, classifierSet):
    ax = axes[i]
    ax.set_title(names[i-1])

    clf = classifierSet[i-1]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    if (len(Z.shape) != 1):
        Z = Z[:,0].reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    else:
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.show()