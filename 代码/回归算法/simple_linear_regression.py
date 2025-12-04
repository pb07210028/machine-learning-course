'''
Simple Linear Regression
'''

# 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# 可视化数据集
plt.figure()
plt.title('dataset visualization')
plt.scatter(X, y, color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# 模拟生成预测的y
coefficient = 9345.94
intercept = 26816.19
y1 = X*(coefficient - 1000) + intercept
y2 = X*coefficient + intercept-150
y3 = X*(coefficient - 1000) + intercept+150

# 可视化拟合的直线
plt.figure()
plt.title('dataset & pattern visualization')
plt.scatter(X, y, color='red')
#plt.plot(X, y1, color = 'pink')
plt.plot(X, y2, color = 'orange')
#plt.plot(X, y3, color = 'purple')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


