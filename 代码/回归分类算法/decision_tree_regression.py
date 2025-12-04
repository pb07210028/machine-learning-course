# 决策树回归预测工资

# 导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据集
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# 构建决策树模型
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0, max_depth = 2)
regressor.fit(X, y)



# 可视化回归效果
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# 预测
# y_pred = regressor.predict(6.5)


# 画出树结构
# pip install graphviz
from sklearn import tree
import graphviz
graphviz.Source(tree.export_graphviz(regressor, out_file='output.dot'))

# 将 dot 文件转换成其它文件
import os
os.system('dot -Tpng output.dot -o output.png') # 转换成png文件
os.system('dot -Tpdf output.dot -o output.pdf') # 转换成pdf文件

