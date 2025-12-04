# Polynomial Regression

# 导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据集
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# 使用简单线性回归拟合
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X, y)

# 可视化拟合效果
plt.figure()
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_1.predict(X), color = 'green')
plt.title('Salary vs Position')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# 创建新的自变量
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# 使用多项式回归拟合
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# 可视化拟合效果
plt.figure()
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'green')
plt.title('Salary vs Position')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# 数学表达式
print('salary = %.2f * level + %.2f * level^2 + %.2f * level^3 + %.2f * level^4 + %.2f' %(lin_reg_2.coef_[1], lin_reg_2.coef_[2], lin_reg_2.coef_[3], lin_reg_2.coef_[4], lin_reg_2.intercept_))

# 预测
pred_salary = lin_reg_2.predict(poly_reg.fit_transform(6.5))

