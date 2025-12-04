# multiple linear regression


#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# avoiding dummy variable trap
X = X[:,1:]


# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train ,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 0)

# finding relationship
import seaborn as sns
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(dataset, hue="State")

# find the optimal model
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # if axis == 1, then along row. else along column.
from statsmodels.formula.api import OLS

# with all variables
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = OLS(endog = y, exog = X_opt)
regressor_OLS = regressor_OLS.fit()
regressor_OLS.summary()

# remove New York
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = OLS(endog = y, exog = X_opt)
regressor_OLS = regressor_OLS.fit()
regressor_OLS.summary()

# remove Floria
X_opt = X[:,[0,3,4,5]]
regressor_OLS = OLS(endog = y, exog = X_opt)
regressor_OLS = regressor_OLS.fit()
regressor_OLS.summary()

# remove Administration
X_opt = X[:,[0,3,5]]
regressor_OLS = OLS(endog = y, exog = X_opt)
regressor_OLS = regressor_OLS.fit()
regressor_OLS.summary()

# remove Marketing Spend
X_opt = X[:,[0,3]]
regressor_OLS = OLS(endog = y, exog = X_opt)
regressor_OLS = regressor_OLS.fit()
regressor_OLS.summary()

# Build the optimal model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train[:,[2]],y_train)

y_pred = regressor.predict(X_test[:,[2]])

# Visualising the Test set results
plt.figure()
plt.scatter(X_test[:,[2]], y_test, color = 'red')
plt.plot(X_test[:,[2]], y_pred, color = 'blue')
plt.title('Profit vs R&D Spend (Test set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

# print hypothesis
print('Profit = %.2f * R&D Spend + %.2f' %(regressor.coef_, regressor.intercept_))

