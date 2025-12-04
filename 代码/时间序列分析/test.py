# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:23:45 2020

@author: reedg
"""



# date_range()函数使用举例
import numpy as np
import pandas as pd

ts1= pd.date_range(start = '2000/1/1', periods = 200)
mydata1 = pd.Series(data = np.random.randn(200), index = ts1)




# period_range()函数使用举例
import numpy as np
import pandas as pd

ts2 = pd.period_range(start = '2000', end = '2020', freq='Y')
mydata2 = pd.Series(data = np.random.randn(len(ts2)), index = ts2)




# 绘制函数
import numpy as np
import pandas as pd

ts2 = pd.period_range(start = '2000', end = '2020', freq='Y')
mydata2 = pd.Series(data = np.random.randn(len(ts2)), index = ts2)
mydata2.plot()


# rolling().mean()函数使用举例
import numpy as np
import pandas as pd
# 导入数据
data = pd.read_csv('sales.csv') 
# 转换成时间序列数据
ts=pd.period_range(start='2001',periods=len(data),freq='Y')
data=pd.DataFrame(data['sales'].values,index=ts,columns=['sales'])
# 求2年移动平均
two_year_ma = data.rolling(window=2).mean()
three_year_ma = data.rolling(window=3).mean()
# 把2年移动平均数据加到data中
data['two_year_ma'] = two_year_ma
# 把3年移动平均数据加到data中
data['three_year_ma'] = three_year_ma
# 画对比图
data.plot()



# SimpleExpSmooting().fit()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('rain_data.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='1813',periods=len(data),freq='Y')
data=pd.DataFrame(data['rain'].values,index=ts,columns=['rain'])

data.plot()

from statsmodels.tsa.api import SimpleExpSmoothing

simple_exp_smoothing = SimpleExpSmoothing(data).fit()

simple_exp_smoothing.summary()

data.plot()
simple_exp_smoothing.fittedvalues.plot(
    label='fitted_values',legend=True)


# Holt().fit()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('sales_2006_2017.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='2006',periods=len(data),freq='Y')
data=pd.DataFrame(data['tons'].values,index=ts,columns=['tons'])

data.plot()

from statsmodels.tsa.api import Holt

secondary_exp_smoothing = Holt(data).fit()

secondary_exp_smoothing.summary()

data.plot()
secondary_exp_smoothing.fittedvalues.plot(
    label='fitted_values',legend=True)





# ExponentialSmoothing().fit()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('passenger.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='194901',periods=len(data),freq='M')
data=pd.DataFrame(data['passenger'].values,index=ts,columns=['passenger'])

data.plot()

log_data = np.log(data)

log_data.plot()

from statsmodels.tsa.api import ExponentialSmoothing

exp_smoothing = ExponentialSmoothing(
    data,trend='add',seasonal='add', seasonal_periods=12).fit()

exp_smoothing.summary()

data.plot()
exp_smoothing.fittedvalues.plot(
    label='fitted_values',legend=True)




# diff()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('sales_2006_2017.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='2006',periods=len(data),freq='Y')
data=pd.DataFrame(data['tons'].values,index=ts,columns=['tons'])

# 画出原始图像
data.plot()

# 画出差分后的图像
data_diff = data.diff()
data_diff.plot()







# adfuller()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('sales_2006_2017.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='2006',periods=len(data),freq='Y')
data=pd.DataFrame(data['tons'].values,index=ts,columns=['tons'])


data_diff = data.diff()

# 平稳性检验
from statsmodels.tsa.stattools import adfuller
data_adf = adfuller(data_diff.dropna()['tons'])
print(data_adf)



# plot_acf()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('sales_2006_2017.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='2006',periods=len(data),freq='Y')
data=pd.DataFrame(data['tons'].values,index=ts,columns=['tons'])

# 差分
data_diff = data.diff()

# 平稳性检验
from statsmodels.tsa.stattools import adfuller
data_adf = adfuller(data_diff.dropna()['tons'])

# 自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data_diff.dropna()['tons'])



# plot_pacf()函数使用举例
import numpy as np
import pandas as pd

# 导入数据
data = pd.read_csv('sales_2006_2017.csv') 

# 转换成时间序列数据
ts=pd.period_range(start='2006',periods=len(data),freq='Y')
data=pd.DataFrame(data['tons'].values,index=ts,columns=['tons'])

# 差分
data_diff = data.diff()

# 平稳性检验
from statsmodels.tsa.stattools import adfuller
data_adf = adfuller(data_diff.dropna()['tons'])

# 偏自相关图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(data_diff.dropna()['tons'])


