
#%%使用pandas理解和编辑数据
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
data=load_wine()
df_x=pd.DataFrame(data.data,columns=data.feature_names)
df_x.head()
df_y=pd.DataFrame(data.target,columns=["kind(target)"])
print(df_y.head())
df=pd.concat([df_x,df_y],axis=1)
print(df.head)
plt.hist(df.loc[:,"alcohol"])
df.corr()
plt.show()
plt.boxplot(df.loc[:,"alcohol"])
from pandas.plotting import  scatter_matrix

scatter_matrix(df.iloc[:,[0,9]])
plt.show()

