import numpy as np
from sklearn.linear_model import LogisticRegression
'''
创建时间=2022/5/9 14:44
用户=Tao zhang
函数功能=逻辑回归
参数说明=
返回说明=
'''
X_train=np.r_[np.random.normal(3,1,size=50),np.random.normal(-1,1,size=50)].reshape(100,-1)
y_train=np.r_[np.ones(50),np.zeros(50)]
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.predict_proba([[0], [1], [2]]))

