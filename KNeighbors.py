import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
#生成数据
X,y=make_moons(n_samples=100,noise=0.3)
plt.scatter(X[y==0,0],X[y==0,1],s=20,c='b')
plt.scatter(X[y==1,0],X[y==1,1],s=20,c='y')
plt.show()




X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_pred, y_test))

