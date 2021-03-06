from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#读取数据
data =load_wine()
X_train,X_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.3)
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(model.feature_importances_)
print(data.feature_names)
print(y_pred)
print(accuracy_score(y_pred, y_test))

