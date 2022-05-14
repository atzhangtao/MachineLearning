import sklearn.metrics
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
X=data.data
y=1-data.target
X=X[:,:10]
from sklearn.linear_model import LogisticRegression
model_lor=LogisticRegression()
model_lor.fit(X,y)
y_pred=model_lor.predict(X)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,y_pred)
ac=sklearn.metrics.accuracy_score(y,y_pred)
print(model_lor.predict_proba(X)[:,1]>0.1)
print(cm)
print(ac)
