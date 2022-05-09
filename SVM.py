from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles

'''
创建时间=2022/5/9 20:03
用户=Tao zhang
函数功能=线性支持向量机
参数说明=线性支持向量机是以间隔最大化为基准，来学习得到尽可能地远离数据的决策边界的算法
返回说明=
'''
def LearnSVM():
#生成数据
 centers=[(-1,-0.125),(0.5,0.5)]
 X,y=make_blobs(n_samples=50,n_features=2,centers=centers,cluster_std=0.3)
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
 model=LinearSVC()
 model.fit(X_train,y_train)
 y_pred=model.predict(X_test)
 print(accuracy_score(y_pred, y_test))
'''
创建时间=2022/5/9 20:02
用户=Tao zhang
函数功能=使用核方法的支持向量机
参数说明=通过核函数学习到高维空间的线性决策边境
返回说明=
'''
def SVC_RBF():
    #生成数据
    X,y=make_gaussian_quantiles(n_features=2,n_classes=2,n_samples=100)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    model=SVC(kernel='linear')
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(accuracy_score(y_pred, y_test))
if __name__ == '__main__':
    SVC_RBF()


