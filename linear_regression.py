import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

'''
创建时间=2022/5/8 19:43
用户=Tao zhang
函数功能=适用于多元回归和单元的线性回归
参数说明=训练数据和真值
返回说明=线性回归模型
'''
def linearRegression(train_X,traget_y):

    X=[[10.0],[8.0 ],[13.0],[9.0],[11.0],[14.0],[6.0],[4.0],[12.0],[7.0],[5.0]]
    y=[8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]
    model=LinearRegression()
    model.fit(X,y)
    #截距
    print(model.intercept_)
    #斜率
    print(model.coef_)
    y_pred=model.predict(X)
    print(y_pred)
    plt.scatter(X,y)
    plt.plot(X,y_pred)
    plt.show()
    return model


'''
创建时间=2022/5/8 19:48
用户=Tao zhang
函数功能=岭回归
参数说明=
返回说明=返回岭回归模型
'''

def ridge_regression():
    train_size = 20
    test_size = 12
    train_X = np.random.uniform(low=0, high=1.2, size=train_size)
    test_X = np.random.uniform(low=0.1, high=1.3, size=test_size)
    train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
    test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)
# 多项式回归
    poly = PolynomialFeatures(6)
    train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
    test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))

    model = Ridge(alpha=2.5)
    model.fit(train_poly_X,train_y)
    train_pred_y=model.predict(train_poly_X)
    test_pred_y=model.predict(test_poly_X)
    print(model.coef_)
    print(model.intercept_)
    print(mean_squared_error(train_pred_y, train_y))
    print(mean_squared_error(test_pred_y, test_y))
'''
创建时间=2022/5/8 20:54
用户=Tao zhang
函数功能=lasso回归，判断其线性学习参数是否应该为0
参数说明=
返回说明=
'''
def lasso_regression():
    train_size = 20
    test_size = 12
    train_X = np.random.uniform(low=0, high=1.2, size=train_size)
    test_X = np.random.uniform(low=0.1, high=1.3, size=test_size)
    train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
    test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)
# 多项式回归
    poly = PolynomialFeatures(6)
    train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
    test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))

    model = Lasso(alpha=0.02)
    model.fit(train_poly_X,train_y)
    train_pred_y=model.predict(train_poly_X)
    test_pred_y=model.predict(test_poly_X)
    print(model.coef_)
    print(model.intercept_)
    print(mean_squared_error(train_pred_y, train_y))
    print(mean_squared_error(test_pred_y, test_y))
if __name__ == '__main__':
    lasso_regression()




