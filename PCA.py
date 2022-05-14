from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
data=load_iris()
n_componenta=2
model=PCA(n_components=n_componenta)
model=model.fit(data.data)
print(model.transform(data.data))
