from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, y.shape)