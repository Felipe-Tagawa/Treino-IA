from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

# print(X.shape, y.shape)

df = pd.DataFrame(X, columns=iris.feature_names)
df ['especie'] = iris.target_names[y]

print(df.groupby('especie').mean().round(2))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4,
    stratify=y, random_state=42
)

#print(X_train.shape, X_test.shape)

model = KNeighborsClassifier(
    n_neighbors=5,
)

# Treinamento

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:8])
print(y_test[:8])

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
