import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = pd.read_csv("IRIS.csv")

print("First 5 rows:")
print(iris.head())
print("\nDataset Description:")
print(iris.describe())

print("\nTarget Labels:", iris["species"].unique())

plt.scatter(
    iris["sepal_width"],
    iris["sepal_length"],
    c=pd.factorize(iris["species"])[0]
)
plt.xlabel("Sepal Width")
plt.ylabel("Sepal Length")
plt.title("Iris Dataset Visualization")
plt.show()

X = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

x_new = np.array([[6.0, 2.9, 1.0, 0.2]])

prediction = knn.predict(x_new)
print("\nPrediction for new sample:", prediction[0])
