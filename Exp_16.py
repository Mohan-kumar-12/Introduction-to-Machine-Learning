import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
iris = pd.read_csv(
    r"C:\Users\MOHAN KUMAR\OneDrive\Desktop\ML Lab\IRIS.csv"
)
print(iris.head())
X = iris.drop("species", axis=1)
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(max_iter=1000)
bernoulli_nb = BernoulliNB()
passive_aggressive = PassiveAggressiveClassifier(random_state=42)
knn.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
bernoulli_nb.fit(X_train, y_train)
passive_aggressive.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
dt_pred = decision_tree.predict(X_test)
lr_pred = logistic_regression.predict(X_test)
nb_pred = bernoulli_nb.predict(X_test)
pa_pred = passive_aggressive.predict(X_test)
results = {
    "Classification Algorithm": [
        "KNN Classifier",
        "Decision Tree Classifier",
        "Logistic Regression",
        "Bernoulli Naive Bayes",
        "Passive Aggressive Classifier"
    ],
    "Accuracy": [
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, pa_pred)
    ]
}
score_df = pd.DataFrame(results)
print("\nModel Accuracy Comparison:\n")
print(score_df)
print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, lr_pred))
