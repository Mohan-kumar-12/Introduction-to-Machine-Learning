import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.io as io
io.renderers.default = 'browser'
data = pd.read_csv("futuresale prediction.csv")
print("First 5 rows:\n", data.head())
print("\nRandom samples:\n", data.sample(5))
print("\nMissing values:\n", data.isnull().sum())
px.scatter(
    data_frame=data,
    x="Sales",
    y="TV",
    size="TV",
    title="Sales vs TV Advertising"
).show()
px.scatter(
    data_frame=data,
    x="Sales",
    y="Newspaper",
    size="Newspaper",
    title="Sales vs Newspaper Advertising"
).show()
px.scatter(
    data_frame=data,
    x="Sales",
    y="Radio",
    size="Radio",
    title="Sales vs Radio Advertising"
).show()
correlation = data.corr()
print("\nCorrelation with Sales:\n")
print(correlation["Sales"].sort_values(ascending=False))
X = data.drop(columns=["Sales"])
y = data["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("\nModel R² Score:", accuracy)
new_features = np.array([[230.1, 37.8, 69.2]])
prediction = model.predict(new_features)

print("\nPredicted Sales:", prediction[0])
