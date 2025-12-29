import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("CarPrice.csv")

# Data exploration
print(data.info())

# Correlation (numeric only)
numeric_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(20, 15))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=True)
plt.show()

# Model training
predict = "price"
features = numeric_data.drop(columns=[predict])
target = numeric_data[predict]

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)

print("Model Score:", model.score(xtest, ytest))
print("MAE:", mean_absolute_error(ytest, predictions))
