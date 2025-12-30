# =========================
# Import Required Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# =========================
# Load Dataset
# =========================
dataset = pd.read_csv("HousePricePrediction.csv")

# =========================
# a) Print First Five Rows
# =========================
print("First five rows of dataset:")
print(dataset.head())

# =========================
# b) Dataset Shape
# =========================
print("\nDataset Shape:", dataset.shape)

# =========================
# c) Basic Statistical Computation
# =========================
print("\nStatistical Summary:")
print(dataset.describe())

# =========================
# d) Print Columns & Data Types
# =========================
print("\nColumn Data Types:")
print(dataset.dtypes)

# =========================
# e) Detect & Replace Null Values with MODE
# =========================
print("\nNull Values Before Handling:")
print(dataset.isnull().sum())

for col in dataset.columns:
    if dataset[col].isnull().sum() > 0:
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

print("\nNull Values After Handling:")
print(dataset.isnull().sum())

# =========================
# f) Heatmap (Numeric Features)
# =========================
numeric_data = dataset.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()

# =========================
# One-Hot Encoding (Categorical)
# =========================
categorical_cols = dataset.select_dtypes(include='object').columns

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_data = pd.DataFrame(
    encoder.fit_transform(dataset[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)

dataset = dataset.drop(categorical_cols, axis=1)
dataset = pd.concat([dataset, encoded_data], axis=1)

# =========================
# g) Train-Test Split
# =========================
X = dataset.drop('SalePrice', axis=1)
y = dataset['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# h) House Price Prediction
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
print("\nLinear Regression MAPE:", mape)
