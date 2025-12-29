# =========================
# Import required libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# =========================
# Load Dataset
# =========================
dataset = pd.read_csv("HousePricePrediction.csv")

# =========================
# Dataset Exploration
# =========================
print(dataset.head())
print("Dataset shape:", dataset.shape)

# Identify column types
object_cols = dataset.select_dtypes(include='object').columns
int_cols = dataset.select_dtypes(include='int64').columns
float_cols = dataset.select_dtypes(include='float64').columns

print("Categorical variables:", len(object_cols))
print("Integer variables:", len(int_cols))
print("Float variables:", len(float_cols))

# =========================
# Correlation Heatmap
# (NUMERIC COLUMNS ONLY)
# =========================
numeric_dataset = dataset.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 6))
sns.heatmap(
    numeric_dataset.corr(),
    cmap='BrBG',
    annot=True,
    fmt='.2f',
    linewidths=1
)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# =========================
# Categorical Feature Analysis
# =========================
unique_values = [dataset[col].nunique() for col in object_cols]

plt.figure(figsize=(10, 6))
plt.title("Number of Unique Values in Categorical Features")
sns.barplot(x=object_cols, y=unique_values)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(18, 36))
plt.title("Categorical Feature Distributions")
index = 1

for col in object_cols:
    plt.subplot(11, 4, index)
    sns.countplot(x=dataset[col])
    plt.xticks(rotation=90)
    index += 1

plt.tight_layout()
plt.show()

# =========================
# Data Cleaning
# =========================
dataset.drop(['Id'], axis=1, inplace=True)

dataset['SalePrice'] = dataset['SalePrice'].fillna(
    dataset['SalePrice'].mean()
)

new_dataset = dataset.dropna()

# =========================
# One-Hot Encoding
# =========================
object_cols = new_dataset.select_dtypes(include='object').columns

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

encoded_cols = pd.DataFrame(
    encoder.fit_transform(new_dataset[object_cols]),
    index=new_dataset.index,
    columns=encoder.get_feature_names_out(object_cols)
)

df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, encoded_cols], axis=1)

# =========================
# Train-Test Split
# =========================
X = df_final.drop('SalePrice', axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# =========================
# Support Vector Regression
# =========================
model_SVR = SVR()
model_SVR.fit(X_train, Y_train)

Y_pred_svr = model_SVR.predict(X_valid)
svr_mape = mean_absolute_percentage_error(Y_valid, Y_pred_svr)

print("SVR MAPE:", svr_mape)

# =========================
# Linear Regression
# =========================
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)

Y_pred_lr = model_LR.predict(X_valid)
lr_mape = mean_absolute_percentage_error(Y_valid, Y_pred_lr)

print("Linear Regression MAPE:", lr_mape)
