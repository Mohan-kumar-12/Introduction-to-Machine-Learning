# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# 2. Load Dataset
# ===============================
data = pd.read_csv("CREDITSCORE.csv")

print("First 5 rows of dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# ===============================
# 3. Data Preprocessing
# ===============================

# Encode Credit_Mix if it is categorical
if data["Credit_Mix"].dtype == object:
    data["Credit_Mix"] = data["Credit_Mix"].map({
        "Bad": 0,
        "Standard": 1,
        "Good": 3
    })

# Remove missing values
data = data.dropna()

# ===============================
# 4. Feature Selection
# ===============================
X = data[
    [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_History_Age",
        "Monthly_Balance",
    ]
]

# Target (1D)
y = data["Credit_Score"]

# ===============================
# 5. Train-Test Split
# ===============================
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# ===============================
# 6. Train Model
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(xtrain, ytrain)

# ===============================
# 7. Evaluate Model
# ===============================
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ===============================
# 8. User Input for Prediction
# ===============================
print("\n--- Enter Customer Details ---")

a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit Cards: "))
e = float(input("Interest Rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average days delayed from due date: "))
h = float(input("Number of delayed payments: "))
i = float(input("Credit Mix (Bad: 0, Standard: 1, Good: 3): "))
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

# ===============================
# 9. Prediction
# ===============================
features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])

prediction = model.predict(features)

print("\nPredicted Credit Score:", prediction[0])
