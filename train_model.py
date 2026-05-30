import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("insurance (1).csv")

# Encode Categorical Columns
le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df["region"] = le.fit_transform(df["region"])

# Features and Target
X = df.drop("charges", axis=1)
y = df["charges"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base Models
base_models = [
    ("lr", LinearRegression()),
    ("dt", DecisionTreeRegressor(max_depth=5)),
    ("rf", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
]

# Meta Model
meta_model = Ridge()

# Stacking Regressor
model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2 :", r2_score(y_test, y_pred))

# Save
os.makedirs("models", exist_ok=True)

pickle.dump(
    model,
    open("models/stacking_regressor.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("Model Saved Successfully!")