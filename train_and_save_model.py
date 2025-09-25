# train_and_save_model.py
import os
import json
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_PATH = "insurance.csv"       # <- change if file name/path differs
MODEL_OUT = "best_model_pipeline.joblib"
META_OUT = "model_metadata.json"
TEST_SIZE = 0.20
RANDOM_STATE = 42
# ----------------------------

print("scikit-learn version:", sklearn.__version__)

# 1) Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put insurance.csv in this folder or change DATA_PATH.")

df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)
print(df.head())

expected_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
if not expected_cols.issubset(set(df.columns)):
    raise ValueError(f"Dataset is missing expected columns. Expected at least: {expected_cols}. Found: {set(df.columns)}")

# 2) Features & target
X = df[['age','sex','bmi','children','smoker','region']].copy()

# 🔹 Use log transform for charges
y = np.log(df['charges'])

# 3) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 4) Preprocessing pipeline
numeric_features = ['age','bmi','children']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['sex','smoker','region']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # sklearn 1.5.x
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# 5) Model
regressor = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

# 6) Train
print("Training pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# 7) Evaluate (convert back from log → actual USD)
y_pred_log = pipeline.predict(X_test)
y_pred_usd = np.exp(y_pred_log)
y_test_usd = np.exp(y_test)

mae = mean_absolute_error(y_test_usd, y_pred_usd)
rmse = mean_squared_error(y_test_usd, y_pred_usd, squared=False)
r2 = r2_score(y_test_usd, y_pred_usd)

print(f"MAE: {mae:.2f} USD, RMSE: {rmse:.2f} USD, R2: {r2:.4f}")

# --- Extra Plots ---
plt.scatter(y_test_usd, y_pred_usd, alpha=0.6)
plt.xlabel("Actual Charges (USD)")
plt.ylabel("Predicted Charges (USD)")
plt.title("Predicted vs Actual (log-trained model)")
plt.plot([y_test_usd.min(), y_test_usd.max()], [y_test_usd.min(), y_test_usd.max()], 'r--')
plt.savefig("pred_vs_actual.png")
plt.close()

errors = y_test_usd - y_pred_usd
plt.hist(errors, bins=30, edgecolor='k')
plt.xlabel("Prediction Error (USD)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.savefig("error_distribution.png")
plt.close()

print("Saved evaluation plots: pred_vs_actual.png, error_distribution.png")

# 8) Save pipeline safely
if os.path.exists(MODEL_OUT):
    bak_name = MODEL_OUT + ".bak"
    if os.path.exists(bak_name):
        os.remove(bak_name)  # delete old backup
    os.rename(MODEL_OUT, bak_name)

joblib.dump(pipeline, MODEL_OUT, compress=3)
print(f"Saved pipeline to {MODEL_OUT}")

# 9) Save metadata
meta = {
    "saved_at": datetime.utcnow().isoformat() + "Z",
    "sklearn_version": sklearn.__version__,
    "test_mae_usd": float(mae),
    "test_rmse_usd": float(rmse),
    "test_r2": float(r2),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "target_transform": "log(charges)"
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata to {META_OUT}")

# 10) Quick sanity predict
example = X_test.iloc[[0]]
pred_log = pipeline.predict(example)[0]
pred_usd = np.exp(pred_log)   # undo log transform
print("Example input:\n", example.to_dict(orient='records')[0])
print(f"Predicted (USD): {pred_usd:.2f}, Predicted (INR ~*83): {pred_usd*83:.2f}")
