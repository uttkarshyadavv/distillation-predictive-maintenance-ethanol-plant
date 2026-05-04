# distillation_pdm_rf.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# ============================
# USER INPUT FILES
# ============================

HEALTHY_CSV = "distillation_healthy.csv"
FAULTY_CSV  = "distillation_faulty.csv"

OUTPUT_PREFIX = "distillation_model"

# ============================
# REQUIRED COLUMNS
# ============================

INPUT_COLS = [
    "Reboiler_Spec",
    "Condenser_Spec"
]

OUTPUT_COLS = [
    "Condenser_Duty",
    "Reboiler_Duty",
    "T_top",
    "T_mid",
    "T_bottom"
]

ALL_COLS = INPUT_COLS + OUTPUT_COLS

# ============================
# LOAD FUNCTION
# ============================

def load_data(path):
    df = pd.read_csv(path)

    for col in ALL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna()

    X = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values

    return df, X, y

# ============================
# LOAD DATA
# ============================

df_h, X_h, y_h = load_data(HEALTHY_CSV)

# ============================
# TRAIN MODEL (HEALTHY ONLY)
# ============================

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_h, y_h)

y_h_pred = model.predict(X_h)

# ============================
# THRESHOLD FROM HEALTHY DATA
# ============================

resid_h = np.abs(y_h - y_h_pred)
threshold = np.percentile(resid_h, 99) * 3.5

print(f"Threshold (from healthy): {threshold:.4f}")

# ============================
# LOAD FAULT DATA
# ============================

df_f, X_f, y_f = load_data(FAULTY_CSV)

y_f_pred = model.predict(X_f)

r2 = r2_score(y_f, y_f_pred)
print(f"R2 on faulty data: {r2:.4f}")

# ============================
# ANOMALY DETECTION
# ============================

resid_f = np.abs(y_f - y_f_pred)

# combine multi-output into single score
resid_f_total = resid_f.mean(axis=1)

anomaly = resid_f_total > threshold

# ============================
# EARLY WARNING LOGIC
# ============================

WARNING_WINDOW = 5
count = 0
first_warning_index = None

for i in range(len(anomaly)):
    if anomaly[i]:
        count += 1
        if count >= WARNING_WINDOW:
            first_warning_index = i - WARNING_WINDOW + 1
            break
    else:
        count = 0

if first_warning_index is not None:
    print(f"⚠️ EARLY WARNING at index: {first_warning_index}")
else:
    print("No early fault detected")

# ============================
# SAVE MODEL
# ============================

os.makedirs("models", exist_ok=True)
joblib.dump(model, f"models/{OUTPUT_PREFIX}.pkl")
#Plot

plt.figure(figsize=(10,5))
plt.plot(resid_f_total, label="Residual")
plt.axhline(threshold, linestyle='--', label="Threshold")

if first_warning_index is not None:
    plt.axvline(first_warning_index, linestyle='--', label="Warning")

plt.legend()
plt.title("Residual vs Threshold")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# ============================
# SAVE OUTPUT
# ============================

df_out = df_f.copy()

for i, col in enumerate(OUTPUT_COLS):
    df_out[f"{col}_pred"] = y_f_pred[:, i]

df_out["residual"] = resid_f_total
df_out["anomaly"] = anomaly.astype(int)

df_out.to_csv("distillation_results.csv", index=False)

print("Results saved to distillation_results.csv")