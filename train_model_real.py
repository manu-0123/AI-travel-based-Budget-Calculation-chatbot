import matplotlib
matplotlib.use("Agg")
import logging
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
LIVE_DATA_FILE = "india_comprehensive_dataset_2026.csv"
CLEAN_DATA_FILE = "processed_india_travel_data.csv"
MODEL_FILE = "travel_budget_model.joblib"
FEATURE_FILE = "trained_feature_columns.joblib"

# -------------------------------------------------
# OUTPUT DIRECTORIES
# -------------------------------------------------
OUTPUT_DIR = "training_outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_FILE = os.path.join(OUTPUT_DIR, "train_logs.txt")

os.makedirs(PLOT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------
# GLOBAL LOGGER (✅ FIXED)
# -------------------------------------------------
def log(msg):
    print(msg)
    logging.info(msg)
# -------------------------------------------------
# DATA CLEANING
# -------------------------------------------------
def retrieve_and_clean_dataset():
    if not os.path.exists(LIVE_DATA_FILE):
        print(f"❌ Missing dataset: {LIVE_DATA_FILE}")
        return None

    df = pd.read_csv(LIVE_DATA_FILE)

    # ---------------------------------------------
    # REQUIRED BASE COLUMNS
    # ---------------------------------------------
    for col in ["Source", "Destination","Travel_Mode"]:
        if col not in df.columns:
            df[col] = "Unknown"
    
     # ------------------------------
    # FIX DAYS COLUMN
    # ------------------------------
    if "Days" in df.columns and "Duration_Days" not in df.columns:
        df["Duration_Days"] = df["Days"]

    if "Duration_Days" not in df.columns:
        df["Duration_Days"] = 3

    if "People" in df.columns and "Num_People" not in df.columns:
        df["Num_People"] = df["People"]

    if "Num_People" not in df.columns:
        df["Num_People"] = 1

    # ------------------------------
    # WEATHER PROCESSING (ROBUST)
    # ------------------------------
    df["Dest_Weather"] = df.get("Dest_Weather", "25°C (Clear)").astype(str)

    df["Temperature"] = (
        df["Dest_Weather"]
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
        .fillna(25.0)
    )

    df["Rain_Flag"] = df["Dest_Weather"].str.lower().apply(
        lambda x: 1 if any(w in x for w in ["rain", "storm", "drizzle"]) else 0
    )

    # Cost cleaning (already numeric-safe)
    df["Total_Cost"] = (
        df["Grand_Total"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
    )
    if "Transport_Cost" not in df.columns:
        df["Transport_Cost"] = 0.0

    # -------------------------------
    # DROP INVALID ROWS
    # -------------------------------
    df = df.dropna(
        subset=[
            "Source",
            "Destination",
            "Travel_Mode",
            "Transport_Cost",
            "Duration_Days",
            "Num_People",
            "Temperature",
            "Rain_Flag",
            "Total_Cost"
        ]
    )

    df.to_csv(CLEAN_DATA_FILE, index=False)
    log(f"✅ Clean dataset saved: {CLEAN_DATA_FILE}")
    log(f"Dataset shape after cleaning: {df.shape}")
    log(f"Columns used: {list(df.columns)}")
    log(f"Sample rows:\n{df.head(3)}")

    return df

# -------------------------------------------------
# VISUALIZATION DASHBOARD
# -------------------------------------------------
def generate_visualizations(df, y_test, preds, model):

    sns.set_theme(style="whitegrid")

    # ===============================
    # 1️⃣ ACTUAL vs PREDICTED
    # ===============================
    plt.figure(figsize=(7, 6))
    sns.regplot(x=y_test, y=preds)
    plt.title("Actual vs Predicted Budget (2026)")
    plt.xlabel("Actual Budget")
    plt.ylabel("Predicted Budget")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()

   # ===============================
   # 2️⃣ FEATURE IMPORTANCE (SAFE)
   # ===============================
    pre = model.named_steps["preprocessor"]
    reg = model.named_steps["regressor"]

   # 🔑 This is the KEY line
    feature_names = pre.get_feature_names_out()

    importances = reg.feature_importances_

    feat_df = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    .sort_values("Importance", ascending=False)
    .head(10)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, x="Importance", y="Feature")
    plt.title("Top 10 Cost Drivers (2026)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()


    # ===============================
    # 3️⃣ WEATHER IMPACT
    # ===============================
    df_plot = df.copy()
    df_plot["Weather"] = df_plot["Rain_Flag"].map({1: "Rainy", 0: "Clear"})

    plt.figure(figsize=(7, 6))
    sns.boxplot(data=df_plot, x="Weather", y="Total_Cost")
    plt.title("Weather Impact on Travel Budget")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "weather_impact.png"), dpi=300,bbox_inches="tight")
    plt.close()

    # ===============================
    # 4️⃣ REGIONAL COST ANALYSIS
    # ===============================
    top_dest = (
        df.groupby("Destination")["Total_Cost"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_dest, x="Total_Cost", y="Destination")
    plt.title("Top 10 Expensive Destinations (2026)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "regional_cost.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    log("✅ All visualization PNGs generated successfully")
    print("✅ All visualization PNGs generated successfully")


# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
def train_model():
    df = retrieve_and_clean_dataset()
    if df is None or len(df) < 20:
        print("⚠️ Not enough data to train model")
        return

    FEATURES = [
        "Source",
        "Destination",
        "Travel_Mode",
        "Transport_Cost",
        "Duration_Days",
        "Num_People",
        "Temperature",
        "Rain_Flag"
    ]
    X = df[FEATURES]
    y = df["Total_Cost"]
    # 🔐 Save feature order for inference
    joblib.dump(FEATURES, FEATURE_FILE)

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["Source", "Destination","Travel_Mode"])
        ],
        remainder="passthrough"
    )
    
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    log(f"Training rows: {len(X_train)}")
    log(f"Testing rows: {len(X_test)}")

    print("🧠 Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"R² Score: {score:.4f}")
    log(f"R² Score: {score:.4f}")
    log(f"Predictions sample: {preds[:5]}")

    joblib.dump(model, MODEL_FILE)
    print(f"💾Model saved: {MODEL_FILE}")

    generate_visualizations(df, y_test, preds, model)

# -------------------------------------------------
if __name__ == "__main__":
    train_model()
