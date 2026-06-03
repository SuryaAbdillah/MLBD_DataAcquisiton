# ============================================================
# ONE-WEEK TEST PREDICTION VISUALIZATION FOR SAVED MODELS
# Models:
# - Naive Lag 24h
# - Naive Lag 168h
# - EIA Demand Forecast
# - Multiple Linear Regression
# - LASSO Regression
# - XGBoost
# - TFT Full Baseline
#
# Output:
# - Combined line chart
# - Separate actual-vs-model line charts
# - CSV prediction table
# ============================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIGURATION
# ============================================================

DATA_PATH = "merge_outputs/all_features_all_timerange.csv"

OUTPUT_DIR = "modeling_outputs/one_week_prediction_plots"
MODEL_OUTPUT_DIR = "modeling_outputs/saved_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick one week from test period
PLOT_START_DATE = "2025-01-01"
PLOT_DAYS = 7

plot_start = pd.Timestamp(PLOT_START_DATE, tz="UTC")
plot_end = plot_start + pd.Timedelta(days=PLOT_DAYS)

print("Plot period:")
print("Start:", plot_start)
print("End  :", plot_end)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df_model = df.copy()

df_model["timestamp_utc"] = pd.to_datetime(df_model["timestamp_utc"], utc=True)
df_model["data_date"] = pd.to_datetime(df_model["data_date"])

if "timestamp_local" in df_model.columns:
    df_model["timestamp_local"] = df_model["timestamp_utc"].dt.tz_convert("America/Los_Angeles")
else:
    df_model["timestamp_local"] = df_model["timestamp_utc"].dt.tz_convert("America/Los_Angeles")

if "local_date" in df_model.columns:
    df_model["local_date"] = pd.to_datetime(df_model["local_date"])

binary_cols = [
    "is_weekend",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_holiday_period",
]

for col in binary_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].map({
            "No": 0,
            "Yes": 1,
            0: 0,
            1: 1,
            False: 0,
            True: 1
        }).astype("int8")

df_model = df_model.sort_values("timestamp_utc").reset_index(drop=True)

# Split boundaries from the notebook
warmup_start_date = pd.Timestamp("2015-07-01")
modeling_start_date = pd.Timestamp("2017-07-01")
train_end_date = pd.Timestamp("2024-01-01")
valid_end_date = pd.Timestamp("2025-01-01")
test_end_date = pd.Timestamp("2026-05-07")

train_df = df_model[
    (df_model["data_date"] >= modeling_start_date) &
    (df_model["data_date"] < train_end_date)
].copy()

valid_df = df_model[
    (df_model["data_date"] >= train_end_date) &
    (df_model["data_date"] < valid_end_date)
].copy()

test_df = df_model[
    (df_model["data_date"] >= valid_end_date) &
    (df_model["data_date"] < test_end_date)
].copy()

# Same feature columns as notebook
feature_cols = [
    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cooling_degree",
    "heating_degree",
    "temp_squared",

    # Calendar and seasonality
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",

    # Holiday
    "is_weekend",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_holiday_period",

    # Historical demand
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "rolling_std_24h",
    "rolling_std_168h",
]

feature_cols = [col for col in feature_cols if col in df_model.columns]

train_model_df = train_df.dropna(
    subset=feature_cols + ["target_region_demand_mw"]
).copy()

valid_model_df = valid_df.dropna(
    subset=feature_cols + ["region_demand_mw"]
).copy()

test_model_df = test_df.dropna(
    subset=feature_cols + ["region_demand_mw"]
).copy()

X_train = train_model_df[feature_cols]
y_train = train_model_df["target_region_demand_mw"]

X_valid = valid_model_df[feature_cols]
y_valid = valid_model_df["region_demand_mw"]

X_test = test_model_df[feature_cols]
y_test = test_model_df["region_demand_mw"]

print("Train:", train_model_df.shape)
print("Valid:", valid_model_df.shape)
print("Test :", test_model_df.shape)

# ============================================================
# 2. CREATE ONE-WEEK TEST DATAFRAME
# ============================================================

plot_df = test_model_df[
    (test_model_df["timestamp_utc"] >= plot_start) &
    (test_model_df["timestamp_utc"] < plot_end)
].copy()

plot_df = plot_df.sort_values("timestamp_utc").reset_index(drop=True)

prediction_df = pd.DataFrame({
    "timestamp_utc": plot_df["timestamp_utc"],
    "timestamp_local": plot_df["timestamp_local"],
    "Actual": plot_df["region_demand_mw"]
})

# ============================================================
# 3. BASELINE PREDICTIONS
# ============================================================

if "lag_24h" in plot_df.columns:
    prediction_df["Naive Lag 24h"] = plot_df["lag_24h"].values

if "lag_168h" in plot_df.columns:
    prediction_df["Naive Lag 168h"] = plot_df["lag_168h"].values

if "region_demand_forecast_mw" in plot_df.columns:
    prediction_df["EIA Demand Forecast"] = plot_df["region_demand_forecast_mw"].values

# ============================================================
# 4. MULTIPLE LINEAR REGRESSION
# ============================================================

linear_model_path = f"{MODEL_OUTPUT_DIR}/multiple_linear_regression_pipeline.joblib"

if os.path.exists(linear_model_path):
    linear_model = joblib.load(linear_model_path)
    print("Loaded Multiple Linear Regression:", linear_model_path)
else:
    print("Linear model file not found. Training Linear Regression again...")
    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    linear_model.fit(X_train, y_train)

prediction_df["Multiple Linear Regression"] = linear_model.predict(plot_df[feature_cols])

# ============================================================
# 5. LASSO REGRESSION
# ============================================================

lasso_model_path = f"{MODEL_OUTPUT_DIR}/lasso_regression_pipeline.joblib"

if os.path.exists(lasso_model_path):
    best_lasso_model = joblib.load(lasso_model_path)
    print("Loaded LASSO Regression:", lasso_model_path)
else:
    print("LASSO model file not found. Re-training best LASSO model from saved tuning result...")

    lasso_tuning_path = "modeling_outputs/lasso_tuning_results.csv"

    if os.path.exists(lasso_tuning_path):
        lasso_tuning_results_df = pd.read_csv(lasso_tuning_path)
        best_alpha = lasso_tuning_results_df.sort_values("RMSE").iloc[0]["alpha"]
        print("Best alpha loaded from tuning CSV:", best_alpha)
    else:
        best_alpha = 0.0001
        print("LASSO tuning CSV not found. Using default alpha:", best_alpha)

    best_lasso_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=best_alpha, max_iter=10000, random_state=42))
    ])

    best_lasso_model.fit(X_train, y_train)

prediction_df["LASSO Regression"] = best_lasso_model.predict(plot_df[feature_cols])

# ============================================================
# 6. XGBOOST
# ============================================================

from xgboost import XGBRegressor

print("Training XGBoost again to avoid joblib/XGBoost version mismatch...")

best_xgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 10.0,
    "reg_alpha": 0.1,
}

best_xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    **best_xgb_params
)

best_xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

prediction_df["XGBoost"] = best_xgb_model.predict(plot_df[feature_cols])

print("XGBoost prediction range:")
print(prediction_df["XGBoost"].min(), prediction_df["XGBoost"].max())

# ============================================================
# 7. TFT FULL BASELINE
# ============================================================

def extract_tft_decoder_time_idx(dataloader):
    """
    Extract decoder time_idx from a PyTorch Forecasting dataloader.
    For prediction length = 1, each row corresponds to one forecast timestamp.
    """
    time_indices = []

    for x, y in dataloader:
        decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()
        decoder_time_idx = decoder_time_idx.reshape(-1)
        time_indices.append(decoder_time_idx)

    return np.concatenate(time_indices)


try:
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer

    print("Preparing TFT dataloader...")

    df_tft = df_model.copy()

    df_tft["timestamp_utc"] = pd.to_datetime(df_tft["timestamp_utc"], utc=True)
    df_tft["data_date"] = pd.to_datetime(df_tft["data_date"])

    df_tft = df_tft.sort_values("timestamp_utc").reset_index(drop=True)

    df_tft["time_idx"] = (
        (df_tft["timestamp_utc"] - df_tft["timestamp_utc"].min())
        .dt.total_seconds() // 3600
    ).astype(int)

    df_tft["Region"] = df_tft["Region"].astype(str)

    df_tft["model_target_mw"] = np.where(
        df_tft["data_date"] < pd.Timestamp("2024-01-01"),
        df_tft["target_region_demand_mw"],
        df_tft["region_demand_mw"]
    )

    df_tft = df_tft.dropna(subset=["model_target_mw"]).copy()

    max_encoder_length = 168
    max_prediction_length = 1

    known_reals = [
        "time_idx",

        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "cooling_degree",
        "heating_degree",
        "temp_squared",

        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",

        "is_weekend",
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "is_holiday_period",
    ]

    known_reals = [col for col in known_reals if col in df_tft.columns]

    static_categoricals = ["Region"]

    valid_start_time_idx = df_tft.loc[
        df_tft["data_date"] >= valid_end_date,
        "time_idx"
    ].min()

    test_start_time_idx = df_tft.loc[
        df_tft["data_date"] >= valid_end_date,
        "time_idx"
    ].min()

    training = TimeSeriesDataSet(
        df_tft[df_tft["data_date"] < train_end_date],
        time_idx="time_idx",
        target="model_target_mw",
        group_ids=["Region"],

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        static_categoricals=static_categoricals,

        time_varying_known_reals=known_reals,

        time_varying_unknown_reals=[
            "model_target_mw"
        ],

        target_normalizer=GroupNormalizer(
            groups=["Region"],
            transformation="softplus"
        ),

        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    test_tft = TimeSeriesDataSet.from_dataset(
        training,
        df_tft[df_tft["data_date"] < test_end_date],
        min_prediction_idx=test_start_time_idx,
        stop_randomization=True
    )

    test_dataloader = test_tft.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=0
    )

    tft_model_path = f"{MODEL_OUTPUT_DIR}/temporal_fusion_transformer_one_step.ckpt"

    if os.path.exists(tft_model_path):
        loaded_tft = TemporalFusionTransformer.load_from_checkpoint(tft_model_path)
        loaded_tft.eval()

        print("Loaded TFT:", tft_model_path)

        tft_pred = loaded_tft.predict(test_dataloader)
        tft_pred_np = tft_pred.detach().cpu().numpy().reshape(-1)

        tft_time_idx = extract_tft_decoder_time_idx(test_dataloader)

        tft_pred_df = pd.DataFrame({
            "time_idx": tft_time_idx,
            "TFT Full Baseline": tft_pred_np
        })

        tft_time_map = df_tft[["time_idx", "timestamp_utc"]].drop_duplicates().copy()
        tft_time_map["timestamp_utc"] = pd.to_datetime(tft_time_map["timestamp_utc"], utc=True)

        tft_pred_df = tft_pred_df.merge(
            tft_time_map,
            on="time_idx",
            how="left"
        )

        tft_pred_df = tft_pred_df[
            (tft_pred_df["timestamp_utc"] >= plot_start) &
            (tft_pred_df["timestamp_utc"] < plot_end)
        ].copy()

        prediction_df = prediction_df.merge(
            tft_pred_df[["timestamp_utc", "TFT Full Baseline"]],
            on="timestamp_utc",
            how="left"
        )

    else:
        print("TFT checkpoint not found. Skipping TFT prediction:", tft_model_path)

except Exception as e:
    print("Skipping TFT because an error occurred:")
    print(e)

# ============================================================
# 8. SAVE PREDICTION TABLE
# ============================================================

csv_path = f"{OUTPUT_DIR}/one_week_test_predictions_selected_models.csv"
prediction_df.to_csv(csv_path, index=False)

print("Saved prediction table to:")
print(csv_path)

# display(prediction_df.head())

# ============================================================
# 9. PLOT COMBINED LINE CHART
# ============================================================

model_cols = [
    col for col in prediction_df.columns
    if col not in ["timestamp_utc", "timestamp_local", "Actual"]
]

plt.figure(figsize=(18, 7))

plt.plot(
    prediction_df["timestamp_local"],
    prediction_df["Actual"],
    label="Actual",
    linewidth=3,
    color="black"
)

for col in model_cols:
    plt.plot(
        prediction_df["timestamp_local"],
        prediction_df[col],
        label=col,
        linewidth=1.5,
        alpha=0.85
    )

plt.xlabel("Local Timestamp")
plt.ylabel("Electricity Demand (MW)")
plt.title("Actual vs Predicted Electricity Demand for One Week in Test Set")
plt.legend(ncol=2)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

combined_plot_path = f"{OUTPUT_DIR}/one_week_actual_vs_all_selected_models.png"
plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved combined plot to:")
print(combined_plot_path)

# ============================================================
# 10. PLOT EACH MODEL SEPARATELY
# ============================================================

for model_col in model_cols:
    plt.figure(figsize=(16, 5))

    plt.plot(
        prediction_df["timestamp_local"],
        prediction_df["Actual"],
        label="Actual",
        linewidth=3,
        color="black"
    )

    plt.plot(
        prediction_df["timestamp_local"],
        prediction_df[model_col],
        label=model_col,
        linewidth=2,
        alpha=0.9
    )

    plt.xlabel("Local Timestamp")
    plt.ylabel("Electricity Demand (MW)")
    plt.title(f"Actual vs {model_col} Prediction for One Week in Test Set")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_model_name = (
        model_col.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    single_plot_path = f"{OUTPUT_DIR}/one_week_actual_vs_{safe_model_name}.png"
    plt.savefig(single_plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", single_plot_path)

print("Done.")