import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os

#make sure it runs in vscode
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

features_df = pd.read_csv("features.csv")
features_df["date"] = pd.to_datetime(features_df["date"])

print("Features loaded:", len(features_df), "rows")

#extract calendar year and week number from each date so daily rows can be grouped into weeks
features_df["year"] = features_df["date"].dt.isocalendar().year.astype(int)
features_df["week"] = features_df["date"].dt.isocalendar().week.astype(int)

# duplicate is_holiday before aggregation: after grouping, max() gives a 1/0 flag for whether any day in the week was a holiday, and sum() counts how many were
features_df["num_holiday_days"] = features_df["is_holiday"]

cuisine_cols = [
    "cuisine_pizza", "cuisine_chinese", "cuisine_italian", "cuisine_mexican",
    "cuisine_american", "cuisine_asian", "cuisine_japanese", "cuisine_indian",
    "cuisine_thai", "cuisine_vietnamese", "cuisine_korean", "cuisine_spanish",
    "cuisine_french", "cuisine_middle_eastern", "cuisine_burger", "cuisine_seafood",
    "cuisine_vegetarian", "cuisine_vegan", "cuisine_cafe", "cuisine_deli"
]


agg_dict = { #since we changed the daily review prediction into weekly review, we have to aggregate the different categories differently
    "daily_review_count": "sum", #becomes weekly_review_count below
    "month":              "first", #takes the month from the first day of the week
    "season":             lambda x: x.mode()[0], #takes the most common season that week
    "temp_max":           "max", #hottest day
    "temp_avg":           "mean", #average temp across week
    "temp_min":           "min", #coldest temp
    "weather_code":       lambda x: x.mode()[0], #most common weather type during the week
    "precipitation":      "sum", #sum the total rainfall accumlated over the week
    "has_precipitation":  "max", #flagged with any day has 
    "heavy_precipitation":"max", #flagged with any day has 
    "is_freezing":        "max", #flagged with any day has 
    "is_holiday":         "max", #flagged with any day has 
    "num_holiday_days":   "sum", #counts how many days in the week are holidays
    "moving_avg_14_day":  "mean", #averages the average lag from SQL queries across the week for 2 weeks
    "moving_avg_30_day":  "mean", #averages the average lag from SQL queries across the week for 4 weeks
    "restaurant_stars":   "first", #restaurant level data that doesn't change
    "total_review_count": "first", #restaurant level data that doesn't change
    "nearby_restaurant_count": "first", #restaurant level data that doesn't change
    "avg_nearby_stars":   "first",#restaurant level data that doesn't change
}

#aggregation for the type of cuisine across week (just takes the first)
for c in cuisine_cols:
    agg_dict[c] = "first" 

weekly_df = features_df.groupby(["business_id", "year", "week"]).agg(agg_dict).reset_index().rename(columns={"daily_review_count": "weekly_review_count"})

#chronological order 
weekly_df = weekly_df.sort_values(["business_id", "year", "week"])
weekly_df["prev_week_review_count"] = (weekly_df.groupby("business_id")["weekly_review_count"].shift(1))

#drop the first week's lag (will be null because no previous weeks)
weekly_clean = weekly_df.dropna(subset=["prev_week_review_count"]).dropna()

print(f"Weekly rows after cleaning: {len(weekly_clean)}")
print(f"Weekly review count stats:\n{weekly_clean['weekly_review_count'].describe()}\n")

numeric_features = [
    "week", "month", "year",
    "prev_week_review_count", "moving_avg_14_day", "moving_avg_30_day",
    "temp_max", "temp_avg", "temp_min", "weather_code", "precipitation",
    "has_precipitation", "heavy_precipitation", "is_freezing",
    "is_holiday", "num_holiday_days",
    "restaurant_stars", "total_review_count",
    "nearby_restaurant_count", "avg_nearby_stars",
] + cuisine_cols

season_dummies = pd.get_dummies(weekly_clean["season"], prefix="season", drop_first=True)

#training
X = weekly_clean[numeric_features].copy()
X = pd.concat([X, season_dummies], axis=1)

y = weekly_clean["weekly_review_count"]

print(f"Features for ML: {X.shape[1]} columns")
print(f"Target variable: weekly_review_count")

train_mask = weekly_clean["year"] < 2019
test_mask  = weekly_clean["year"] == 2019

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain set: {len(X_train)} rows (2013-2018)")
print(f"Test set:  {len(X_test)} rows (2019)")


#linear regression model
print("\n" + "="*50)
print("Linear Regression")
print("="*50)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"MAE: {mae_lr:.3f}")
print(f"RMSE: {rmse_lr:.3f}")
print(f"R²: {r2_lr:.3f}")

#random forest model
print("\n" + "="*50)
print("Random Forest")
print("="*50)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print(f"MAE: {mae_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")
print(f"R²: {r2_rf:.3f}")

#XGBoost model
print("\n" + "="*50)
print("XGBoost")
print("="*50)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, verbose=False)
y_pred_xgb = xgb_model.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb   = r2_score(y_test, y_pred_xgb)

print(f"MAE: {mae_xgb:.3f}")
print(f"RMSE: {rmse_xgb:.3f}")
print(f"R²: {r2_xgb:.3f}")

#comparing the models
print("\n" + "="*50)
print("Comparing the Models")
print("="*50)
print(f"Linear Regression R²: {r2_lr:.3f}")
print(f"Random Forest R²:     {r2_rf:.3f}")
print(f"XGBoost R²:           {r2_xgb:.3f}")

models = {"Linear Regression": r2_lr, "Random Forest": r2_rf, "XGBoost": r2_xgb}
winner = max(models, key=models.get)
print(f"\nWinner: {winner} (R² = {models[winner]:.3f})")

#after seeing that xgboost was our most successful model, we added this code to see which features made the most impact
print("\n" + "="*50)
print("Feature Importance (XGBoost)")
print("="*50)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print(feature_importance.head(10))

#plots
feat_names = np.array(X.columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#actual vs predicted (XGBoost)
axes[0, 0].scatter(y_test, y_pred_xgb, alpha=0.5, color="green")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel("Actual Weekly Review Count")
axes[0, 0].set_ylabel("Predicted Weekly Review Count")
axes[0, 0].set_title(f"XGBoost: Actual vs Predicted (R² = {r2_xgb:.3f})")
axes[0, 0].grid(True, alpha=0.3)

#comparing R^2 scores across models to see how they all compare
model_names = ["Linear\nRegression", "Random\nForest", "XGBoost"]
r2_scores = [r2_lr, r2_rf, r2_xgb]
colors = ["blue", "orange", "green"]
axes[0, 1].bar(model_names, r2_scores, color=colors, alpha=0.7)
axes[0, 1].set_ylabel("R² Score")
axes[0, 1].set_title("Model Comparison")
axes[0, 1].set_ylim([0, max(r2_scores) * 1.2])
for i, v in enumerate(r2_scores):
    axes[0, 1].text(i, v + 0.002, f"{v:.3f}", ha="center")

#demand per week
test_data = weekly_clean[test_mask].copy()
test_data["predicted"] = y_pred_xgb
monthly_demand = test_data.groupby("month")[["weekly_review_count", "predicted"]].mean()
monthly_demand.columns = ["Actual", "Predicted"]
monthly_demand.plot(kind="bar", ax=axes[1, 0], color=["steelblue", "green"], alpha=0.7)
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Average Weekly Reviews")
axes[1, 0].set_title("Demand by Month (XGBoost)")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].legend()

#demand per weather condition
test_data["has_rain"] = test_data["has_precipitation"].map({1: "Rainy Week", 0: "Dry Week"})
weather_demand = test_data.groupby("has_rain")[["weekly_review_count", "predicted"]].mean()
weather_demand.columns = ["Actual", "Predicted"]
weather_demand.plot(kind="bar", ax=axes[1, 1], color=["steelblue", "green"], alpha=0.7)
axes[1, 1].set_xlabel("Weather Condition")
axes[1, 1].set_ylabel("Average Weekly Reviews")
axes[1, 1].set_title("Demand by Weather Condition (XGBoost)")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("ml_results.png", dpi=100)
print("\nPlot saved as ml_results.png")
plt.close(fig)


#top 15 feature importances for tree models
for model, name, out in (
    (rf_model,  "Random forest", "ml_importance_random_forest.png"),
    (xgb_model, "XGBoost",       "ml_importance_xgboost.png"),
):
    imp = model.feature_importances_
    top = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_names[top][::-1], imp[top][::-1], color="steelblue", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_title(f"{name}: top 15 features")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)

print("Saved importance plots.")


#time series: avg actual vs predicted per week across 2019 (xgboost)
ts = weekly_clean[test_mask].copy()
ts["predicted"] = y_pred_xgb
ts_agg = ts.groupby("week")[["weekly_review_count", "predicted"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ts_agg["week"], ts_agg["weekly_review_count"], label="Actual", color="steelblue")
ax.plot(ts_agg["week"], ts_agg["predicted"], label="Predicted", color="green", linestyle="--")
ax.set_xlabel("Week of Year (2019)")
ax.set_ylabel("Avg Weekly Reviews")
ax.set_title("Actual vs Predicted Weekly Reviews Over Time (2019): XGBoost")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plot_timeseries_2019.png", dpi=100)
plt.close(fig)

print("Saved: plot_residuals_xgb.png, plot_timeseries_2019.png")

#actual vs predicted for linear regression and random forest
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_pred, name, color in [
    (axes[0], y_pred_lr, "Linear Regression", "blue"),
    (axes[1], y_pred_rf, "Random Forest",     "orange"),
]:
    r2 = r2_score(y_test, y_pred)
    ax.scatter(y_test, y_pred, alpha=0.5, color=color)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Weekly Review Count")
    ax.set_ylabel("Predicted Weekly Review Count")
    ax.set_title(f"{name}: Actual vs Predicted (R² = {r2:.3f})")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("plot_actual_vs_predicted_lr_rf.png", dpi=100)
plt.close(fig)

print("Saved: plot_actual_vs_predicted_lr_rf.png")

#time series comparison for linear regression and random forest
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for ax, y_pred, name, color in [
    (axes[0], y_pred_lr, "Linear Regression", "blue"),
    (axes[1], y_pred_rf, "Random Forest", "orange"),
]:
    ts2 = weekly_clean[test_mask].copy()
    ts2["predicted"] = y_pred
    ts2_agg = ts2.groupby("week")[["weekly_review_count", "predicted"]].mean().reset_index()
    ax.plot(ts2_agg["week"], ts2_agg["weekly_review_count"], label="Actual", color="steelblue")
    ax.plot(ts2_agg["week"], ts2_agg["predicted"], label="Predicted", color=color, linestyle="--")
    ax.set_xlabel("Week of Year (2019)")
    ax.set_ylabel("Avg Weekly Reviews")
    ax.set_title(f"{name}: Actual vs Predicted Over Time (2019)")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("plot_timeseries_lr_rf.png", dpi=100)
plt.close(fig)

print("Saved: plot_timeseries_lr_rf.png")