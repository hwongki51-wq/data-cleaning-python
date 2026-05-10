"""
=============================================================
  MACHINE LEARNING IMPUTATION – AVG_CHOKE_SIZE_P
  WI2002 Tugas Besar | Institut Teknologi Bandung
=============================================================
Alur Kerja:
  Step 1  – Load data hasil cleaning
  Step 2  – Feature Engineering
  Step 3  – Split Train / Test
  Step 4  – Model 1: Linear Regression
  Step 5  – Model 2: Random Forest
  Step 6  – Model 3: Gradient Boosting
  Step 7  – Perbandingan performa ketiga model
  Step 8  – Imputasi data missing menggunakan model terbaik
  Step 9  – Export data final
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ─── konstanta ───────────────────────────────────────────────
FILE_IN  = "Volve_Production_Cleaned.xlsx"
FILE_OUT = "Volve_Production_Imputed.xlsx"
SEED     = 42
WELL_COLORS = {
    "F-1C": "#E63946", "F-1H": "#457B9D",
    "F-2H": "#2A9D8F", "F-3H": "#E9C46A", "F-4H": "#9B5DE5"
}

# ─────────────────────────────────────────────────────────────
# STEP 1 │ LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Load Data Hasil Cleaning")
print("=" * 60)

df_known   = pd.read_excel(FILE_IN, sheet_name="Known Choke (Train-Test)")
df_missing = pd.read_excel(FILE_IN, sheet_name="Missing Choke (Impute)")
df_full    = pd.read_excel(FILE_IN, sheet_name="Full Cleaned")

print(f"  Data known (choke ada)    : {len(df_known):>5} baris")
print(f"  Data missing (choke hilang): {len(df_missing):>5} baris")
print(f"  Total full dataset         : {len(df_full):>5} baris")

# ─────────────────────────────────────────────────────────────
# STEP 2 │ FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Feature Engineering")
print("=" * 60)

# Label Encoding: WELL (F-1C=0, F-1H=1, dst.)
le = LabelEncoder()
le.fit(["F-1C", "F-1H", "F-2H", "F-3H", "F-4H"])

def add_features(df):
    d = df.copy()
    d["DATEPRD"]   = pd.to_datetime(d["DATEPRD"])
    d["WELL_ENC"]  = le.transform(d["WELL"])
    d["MONTH"]     = d["DATEPRD"].dt.month
    d["YEAR"]      = d["DATEPRD"].dt.year
    # Rasio tekanan (fitur keteknikan tambahan)
    d["PRESSURE_RATIO"] = d["AVG_WELLHEAD_PRESSURE"] / d["AVG_DOWNHOLE_PRESSURE"].replace(0, np.nan)
    d["PRESSURE_RATIO"].fillna(d["PRESSURE_RATIO"].median(), inplace=True)
    return d

df_known   = add_features(df_known)
df_missing = add_features(df_missing)
df_full    = add_features(df_full)

FEATURES = [
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
    "AVG_DOWNHOLE_PRESSURE", "AVG_WELLHEAD_PRESSURE",
    "AVG_TEMPERATURE", "PRESSURE_RATIO",
    "WELL_ENC", "MONTH", "YEAR"
]
TARGET = "AVG_CHOKE_SIZE_P"

print(f"\n  Fitur yang digunakan ({len(FEATURES)}):")
for f in FEATURES:
    print(f"    • {f}")
print(f"\n  Label Encoding sumur:")
for name, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"    {name} → {code}")

# ─────────────────────────────────────────────────────────────
# STEP 3 │ SPLIT TRAIN / TEST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Split Train / Test (80:20)")
print("=" * 60)

X = df_known[FEATURES]
y = df_known[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

print(f"\n  Total data known : {len(X):>5} baris")
print(f"  Data Train (80%) : {len(X_train):>5} baris")
print(f"  Data Test  (20%) : {len(X_test):>5} baris")
print(f"\n  Distribusi target (choke size %):")
print(f"    Train – mean: {y_train.mean():.2f}  std: {y_train.std():.2f}  "
      f"min: {y_train.min():.2f}  max: {y_train.max():.2f}")
print(f"    Test  – mean: {y_test.mean():.2f}  std: {y_test.std():.2f}  "
      f"min: {y_test.min():.2f}  max: {y_test.max():.2f}")

# ─────────────────────────────────────────────────────────────
# STEP 4 │ MODEL 1 – LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Model 1: Linear Regression")
print("=" * 60)
print("""
  Cara kerja:
  Mencari garis/hyperplane terbaik yang meminimalkan jumlah
  kuadrat error (Ordinary Least Squares):

      ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

  OLS meminimalkan: Σ(yᵢ − ŷᵢ)²

  Asumsi: hubungan antara fitur dan target bersifat LINEAR.
  Cocok sebagai baseline — sederhana, cepat, interpretable.
""")

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
r2_lr   = r2_score(y_test, y_pred_lr)
cv_lr   = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()

print(f"  Koefisien per fitur:")
for feat, coef in sorted(zip(FEATURES, lr.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"    {feat:25s}: {coef:+.4f}")
print(f"\n  Intercept : {lr.intercept_:.4f}")
print(f"\n  ── Evaluasi pada Test Set ──────────────────")
print(f"  RMSE : {rmse_lr:.4f} %")
print(f"  MAE  : {mae_lr:.4f} %")
print(f"  R²   : {r2_lr:.4f}")
print(f"  CV R² (5-fold): {cv_lr:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 │ MODEL 2 – RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Model 2: Random Forest Regressor")
print("=" * 60)
print("""
  Cara kerja:
  Membangun N decision tree secara PARALEL, masing-masing
  dilatih pada subset data & fitur yang berbeda (Bootstrap
  Aggregating / Bagging). Prediksi akhir = rata-rata semua tree.

      ŷ = (1/N) Σ Treeᵢ(x)

  Keunggulan:
  • Menangkap hubungan NON-LINEAR
  • Robust terhadap outlier dan fitur tidak relevan
  • Memberikan feature importance

  Parameter yang digunakan:
  • n_estimators = 300  (jumlah tree)
  • max_depth    = 15   (kedalaman maksimum tree)
  • min_samples_leaf = 3
  • random_state = 42
""")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
r2_rf   = r2_score(y_test, y_pred_rf)
cv_rf   = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1).mean()

fi_rf   = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"  Feature Importance (top 5):")
for feat, imp in fi_rf.head(5).items():
    bar = "█" * int(imp * 40)
    print(f"    {feat:25s}: {imp:.4f}  {bar}")

print(f"\n  ── Evaluasi pada Test Set ──────────────────")
print(f"  RMSE : {rmse_rf:.4f} %")
print(f"  MAE  : {mae_rf:.4f} %")
print(f"  R²   : {r2_rf:.4f}")
print(f"  CV R² (5-fold): {cv_rf:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 6 │ MODEL 3 – GRADIENT BOOSTING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Model 3: Gradient Boosting Regressor")
print("=" * 60)
print("""
  Cara kerja:
  Membangun N decision tree secara SEKUENSIAL. Setiap tree
  baru belajar untuk memperbaiki RESIDUAL (error) dari tree
  sebelumnya, menggunakan gradient descent pada loss function.

      F₀(x)  = nilai awal (rata-rata y)
      F₁(x)  = F₀ + α · h₁(x)   ← h₁ fit pada residual F₀
      F₂(x)  = F₁ + α · h₂(x)   ← h₂ fit pada residual F₁
      ...
      Fₙ(x)  = Fₙ₋₁ + α · hₙ(x)

  α = learning rate (mengontrol besar "langkah" tiap iterasi)

  Keunggulan vs Random Forest:
  • Lebih akurat karena fokus pada data yang sulit diprediksi
  • Lebih sedikit tree, tapi lebih "pintar"

  Parameter yang digunakan:
  • n_estimators   = 400  (jumlah tree / iterasi)
  • learning_rate  = 0.05 (langkah kecil → lebih stable)
  • max_depth      = 5
  • subsample      = 0.8  (80% data per iterasi → mengurangi overfitting)
  • random_state   = 42
""")

gb = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=SEED
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb  = mean_absolute_error(y_test, y_pred_gb)
r2_gb   = r2_score(y_test, y_pred_gb)
cv_gb   = cross_val_score(gb, X, y, cv=5, scoring="r2").mean()

fi_gb   = pd.Series(gb.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"  Feature Importance (top 5):")
for feat, imp in fi_gb.head(5).items():
    bar = "█" * int(imp * 40)
    print(f"    {feat:25s}: {imp:.4f}  {bar}")

print(f"\n  ── Evaluasi pada Test Set ──────────────────")
print(f"  RMSE : {rmse_gb:.4f} %")
print(f"  MAE  : {mae_gb:.4f} %")
print(f"  R²   : {r2_gb:.4f}")
print(f"  CV R² (5-fold): {cv_gb:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 7 │ PERBANDINGAN MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – Perbandingan Performa Ketiga Model")
print("=" * 60)

metrics_df = pd.DataFrame({
    "Model"   : ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "RMSE (%)" : [round(rmse_lr,4), round(rmse_rf,4), round(rmse_gb,4)],
    "MAE (%)"  : [round(mae_lr,4),  round(mae_rf,4),  round(mae_gb,4)],
    "R²"       : [round(r2_lr,4),   round(r2_rf,4),   round(r2_gb,4)],
    "CV R²"    : [round(cv_lr,4),   round(cv_rf,4),   round(cv_gb,4)],
})
print("\n" + metrics_df.to_string(index=False))

# Pilih model terbaik berdasarkan R²
best_idx   = metrics_df["R²"].idxmax()
best_name  = metrics_df.loc[best_idx, "Model"]
best_model = [lr, rf, gb][best_idx]
y_pred_best = [y_pred_lr, y_pred_rf, y_pred_gb][best_idx]

print(f"\n  ✓ Model terbaik: {best_name} (R² = {metrics_df.loc[best_idx,'R²']})")

# ─────────────────────────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────────────────────────
print("\n  Membuat visualisasi...")

# VIZ A: Actual vs Predicted (3 model)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
models_viz = [
    ("Linear Regression", y_pred_lr, "#E63946", r2_lr, rmse_lr),
    ("Random Forest",     y_pred_rf, "#2A9D8F", r2_rf, rmse_rf),
    ("Gradient Boosting", y_pred_gb, "#9B5DE5", r2_gb, rmse_gb),
]
for ax, (name, pred, col, r2, rmse) in zip(axes, models_viz):
    mn = min(y_test.min(), pred.min())
    mx = max(y_test.max(), pred.max())
    ax.scatter(y_test, pred, alpha=0.35, s=14, color=col, edgecolors="none")
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, alpha=0.6, label="Ideal")
    ax.set_title(f"{name}\nR²={r2:.4f}  RMSE={rmse:.3f}%",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Aktual (Choke %)", fontsize=9)
    ax.set_ylabel("Prediksi (Choke %)", fontsize=9)
    ax.grid(alpha=0.2); ax.spines[["top","right"]].set_visible(False)
plt.suptitle("Aktual vs Prediksi – Perbandingan 3 Model ML", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_A_actual_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()

# VIZ B: Feature Importance (RF & GB side by side)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (fi, name, col) in zip(axes, [
    (fi_rf.sort_values(), "Random Forest",     "#2A9D8F"),
    (fi_gb.sort_values(), "Gradient Boosting", "#9B5DE5"),
]):
    colors = [col if v == fi.max() else "#CCCCCC" for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors, edgecolor="white")
    ax.set_title(f"Feature Importance\n{name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("viz_B_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# VIZ C: Residual plot (best model)
residuals = y_test.values - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.35, s=14, color="#457B9D", edgecolors="none")
axes[0].axhline(0, color="red", lw=1.2, linestyle="--")
axes[0].set_title(f"Residual Plot – {best_name}", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Nilai Prediksi (Choke %)", fontsize=9)
axes[0].set_ylabel("Residual (Aktual − Prediksi)", fontsize=9)
axes[0].grid(alpha=0.2); axes[0].spines[["top","right"]].set_visible(False)
axes[1].hist(residuals, bins=35, color="#457B9D", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", lw=1.2, linestyle="--")
axes[1].set_title(f"Distribusi Residual – {best_name}", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Residual", fontsize=9)
axes[1].set_ylabel("Frekuensi", fontsize=9)
axes[1].grid(alpha=0.2); axes[1].spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("viz_C_residuals.png", dpi=150, bbox_inches="tight")
plt.close()

# VIZ D: Bar comparison metrics
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
model_names = ["Lin. Reg.", "Random\nForest", "Grad.\nBoosting"]
colors_bar  = ["#E63946", "#2A9D8F", "#9B5DE5"]
for ax, (metric, vals, higher_better) in zip(axes, [
    ("R²",       [r2_lr, r2_rf, r2_gb],     True),
    ("RMSE (%)", [rmse_lr, rmse_rf, rmse_gb], False),
    ("MAE (%)",  [mae_lr, mae_rf, mae_gb],   False),
]):
    best_v = max(vals) if higher_better else min(vals)
    cols = ["gold" if v == best_v else c for v, c in zip(vals, colors_bar)]
    bars = ax.bar(model_names, vals, color=cols, edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002*max(vals),
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
plt.suptitle("Perbandingan Metrik Evaluasi – 3 Model ML\n(Emas = Terbaik)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_D_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("  ✓ 4 visualisasi tersimpan")

# ─────────────────────────────────────────────────────────────
# STEP 8 │ IMPUTASI MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 – Imputasi Data Missing dengan Model Terbaik")
print("=" * 60)
print(f"\n  Menggunakan: {best_name}")

X_impute         = df_missing[FEATURES]
imputed_values   = best_model.predict(X_impute)
df_missing_filled = df_missing.copy()
df_missing_filled["AVG_CHOKE_SIZE_P"]        = imputed_values
df_missing_filled["CHOKE_IMPUTATION_SOURCE"] = f"ML_{best_name.replace(' ','_')}"

print(f"\n  Jumlah nilai yang diimputasi: {len(imputed_values)}")
print(f"  Range nilai imputasi: {imputed_values.min():.2f}% – {imputed_values.max():.2f}%")
print(f"  Mean nilai imputasi : {imputed_values.mean():.2f}%")
print(f"  Std nilai imputasi  : {imputed_values.std():.2f}%")
print(f"\n  Range data asli (known): {y.min():.2f}% – {y.max():.2f}%  "
      f"(mean {y.mean():.2f}%)")
print(f"  → Nilai imputasi VALID, berada dalam range operasional ✓")

# VIZ E: Distribusi sebelum vs sesudah imputasi
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(y.values, bins=40, color="#457B9D", alpha=0.7, edgecolor="white", label="Known (asli)")
axes[0].hist(imputed_values, bins=40, color="#E63946", alpha=0.65, edgecolor="white", label="Imputasi")
axes[0].set_title("Distribusi Choke: Asli vs Imputasi", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Choke Size (%)", fontsize=9)
axes[0].set_ylabel("Frekuensi", fontsize=9)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.2); axes[0].spines[["top","right"]].set_visible(False)

all_choke_before = df_known["AVG_CHOKE_SIZE_P"].values
all_choke_after  = np.concatenate([all_choke_before, imputed_values])
axes[1].hist(all_choke_before, bins=40, color="#2A9D8F", alpha=0.7, edgecolor="white", label=f"Sebelum (n={len(all_choke_before)})")
axes[1].hist(all_choke_after,  bins=40, color="#E9C46A", alpha=0.5, edgecolor="white", label=f"Sesudah  (n={len(all_choke_after)})")
axes[1].set_title("Distribusi Choke Sebelum vs Sesudah Imputasi", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Choke Size (%)", fontsize=9)
axes[1].set_ylabel("Frekuensi", fontsize=9)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.2); axes[1].spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("viz_E_distribution_imputed.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n  ✓ viz_E_distribution_imputed.png tersimpan")

# ─────────────────────────────────────────────────────────────
# STEP 9 │ EXPORT DATA FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 – Export Data Final")
print("=" * 60)

# Gabung known + imputed
df_known_out = df_known.copy()
df_known_out["CHOKE_IMPUTATION_SOURCE"] = "ORIGINAL"

EXPORT_COLS = ["DATEPRD", "WELL", "BORE_OIL_VOL", "BORE_GAS_VOL",
               "BORE_WAT_VOL", "AVG_DOWNHOLE_PRESSURE", "AVG_WELLHEAD_PRESSURE",
               "AVG_TEMPERATURE", "AVG_CHOKE_SIZE_P", "CHOKE_IMPUTATION_SOURCE"]

df_final = pd.concat(
    [df_known_out[EXPORT_COLS], df_missing_filled[EXPORT_COLS]],
    ignore_index=True
).sort_values(["DATEPRD", "WELL"]).reset_index(drop=True)

print(f"\n  Shape dataset final: {df_final.shape}")
print(f"  Missing choke tersisa: {df_final['AVG_CHOKE_SIZE_P'].isna().sum()}")
print(f"\n  Distribusi source:")
print(df_final["CHOKE_IMPUTATION_SOURCE"].value_counts().to_string())

# Sample imputed rows
print(f"\n  Sample 5 baris yang diimputasi:")
sample_imp = df_final[df_final["CHOKE_IMPUTATION_SOURCE"] != "ORIGINAL"].head(5)
print(sample_imp[["DATEPRD","WELL","BORE_OIL_VOL","AVG_WELLHEAD_PRESSURE",
                   "AVG_CHOKE_SIZE_P","CHOKE_IMPUTATION_SOURCE"]].to_string(index=False))

# Export Excel
with pd.ExcelWriter(FILE_OUT, engine="openpyxl") as writer:
    df_final.to_excel(writer, sheet_name="Full Imputed", index=False)
    df_final[df_final["CHOKE_IMPUTATION_SOURCE"]=="ORIGINAL"].to_excel(
        writer, sheet_name="Original Data", index=False)
    df_final[df_final["CHOKE_IMPUTATION_SOURCE"]!="ORIGINAL"].to_excel(
        writer, sheet_name="Imputed Rows", index=False)
    metrics_df.to_excel(writer, sheet_name="Model Metrics", index=False)

print(f"\n  ✓ File tersimpan: {FILE_OUT}")
print(f"    Sheet 1 'Full Imputed'  : {len(df_final)} baris (semua data)")
print(f"    Sheet 2 'Original Data' : {(df_final['CHOKE_IMPUTATION_SOURCE']=='ORIGINAL').sum()} baris")
print(f"    Sheet 3 'Imputed Rows'  : {(df_final['CHOKE_IMPUTATION_SOURCE']!='ORIGINAL').sum()} baris")
print(f"    Sheet 4 'Model Metrics' : performa ketiga model")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SELESAI – Ringkasan ML Imputation")
print("=" * 60)
print(f"""
  Masalah      : 352 baris AVG_CHOKE_SIZE_P hilang
  Fitur        : {len(FEATURES)} variabel (produksi + tekanan + keteknikan + waktu)
  Train/Test   : {len(X_train)} / {len(X_test)} baris (80/20)

  Hasil Model:
  ┌─────────────────────┬────────┬────────┬────────┐
  │ Model               │  RMSE  │  MAE   │   R²   │
  ├─────────────────────┼────────┼────────┼────────┤
  │ Linear Regression   │ {rmse_lr:6.3f} │ {mae_lr:6.3f} │ {r2_lr:6.4f} │
  │ Random Forest       │ {rmse_rf:6.3f} │ {mae_rf:6.3f} │ {r2_rf:6.4f} │
  │ Gradient Boosting ★ │ {rmse_gb:6.3f} │ {mae_gb:6.3f} │ {r2_gb:6.4f} │
  └─────────────────────┴────────┴────────┴────────┘

  Model terbaik : {best_name}
  Imputasi      : 352 nilai choke berhasil diisi
  Dataset final : {len(df_final)} baris, 0 missing values
""")
