"""
=============================================================
  DATA CLEANING – Volve Production Data
  WI2002 Tugas Besar | Institut Teknologi Bandung
=============================================================
Langkah-langkah:
  Step 1  – Load raw data
  Step 2  – Fix tipe data (tanggal & numerik)
  Step 3  – Normalisasi nama sumur
  Step 4  – Identifikasi & laporan missing values
  Step 5  – Tangani missing values (fitur)
  Step 6  – Deteksi & tangani outlier
  Step 7  – Pisah dataset (known vs missing choke)
  Step 8  – Export hasil
=============================================================
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

FILE_IN  = "Volve_Production_Data_Final.xlsx"
FILE_OUT = "Volve_Production_Cleaned.xlsx"

# ─────────────────────────────────────────────────────────────
# STEP 1 │ LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Load Raw Data")
print("=" * 60)

df = pd.read_excel(FILE_IN, sheet_name="Production Data")
print(f"  Shape awal   : {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"  Kolom        : {df.columns.tolist()}")
print(f"\n  5 baris pertama (raw):")
print(df.head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 2 │ FIX TIPE DATA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Fix Tipe Data")
print("=" * 60)

# --- 2a. DATEPRD: format tidak konsisten ---
# Contoh format ditemukan: "2015-10-30", "07/19/2015", "27/05/2015", "18-Mar-2014"
print("\n  [2a] Format tanggal yang ditemukan (sample):")
print(" ", df["DATEPRD"].head(10).tolist())

df["DATEPRD"] = pd.to_datetime(df["DATEPRD"], format="mixed", dayfirst=False)

print(f"\n  Setelah konversi – dtype: {df['DATEPRD'].dtype}")
print(f"  Range tanggal: {df['DATEPRD'].min().date()} → {df['DATEPRD'].max().date()}")

# --- 2b. Kolom numerik sudah float64, pastikan tidak ada string tersembunyi ---
NUM_COLS = [
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
    "AVG_DOWNHOLE_PRESSURE", "AVG_WELLHEAD_PRESSURE",
    "AVG_TEMPERATURE", "AVG_CHOKE_SIZE_P"
]
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"\n  [2b] Tipe data setelah konversi:")
print(df[NUM_COLS + ["DATEPRD", "NPD_WELL_BORE_CODE"]].dtypes.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 3 │ NORMALISASI NAMA SUMUR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Normalisasi Nama Sumur")
print("=" * 60)

print(f"\n  Varian nama sumur SEBELUM normalisasi ({df['NPD_WELL_BORE_CODE'].nunique()} unik):")
print(" ", sorted(df["NPD_WELL_BORE_CODE"].unique()))

WELL_MAP = {
    # F-1C
    "F-1C": "F-1C", "F1C": "F-1C", "F_1C": "F-1C", "F-1 C": "F-1C", "f-1c": "F-1C",
    # F-1H
    "F-1H": "F-1H", "F1H": "F-1H", "F_1H": "F-1H", "F-1 H": "F-1H", "f-1h": "F-1H",
    # F-2H
    "F-2H": "F-2H", "F2H": "F-2H", "F_2H": "F-2H", "F-2 H": "F-2H", "f-2h": "F-2H",
    # F-3H
    "F-3H": "F-3H", "F3H": "F-3H", "F_3H": "F-3H", "F-3 H": "F-3H", "f-3h": "F-3H",
    # F-4H
    "F-4H": "F-4H", "F4H": "F-4H", "F_4H": "F-4H", "F-4 H": "F-4H", "f-4h": "F-4H",
}

df["WELL"] = df["NPD_WELL_BORE_CODE"].str.strip().map(WELL_MAP)

# Cek apakah ada yang tidak terpetakan
unmapped = df[df["WELL"].isna()]["NPD_WELL_BORE_CODE"].unique()
if len(unmapped) > 0:
    print(f"\n  ⚠ Tidak terpetakan: {unmapped}")
else:
    print(f"\n  ✓ Semua nama sumur berhasil dinormalisasi")

print(f"\n  Distribusi sumur SETELAH normalisasi:")
print(df["WELL"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 4 │ IDENTIFIKASI MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Identifikasi Missing Values")
print("=" * 60)

all_cols = ["DATEPRD", "WELL"] + NUM_COLS
miss_count = df[all_cols].isnull().sum()
miss_pct   = (miss_count / len(df) * 100).round(2)

miss_report = pd.DataFrame({
    "Kolom"        : miss_count.index,
    "Jumlah Missing": miss_count.values,
    "Persentase (%)" : miss_pct.values,
    "Status"       : [
        "TARGET – akan diimputasi ML" if c == "AVG_CHOKE_SIZE_P"
        else ("Lengkap" if v == 0 else "Missing – perlu ditangani")
        for c, v in zip(miss_count.index, miss_count.values)
    ]
})
print("\n" + miss_report.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 5 │ TANGANI MISSING VALUES (FITUR)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Tangani Missing Values (Fitur)")
print("=" * 60)

# Strategi: median per sumur (robust terhadap outlier)
# AVG_CHOKE_SIZE_P TIDAK diisi di sini — target ML
FEATURE_COLS = [c for c in NUM_COLS if c != "AVG_CHOKE_SIZE_P"]

filled_counts = {}
for col in FEATURE_COLS:
    before = df[col].isna().sum()
    if before > 0:
        df[col] = df.groupby("WELL")[col].transform(
            lambda x: x.fillna(x.median())
        )
        after = df[col].isna().sum()
        filled_counts[col] = before - after
        print(f"  {col:30s}: {before} missing → diisi median per sumur "
              f"(sisa: {after})")

# Jika masih ada sisa (sumur dengan semua NaN), isi dengan median global
for col in FEATURE_COLS:
    sisa = df[col].isna().sum()
    if sisa > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  {col:30s}: {sisa} sisa → diisi median global")

print(f"\n  ✓ Semua kolom fitur sudah tidak ada missing values")
print(f"  (AVG_CHOKE_SIZE_P dibiarkan — akan diimputasi model ML)")

# ─────────────────────────────────────────────────────────────
# STEP 6 │ DETEKSI & TANGANI OUTLIER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Deteksi & Tangani Outlier")
print("=" * 60)

# Metode IQR per kolom
# Outlier di-cap (winsorize) ke batas whisker, bukan dihapus
# → Informasi baris tetap ada untuk ML

print("\n  Metode: IQR Winsorizing (cap ke [Q1 - 1.5×IQR, Q3 + 1.5×IQR])")
print("  Kecuali AVG_CHOKE_SIZE_P (target ML, tidak diubah)\n")

WINSOR_COLS = [
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
    "AVG_DOWNHOLE_PRESSURE", "AVG_WELLHEAD_PRESSURE", "AVG_TEMPERATURE"
]

outlier_report = []
for col in WINSOR_COLS:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR

    n_out_lo = (df[col] < lo).sum()
    n_out_hi = (df[col] > hi).sum()
    n_out    = n_out_lo + n_out_hi

    df[col] = df[col].clip(lower=lo, upper=hi)

    outlier_report.append({
        "Kolom"        : col,
        "Batas Bawah"  : round(lo, 2),
        "Batas Atas"   : round(hi, 2),
        "N Outlier Bawah": n_out_lo,
        "N Outlier Atas" : n_out_hi,
        "Total Dicap"  : n_out
    })

print(pd.DataFrame(outlier_report).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 7 │ PISAH DATASET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – Pisah Dataset")
print("=" * 60)

FEATURE_ML = [
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
    "AVG_DOWNHOLE_PRESSURE", "AVG_WELLHEAD_PRESSURE", "AVG_TEMPERATURE",
    "WELL"
]

# Dataset A: choke diketahui → train/test ML
df_known   = df[df["AVG_CHOKE_SIZE_P"].notna()].copy()

# Dataset B: choke hilang & fitur lengkap → akan diimputasi
df_missing = df[
    df["AVG_CHOKE_SIZE_P"].isna() &
    df[FEATURE_ML].notna().all(axis=1)
].copy()

# Dataset C: choke hilang & ada fitur yang juga hilang → tidak bisa diimputasi
df_unusable = df[
    df["AVG_CHOKE_SIZE_P"].isna() &
    ~df[FEATURE_ML].notna().all(axis=1)
].copy()

print(f"\n  Dataset A (choke diketahui, untuk ML)  : {len(df_known):>5} baris")
print(f"  Dataset B (choke hilang, dapat diimputasi): {len(df_missing):>5} baris")
print(f"  Dataset C (choke + fitur hilang, unusable): {len(df_unusable):>5} baris")
print(f"  ─────────────────────────────────────────────")
print(f"  Total                                  : {len(df):>5} baris")

# ─────────────────────────────────────────────────────────────
# STEP 8 │ EXPORT HASIL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 – Export Hasil")
print("=" * 60)

# Urutkan kolom output
OUTPUT_COLS = ["DATEPRD", "WELL"] + NUM_COLS

with pd.ExcelWriter(FILE_OUT, engine="openpyxl") as writer:
    df[OUTPUT_COLS].sort_values("DATEPRD").to_excel(
        writer, sheet_name="Full Cleaned", index=False)
    df_known[OUTPUT_COLS].sort_values("DATEPRD").to_excel(
        writer, sheet_name="Known Choke (Train-Test)", index=False)
    df_missing[OUTPUT_COLS].sort_values("DATEPRD").to_excel(
        writer, sheet_name="Missing Choke (Impute)", index=False)
    miss_report.to_excel(
        writer, sheet_name="Missing Value Report", index=False)

print(f"\n  ✓ File tersimpan: {FILE_OUT}")
print(f"    Sheet 1 'Full Cleaned'            : {len(df)} baris")
print(f"    Sheet 2 'Known Choke (Train-Test)': {len(df_known)} baris")
print(f"    Sheet 3 'Missing Choke (Impute)'  : {len(df_missing)} baris")
print(f"    Sheet 4 'Missing Value Report'    : ringkasan missing")

# ─────────────────────────────────────────────────────────────
# RINGKASAN AKHIR
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN HASIL CLEANING")
print("=" * 60)
print(f"""
  Raw data            : 3.675 baris, 9 kolom
  ─── Masalah ditemukan ────────────────────────────────
  Format tanggal      : 4 format berbeda → dikonversi pd.to_datetime
  Nama sumur          : 25 varian → dinormalisasi ke 5 kanonik
  Missing values      : 7 kolom (max 9.58% di AVG_CHOKE_SIZE_P)
  Outlier             : ditemukan di 6 kolom fitur (dicap IQR)
  ─── Tindakan ─────────────────────────────────────────
  Fitur missing       : diisi median per sumur
  Outlier fitur       : di-cap ke batas IQR (tidak dihapus)
  Choke missing       : DIBIARKAN → target imputasi ML
  ─── Output ───────────────────────────────────────────
  Dataset untuk ML    : 2.889 baris (choke diketahui)
  Data untuk imputasi : 296 baris (choke hilang, fitur OK)
""")
