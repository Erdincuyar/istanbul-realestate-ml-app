"""
Fiyat Tahmin Pipeline — Istanbul Daire Fiyat Araligi Tahmini
=============================================================
2026 veri setiyle egitim -> artifact kaydetme -> yeni daire tahmini

Encoding stratejisi:
  - district / neighborhood  : TargetEncoder (log_price'in ortalamasini ogrenip kodlar)
  - diger kategorikler       : OrdinalEncoder

Quantile modeller (XGBoost reg:quantileerror):
  - Q10 -> alt sinir
  - Q50 -> nokta tahmin
  - Q90 -> ust sinir

Kullanim:
    python fiyat_tahmin_pipeline.py --egit
    python fiyat_tahmin_pipeline.py --tahmin
    python fiyat_tahmin_pipeline.py --tahmin-json '{"district":"Kadikoy","gross_sqm":100,...}'

Gereksinimler:
    pip install pandas numpy scikit-learn>=1.3 xgboost joblib
"""

import argparse
import json
import os
import warnings
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIG
# ==============================================================================
VERI_YOLU    = "istanbul_apartment_prices_2026.csv"
ARTIFACT_DIR = "model_artifacts"
TARGET       = "price"

# Sayisal ozellikler
FEATURES_NUM = [
    "gross_sqm", "net_sqm", "total_rooms", "floor",
    "total_floors", "building_age", "bathroom_count",
]

# Yuksek kardinalite -> TargetEncoder (log_price hedef alarak encode)
FEATURES_TARGET_ENC = ["district", "neighborhood"]

# Dusuk kardinalite -> OrdinalEncoder
FEATURES_ORD_ENC = [
    "heating_type", "furnished", "usage_status",
    "building_type", "building_condition", "floor_category", "orientation",
]

FEATURES_CAT = FEATURES_TARGET_ENC + FEATURES_ORD_ENC

QUANTILE_LEVELS = [10, 50, 90]

XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    tree_method="hist",
)


# ==============================================================================
# 1. VERI YUKLEME & TEMIZLIK
# ==============================================================================
def yukle_ve_temizle(path: str):
    df = pd.read_csv(path, low_memory=False)
    print(f"Ham veri         : {df.shape[0]:,} satir, {df.shape[1]} sutun")

    df = df.dropna(subset=[TARGET]).copy()
    print(f"Fiyatli satirlar : {df.shape[0]:,}")

    # Sayisallara zorunlu donusum
    sayi_cols = FEATURES_NUM + [TARGET, "is_in_complex", "maintenance_fee"]
    for col in sayi_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputation (median / mode)
    imputer_vals = {}

    for col in FEATURES_NUM:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            imputer_vals[col] = float(med)

    for col in FEATURES_CAT:
        if col in df.columns:
            modes = df[col].mode()
            fill  = modes.iloc[0] if len(modes) > 0 else "Unknown"
            df[col] = df[col].fillna(fill)
            imputer_vals[col] = str(fill)

    if "is_in_complex" in df.columns:
        med_ic = df["is_in_complex"].median()
        fill_ic = 0.0 if (isinstance(med_ic, float) and np.isnan(med_ic)) else float(med_ic)
        df["is_in_complex"] = df["is_in_complex"].fillna(fill_ic)
        imputer_vals["is_in_complex"] = fill_ic

    if "maintenance_fee" in df.columns:
        med_mf = df["maintenance_fee"].median()
        fill_mf = 0.0 if (isinstance(med_mf, float) and np.isnan(med_mf)) else float(med_mf)
        df["maintenance_fee"] = df["maintenance_fee"].fillna(fill_mf)
        imputer_vals["maintenance_fee"] = fill_mf

    # Outlier temizleme (%2 - %98)
    q_lo, q_hi = df[TARGET].quantile([0.02, 0.98])
    df = df[df[TARGET].between(q_lo, q_hi)].copy()

    # Fiziksel tutarsizliklar
    if "net_sqm" in df.columns:
        df = df[~(df["net_sqm"] > df["gross_sqm"] * 1.05)]
    if "floor" in df.columns and "total_floors" in df.columns:
        df = df[~(df["floor"] > df["total_floors"] + 2)]
    df = df[df["building_age"] <= 100]
    df = df[df["gross_sqm"] >= 20]
    if "total_rooms" in df.columns:
        df = df[df["total_rooms"].between(1, 20)]
    if "maintenance_fee" in df.columns:
        df = df[df["maintenance_fee"] <= 150_000]

    print(f"Temizlik sonrasi : {df.shape[0]:,} satir")
    return df, imputer_vals


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sqm_per_room"] = df["gross_sqm"] / df["total_rooms"].clip(lower=1)
    df["floor_ratio"]  = df["floor"] / df["total_floors"].clip(lower=1)
    df["yuksek_bina"]  = (df["total_floors"] >= 10).astype(int)
    df["ust_kat"]      = (df["floor"] >= df["total_floors"] - 1).astype(int)
    df["alt_kat"]      = (df["floor"] <= 0).astype(int)
    df["log_price"]    = np.log1p(df[TARGET])
    return df


# ==============================================================================
# 3. ENCODING
# ==============================================================================
def encode_fit(df: pd.DataFrame):
    """
    Egitim verisine fit eder, encoder'lari dondurur.
    TargetEncoder: district / neighborhood -> log_price hedefiyle kodlar
    OrdinalEncoder: kalan kategorikler
    """
    df = df.copy()

    # --- TargetEncoder (district, neighborhood) ---
    te_cols = [c for c in FEATURES_TARGET_ENC if c in df.columns]
    target_enc = TargetEncoder(smooth="auto", cv=5, random_state=42)
    te_vals = target_enc.fit_transform(
        df[te_cols].astype(str), df["log_price"]
    )
    for i, col in enumerate(te_cols):
        df[f"{col}_te"] = te_vals[:, i]

    # --- OrdinalEncoder (dusuk kardinalite) ---
    ord_cols = [c for c in FEATURES_ORD_ENC if c in df.columns]
    ord_enc  = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )
    ord_enc.fit(df[ord_cols].astype(str))
    ord_vals = ord_enc.transform(df[ord_cols].astype(str))
    for i, col in enumerate(ord_cols):
        df[f"{col}_enc"] = ord_vals[:, i]

    return df, target_enc, ord_enc, te_cols, ord_cols


def encode_transform(df: pd.DataFrame, target_enc, ord_enc, te_cols, ord_cols):
    """Kaydedilmis encoder'lari kullanarak yeni veriyi encode eder."""
    df = df.copy()

    te_present  = [c for c in te_cols  if c in df.columns]
    ord_present = [c for c in ord_cols if c in df.columns]

    if te_present:
        te_vals = target_enc.transform(df[te_present].astype(str))
        for i, col in enumerate(te_present):
            df[f"{col}_te"] = te_vals[:, i]

    if ord_present:
        ord_vals = ord_enc.transform(df[ord_present].astype(str))
        for i, col in enumerate(ord_present):
            df[f"{col}_enc"] = ord_vals[:, i]

    return df


# ==============================================================================
# 4. MODEL EGITIMI & ARTIFACT KAYDETME
# ==============================================================================
def model_egit_ve_kaydet(df, target_enc, ord_enc, te_cols, ord_cols, imputer_vals):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    te_feat_cols  = [f"{c}_te"  for c in te_cols  if f"{c}_te"  in df.columns]
    ord_feat_cols = [f"{c}_enc" for c in ord_cols if f"{c}_enc" in df.columns]
    derived       = ["sqm_per_room", "floor_ratio", "yuksek_bina",
                     "ust_kat", "alt_kat", "is_in_complex"]

    model_feats = FEATURES_NUM + derived + te_feat_cols + ord_feat_cols
    if "maintenance_fee" in df.columns:
        model_feats.append("maintenance_fee")
    model_feats = [f for f in model_feats if f in df.columns]

    X = df[model_feats].values.astype(np.float64)
    y = df["log_price"].values

    print(f"\nModel ozellikleri ({len(model_feats)}):")
    for i, f in enumerate(model_feats, 1):
        print(f"  {i:2d}. {f}")

    # --- Cross-validation ---
    cv_pred = cross_val_predict(
        xgb.XGBRegressor(**XGB_PARAMS), X, y, cv=5
    )
    cv_r2   = r2_score(y, cv_pred)
    cv_mape = mean_absolute_percentage_error(
        df[TARGET].values, np.expm1(cv_pred)
    )
    print(f"\nCV-R2  (log_price, 5-fold) : {cv_r2:.4f}")
    print(f"CV-MAPE (fiyat, 5-fold)    : {cv_mape*100:.2f}%")

    # --- Quantile modeller ---
    print()
    for qlevel in QUANTILE_LEVELS:
        q = qlevel / 100
        print(f"[Q{qlevel:02d}] %{qlevel} quantile egitiliyor...", end=" ", flush=True)
        try:
            params = {
                **XGB_PARAMS,
                "objective"     : "reg:quantileerror",
                "quantile_alpha": q,
            }
            m = xgb.XGBRegressor(**params)
            m.fit(X, y)
        except Exception:
            print("(quantile hatasi, standart reg kullaniliyor)", end=" ")
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X, y)

        fname = os.path.join(ARTIFACT_DIR, f"xgb_q{qlevel:02d}.json")
        m.save_model(fname)
        print(f"kaydedildi")

    # --- StandardScaler (sayisal kolonlar icin, mesafe bazli islemler) ---
    num_in_feats = [f for f in FEATURES_NUM if f in model_feats]
    scaler = StandardScaler()
    scaler.fit(df[num_in_feats])
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    # --- Encoder'lari kaydet ---
    joblib.dump(target_enc, os.path.join(ARTIFACT_DIR, "target_encoder.pkl"))
    joblib.dump(ord_enc,    os.path.join(ARTIFACT_DIR, "ord_encoder.pkl"))

    # --- Kategorik gecerli degerler (kullanici rehberi icin) ---
    cat_categories = {}
    for col in FEATURES_CAT:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            cat_categories[col] = [str(v) for v in vals]

    # --- Feature config ---
    config = {
        "model_feats"       : model_feats,
        "features_num"      : FEATURES_NUM,
        "features_cat"      : FEATURES_CAT,
        "features_target_enc": te_cols,
        "features_ord_enc"  : ord_cols,
        "num_in_feats"      : num_in_feats,
        "imputer_vals"      : imputer_vals,
        "cat_categories"    : cat_categories,
        "quantile_levels"   : QUANTILE_LEVELS,
        "cv_r2"             : round(cv_r2, 4),
        "cv_mape_pct"       : round(cv_mape * 100, 2),
        "target_stats": {
            "min" : float(df[TARGET].min()),
            "max" : float(df[TARGET].max()),
            "mean": float(df[TARGET].mean()),
            "p25" : float(df[TARGET].quantile(0.25)),
            "p75" : float(df[TARGET].quantile(0.75)),
        },
    }
    with open(os.path.join(ARTIFACT_DIR, "feature_config.json"), "w",
              encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'-'*50}")
    print("Artifactlar kaydedildi:")
    for fname in sorted(os.listdir(ARTIFACT_DIR)):
        size = os.path.getsize(os.path.join(ARTIFACT_DIR, fname))
        print(f"  {fname:<35} {size/1024:>7.1f} KB")

    return model_feats


# ==============================================================================
# 5. TAHMIN FONKSIYONU
# ==============================================================================
def tahmin_et(ozellikler: dict, artifact_dir: str = ARTIFACT_DIR) -> dict:
    """
    Yeni bir daire icin fiyat araligi tahmin eder.

    Parametre
    ---------
    ozellikler : dict
        Eksik olanlar egitim medyan/mode ile doldurulur.
        Ornek:
            {
                "district"      : "Kadikoy",
                "neighborhood"  : "Bostanci",
                "gross_sqm"     : 100,
                "total_rooms"   : 3,
                "floor"         : 4,
                "total_floors"  : 8,
                "building_age"  : 15,
                "is_in_complex" : 1,
                "heating_type"  : "Combi Boiler",
            }

    Dondurur
    --------
    dict: alt_sinir, tahmin, ust_sinir (float TL) + formatlı string versiyonlari
    """
    cfg_path = os.path.join(artifact_dir, "feature_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"'{cfg_path}' bulunamadi. Once --egit ile modeli egitin."
        )

    with open(cfg_path, encoding="utf-8") as fp:
        cfg = json.load(fp)

    target_enc = joblib.load(os.path.join(artifact_dir, "target_encoder.pkl"))
    ord_enc    = joblib.load(os.path.join(artifact_dir, "ord_encoder.pkl"))

    models = {}
    for qlevel in cfg["quantile_levels"]:
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(artifact_dir, f"xgb_q{qlevel:02d}.json"))
        models[qlevel] = m

    imp = cfg["imputer_vals"]

    # Satir olustur
    row = {}
    for col in cfg["features_num"]:
        row[col] = float(ozellikler.get(col, imp.get(col, 0.0)))
    for col in cfg["features_cat"]:
        row[col] = str(ozellikler.get(col, imp.get(col, "Unknown")))
    row["is_in_complex"]   = float(ozellikler.get("is_in_complex",  imp.get("is_in_complex", 0.0)))
    row["maintenance_fee"] = float(ozellikler.get("maintenance_fee", imp.get("maintenance_fee", 0.0)))

    df_row = pd.DataFrame([row])

    # Feature engineering
    gross_sqm    = df_row["gross_sqm"].iloc[0]
    total_rooms  = max(df_row["total_rooms"].iloc[0], 1)
    floor        = df_row["floor"].iloc[0]
    total_floors = max(df_row["total_floors"].iloc[0], 1)

    df_row["sqm_per_room"] = gross_sqm / total_rooms
    df_row["floor_ratio"]  = floor / total_floors
    df_row["yuksek_bina"]  = int(total_floors >= 10)
    df_row["ust_kat"]      = int(floor >= total_floors - 1)
    df_row["alt_kat"]      = int(floor <= 0)

    # Encoding
    te_cols  = cfg["features_target_enc"]
    ord_cols = cfg["features_ord_enc"]
    df_row = encode_transform(df_row, target_enc, ord_enc, te_cols, ord_cols)

    # Eksik feature kolonlari icin 0 doldur
    model_feats = cfg["model_feats"]
    for f in model_feats:
        if f not in df_row.columns:
            df_row[f] = 0.0

    X = df_row[model_feats].values.astype(np.float64)

    # Tahmin
    preds = {}
    for qlevel, m in models.items():
        log_pred = m.predict(X)[0]
        preds[qlevel] = float(np.expm1(log_pred))

    alt_sinir = preds[10]
    tahmin    = preds[50]
    ust_sinir = preds[90]

    def fmt(v: float) -> str:
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M TL"
        if v >= 1_000:
            return f"{v / 1_000:.0f}K TL"
        return f"{v:,.0f} TL"

    return {
        "alt_sinir"     : alt_sinir,
        "tahmin"        : tahmin,
        "ust_sinir"     : ust_sinir,
        "alt_sinir_fmt" : fmt(alt_sinir),
        "tahmin_fmt"    : fmt(tahmin),
        "ust_sinir_fmt" : fmt(ust_sinir),
        "aralik"        : f"{fmt(alt_sinir)} — {fmt(ust_sinir)}",
    }


# ==============================================================================
# 6. INTERAKTIF TAHMIN
# ==============================================================================
def interaktif_tahmin():
    cfg_path = os.path.join(ARTIFACT_DIR, "feature_config.json")
    if not os.path.exists(cfg_path):
        print(f"HATA: '{cfg_path}' bulunamadi.")
        print("Once modeli egitin: python fiyat_tahmin_pipeline.py --egit")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as fp:
        cfg = json.load(fp)

    imp   = cfg["imputer_vals"]
    stats = cfg["target_stats"]

    print("\n" + "=" * 60)
    print("     ISTANBUL DAIRE FIYAT TAHMIN SISTEMI")
    print("=" * 60)
    print(f"  Egitim verisi : {stats['min']/1e6:.1f}M - {stats['max']/1e6:.1f}M TL")
    print(f"  Model CV-R2   : {cfg.get('cv_r2', 'N/A')}")
    print(f"  Model CV-MAPE : %{cfg.get('cv_mape_pct', 'N/A')}")
    print("  (Enter = varsayilan deger)\n")

    SORULAR = [
        ("district",          "Ilce",
         "Besiktas / Kadikoy / Sisli / Uskudar ...", None),
        ("neighborhood",      "Mahalle",
         "Levent / Bostanci / Cihangir ...", None),
        ("gross_sqm",         "Brut m2",
         "ornek: 120", float),
        ("net_sqm",           "Net m2",
         "ornek: 95", float),
        ("total_rooms",       "Toplam oda sayisi",
         "3+1 -> 4, 2+1 -> 3", float),
        ("floor",             "Bulundugu kat",
         "ornek: 3", float),
        ("total_floors",      "Binadaki toplam kat",
         "ornek: 8", float),
        ("building_age",      "Bina yasi (yil)",
         "ornek: 10", float),
        ("bathroom_count",    "Banyo sayisi",
         "ornek: 2", float),
        ("is_in_complex",     "Site icinde mi?",
         "1=Evet  0=Hayir", float),
        ("heating_type",      "Isinma tipi",
         "Combi Boiler / Central / Air Conditioning / Floor Heating", None),
        ("furnished",         "Esya durumu",
         "Furnished / Unfurnished / Semi-Furnished", None),
        ("usage_status",      "Kullanim durumu",
         "Vacant / Owner-occupied / Tenant-occupied", None),
        ("building_type",     "Yapi tipi",
         "Reinforced Concrete / Masonry / Wood", None),
        ("building_condition","Yapi durumu",
         "New / Second-hand / Under Construction", None),
        ("floor_category",    "Kat kategorisi",
         "Ground Floor / Middle Floor / Top Floor / Garden Floor", None),
        ("orientation",       "Cephe (istege bagli)",
         "North / South / East / West", None),
        ("maintenance_fee",   "Aidat TL/ay (istege bagli)",
         "ornek: 2000", float),
    ]

    ozellikler = {}
    for key, etiket, ipucu, tip in SORULAR:
        varsayilan = imp.get(key, "")
        girdi = input(f"  {etiket:<28} [{varsayilan}]  ({ipucu}): ").strip()
        if girdi:
            if tip == float:
                try:
                    ozellikler[key] = float(girdi)
                except ValueError:
                    print(f"    [!] Gecersiz sayi, varsayilan kullaniliyor: {varsayilan}")
            else:
                ozellikler[key] = girdi

    print("\n  Hesaplaniyor...")
    sonuc = tahmin_et(ozellikler)

    print("\n" + "=" * 52)
    print(f"  ALT SINIR  (%10)  :  {sonuc['alt_sinir_fmt']:>15}")
    print(f"  TAHMIN     (%50)  :  {sonuc['tahmin_fmt']:>15}  <- nokta tahmin")
    print(f"  UST SINIR  (%90)  :  {sonuc['ust_sinir_fmt']:>15}")
    print(f"\n  FIYAT ARALIGI     :  {sonuc['aralik']}")
    print("=" * 52 + "\n")
    return sonuc


# ==============================================================================
# ANA AKIS
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Istanbul Daire Fiyat Tahmin Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python fiyat_tahmin_pipeline.py --egit
  python fiyat_tahmin_pipeline.py --tahmin
  python fiyat_tahmin_pipeline.py --tahmin-json '{"district":"Kadikoy","gross_sqm":100,"total_rooms":3}'
        """
    )
    parser.add_argument("--egit",        action="store_true",
                        help="Modeli egit ve artifactlari kaydet")
    parser.add_argument("--tahmin",      action="store_true",
                        help="Interaktif tahmin modu")
    parser.add_argument("--tahmin-json", metavar="JSON",
                        help="JSON string ile direkt tahmin")
    args = parser.parse_args()

    if args.egit:
        print("=" * 60)
        print("    MODEL EGITIM MODU")
        print(f"    Veri: {VERI_YOLU}")
        print("=" * 60 + "\n")

        df, imputer_vals                           = yukle_ve_temizle(VERI_YOLU)
        df                                         = feature_engineer(df)
        df, target_enc, ord_enc, te_cols, ord_cols = encode_fit(df)
        model_egit_ve_kaydet(
            df, target_enc, ord_enc, te_cols, ord_cols, imputer_vals
        )
        print("\n[OK] Egitim tamamlandi.")
        print("     Tahmin icin: python fiyat_tahmin_pipeline.py --tahmin")

    elif args.tahmin:
        interaktif_tahmin()

    elif args.tahmin_json:
        try:
            ozellikler = json.loads(args.tahmin_json)
        except json.JSONDecodeError as e:
            print(f"HATA: Gecersiz JSON -> {e}")
            sys.exit(1)
        sonuc = tahmin_et(ozellikler)
        print(json.dumps(sonuc, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
