"""
Fiyat Adaleti Skoru Pipeline — İlçe İçi KNN
İstanbul Apartman Verisi
 
Her daire sadece kendi ilçesindeki dairelerle karşılaştırılır.
"Beşiktaş'ta pahalı mı?" sorusu Beşiktaş standartlarıyla cevaplanır.
 
Gereksinimler:
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")
 
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TARGET       = "price"
 
FEATURES_NUM = ["gross_sqm", "net_sqm", "total_rooms", "floor",
                "total_floors", "building_age", "maintenance_fee"]
 
FEATURES_CAT = ["district", "neighborhood", "floor_category",
                "building_type", "building_condition", "heating_type",
                "furnished", "usage_status", "is_in_complex", "orientation"]
 
# İlçe içi KNN özellikleri — ilçe yok, sadece fiziksel özellikler
KNN_FEATS    = ["gross_sqm", "total_rooms", "building_age",
                "floor_ratio", "yuksek_bina", "is_in_complex",
                "total_floors", "neighborhood_enc"]
 
# İlçe başına minimum ilan — altındaysa komşu ilçeyle birleştir
MIN_ILCE_ILAN = 30
 
K_NEIGHBORS   = 20      # ilçe içinde kaç komşu — ilçe küçük olabilir, 20 yeterli
UCUZ_ESIK     = -0.20   # −%20 altı → fırsat
PAHALI_ESIK   =  0.20   # +%20 üstü → pahalı
 
 
# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME & ÖN İŞLEME
# ─────────────────────────────────────────────
def yukle_ve_hazirla(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Ham veri: {df.shape[0]:,} satır, {df.shape[1]} sütun")
 
    for col in FEATURES_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
 
    for col in FEATURES_CAT:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
 
    # Fiyat outlier temizleme (%2-%98)
    q1, q3 = df[TARGET].quantile([0.02, 0.98])
    df = df[df[TARGET].between(q1, q3)].copy()
 
    # Fiziksel imkansızlıkları temizle
    df = df[~(df["net_sqm"] > df["gross_sqm"])]
    df = df[~(df["floor"] > df["total_floors"])]
    df = df[df["building_age"] <= 100]
    df = df[df["gross_sqm"] >= 20]
    df = df[df["rooms"] <= 20]
    if "maintenance_fee" in df.columns:
        df = df[df["maintenance_fee"].isna() | (df["maintenance_fee"] <= 100000)]
 
    print(f"Temizlik sonrası: {df.shape[0]:,} satır")
 
    # Kategorik encode
    le = LabelEncoder()
    for col in FEATURES_CAT:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
 
    # Türetilmiş özellikler
    df["sqm_per_room"] = df["gross_sqm"] / df["total_rooms"].replace(0, 1)
    df["floor_ratio"]  = df["floor"] / df["total_floors"].replace(0, 1)
    df["yuksek_bina"]  = (df["total_floors"] >= 10).astype(int)
    df["ust_kat"]      = (df["floor"] >= df["total_floors"] - 1).astype(int)
    df["alt_kat"]      = (df["floor"] <= 0).astype(int)
 
    df["log_price"] = np.log1p(df[TARGET])
 
    return df
 
 
# ─────────────────────────────────────────────
# 2. İLÇE İÇİ KNN
# ─────────────────────────────────────────────
def ilce_ici_knn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her daire sadece kendi ilçesindeki dairelerle karşılaştırılır.
 
    Az ilanlı ilçeler (< MIN_ILCE_ILAN) için:
    → Global KNN'e fallback yapılır, güven skoru düşük tutulur.
 
    Çıktı sütunları:
    - referans_fiyat : ilçe içi komşu medyanı
    - knn_mesafe     : ortalama komşu uzaklığı (düşük = güvenilir)
    - knn_mod        : "ilce_ici" veya "global" (fallback)
    """
    feats = [f for f in KNN_FEATS if f in df.columns]
    print(f"\nKNN özellikleri: {feats}")
 
    # Temiz index — karışıklığı önle
    df = df.reset_index(drop=True)
 
    df["referans_fiyat"] = np.nan
    df["knn_mesafe"]     = np.nan
    df["knn_mod"]        = "ilce_ici"
 
    ilce_sayilari = df["district"].value_counts()
    kucuk_ilceler = ilce_sayilari[ilce_sayilari < MIN_ILCE_ILAN].index.tolist()
 
    if kucuk_ilceler:
        print(f"Az ilanlı ilçeler (global fallback): {kucuk_ilceler}")
 
    # ── İlçe içi KNN ──
    for ilce, grup in df.groupby("district"):
        pos_idx = grup.index.tolist()   # reset_index sonrası 0-tabanlı
 
        if len(grup) < MIN_ILCE_ILAN:
            df.loc[pos_idx, "knn_mod"] = "global"
            continue
 
        k = min(K_NEIGHBORS, len(grup) - 1)
 
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(grup[feats])
 
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(X_scaled)
 
        prices = grup[TARGET].values
        df.loc[pos_idx, "referans_fiyat"] = [
            np.median(prices[indices[i, 1:]])
            for i in range(len(grup))
        ]
        df.loc[pos_idx, "knn_mesafe"] = distances[:, 1:].mean(axis=1)
 
    # ── Global fallback: az ilanlı ilçeler ──
    fallback_mask = df["knn_mod"] == "global"
    if fallback_mask.sum() > 0:
        scaler   = StandardScaler()
        X_all    = scaler.fit_transform(df[feats])
        knn_glob = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1,
                                    metric="euclidean", n_jobs=-1)
        knn_glob.fit(X_all)
 
        fb_pos       = df[fallback_mask].index.tolist()
        X_fb         = X_all[fb_pos]
        dist_fb, idx_fb = knn_glob.kneighbors(X_fb)
 
        prices = df[TARGET].values
        df.loc[fb_pos, "referans_fiyat"] = [
            np.median(prices[idx_fb[i, 1:]])
            for i in range(len(fb_pos))
        ]
        df.loc[fb_pos, "knn_mesafe"] = dist_fb[:, 1:].mean(axis=1)
 
    ilce_ici_n = (df["knn_mod"] == "ilce_ici").sum()
    global_n   = (df["knn_mod"] == "global").sum()
    nan_n      = df["referans_fiyat"].isnull().sum()
    print(f"İlçe içi KNN  : {ilce_ici_n:,} ilan")
    print(f"Global fallback: {global_n:,} ilan")
    print(f"NaN kalan      : {nan_n:,} ilan")
 
    return df
 
 
# ─────────────────────────────────────────────
# 3. XGBoost — İlçe içi gruplu model
# ─────────────────────────────────────────────
def model_egit(df: pd.DataFrame):
    """
    Her ilçe için ayrı XGBoost modeli eğitir.
    Az ilanlı ilçeler için global model kullanılır.
    Final tahmin: KNN referans + XGBoost ortalaması.
    """
    enc_cols    = [c for c in df.columns if c.endswith("_enc")
                   if "district" not in c]   # district_enc modelde yok — zaten ilçe içindeyiz
    model_feats = FEATURES_NUM + enc_cols + ["sqm_per_room", "floor_ratio",
                                              "yuksek_bina", "ust_kat", "alt_kat"]
    model_feats = [f for f in model_feats if f in df.columns]
 
    df["xgb_predicted"] = np.nan
 
    def xgb_egit(X, y):
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(X, y)
        return model
 
    # Global model (fallback ve CV skoru için)
    X_all = df[model_feats]
    y_all = df["log_price"]
    global_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    cv_r2 = cross_val_score(global_model, X_all, y_all, cv=5, scoring="r2").mean()
    print(f"\nXGBoost Global CV-R²: {cv_r2:.3f}")
    global_model.fit(X_all, y_all)
 
    # İlçe bazlı model
    for ilce, grup in df.groupby("district"):
        idx = grup.index
        if len(grup) < MIN_ILCE_ILAN:
            df.loc[idx, "xgb_predicted"] = np.expm1(
                global_model.predict(grup[model_feats])
            )
            continue
 
        m = xgb_egit(grup[model_feats], grup["log_price"])
        df.loc[idx, "xgb_predicted"] = np.expm1(m.predict(grup[model_feats]))
 
    # Final tahmin: KNN + XGBoost ortalaması
    df["predicted_price"] = (df["referans_fiyat"] + df["xgb_predicted"]) / 2
 
    return df, global_model, model_feats
 
 
# ─────────────────────────────────────────────
# 4. FİYAT ADALETİ SKORU
# ─────────────────────────────────────────────
def skor_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    df["price_gap"] = (df[TARGET] - df["predicted_price"]) / df["predicted_price"]
 
    conditions = [
        df["price_gap"] < UCUZ_ESIK,
        df["price_gap"] > PAHALI_ESIK,
    ]
    choices = ["Ucuz / Fırsat", "Pahalı"]
    df["etiket"] = np.select(conditions, choices, default="Adil fiyat")
 
    # Güven skoru
    mesafe_max  = df["knn_mesafe"].quantile(0.95)
    df["guven"] = (1 - (df["knn_mesafe"] / mesafe_max).clip(0, 1)).round(2)
    # Fallback ilanlarda güveni düşür
    df.loc[df["knn_mod"] == "global", "guven"] *= 0.5
 
    print("\n=== Fiyat Adaleti Dağılımı ===")
    print(df["etiket"].value_counts().to_string())
    print(f"\nOrtalama güven skoru: {df['guven'].mean():.2f}")
 
    return df
 
 
# ─────────────────────────────────────────────
# 5. SHAP AÇIKLAMASI
# ─────────────────────────────────────────────
def shap_analiz(df: pd.DataFrame, model, model_feats: list):
    pahali = df[df["etiket"] == "Pahalı"].sample(
        min(500, (df["etiket"] == "Pahalı").sum()), random_state=42
    )
 
    X           = pahali[model_feats]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
 
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("'Pahalı' etiketli ilanlar — En etkili özellikler (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
    print("\nSHAP grafiği kaydedildi: shap_summary.png")
 
    idx = pahali.index[0]
    print(f"\n=== Örnek Pahalı İlan ===")
    print(f"  listing_id   : {df.loc[idx, 'listing_id'] if 'listing_id' in df.columns else idx}")
    print(f"  İlçe         : {df.loc[idx, 'district']}")
    print(f"  Gerçek fiyat : {df.loc[idx, TARGET]:,.0f} TL")
    print(f"  Tahmin fiyat : {df.loc[idx, 'predicted_price']:,.0f} TL")
    print(f"  Sapma        : %{df.loc[idx, 'price_gap']*100:+.1f}")
    print(f"  Güven        : {df.loc[idx, 'guven']:.2f}")
    print(f"  KNN modu     : {df.loc[idx, 'knn_mod']}")
 
 
# ─────────────────────────────────────────────
# 6. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────
def gorselleştir(df: pd.DataFrame):
    palette = {
        "Ucuz / Fırsat": "#2ecc71",
        "Adil fiyat"   : "#95a5a6",
        "Pahalı"       : "#e74c3c",
    }
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    for etiket, renk in palette.items():
        sub = df[df["etiket"] == etiket]["price_gap"]
        axes[0].hist(sub, bins=60, alpha=0.65, color=renk, label=etiket)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fiyat sapması (gerçek − tahmin) / tahmin")
    axes[0].set_ylabel("İlan sayısı")
    axes[0].set_title("Fiyat adaleti dağılımı")
    axes[0].legend()
 
    top_ilceler = df["district"].value_counts().head(15).index
    ilce_df = (
        df[df["district"].isin(top_ilceler)]
        .groupby("district")["etiket"]
        .apply(lambda x: (x == "Pahalı").mean() * 100)
        .sort_values(ascending=False)
        .reset_index()
    )
    ilce_df.columns = ["district", "pahali_oran"]
    sns.barplot(data=ilce_df, y="district", x="pahali_oran",
                palette="Reds_r", ax=axes[1])
    axes[1].set_xlabel("Pahalı ilan oranı (%)")
    axes[1].set_ylabel("")
    axes[1].set_title("İlçe bazlı pahalı ilan oranı (Top 15)")
 
    plt.tight_layout()
    plt.savefig("fiyat_adaleti_gorsel.png", dpi=150)
    plt.close()
    print("Görsel kaydedildi: fiyat_adaleti_gorsel.png")
 
 
# ─────────────────────────────────────────────
# 7. ÇIKTI
# ─────────────────────────────────────────────
def kaydet(df: pd.DataFrame):
    cols_out = ["listing_id", TARGET, "referans_fiyat", "xgb_predicted",
                "predicted_price", "price_gap", "etiket", "guven", "knn_mod",
                "district", "neighborhood", "gross_sqm", "total_rooms",
                "building_age", "floor", "total_floors", "floor_ratio",
                "is_in_complex", "knn_mesafe"]
    cols_out = [c for c in cols_out if c in df.columns]
    df[cols_out].to_csv("fiyat_adaleti_skorlu.csv", index=False)
    print("Çıktı kaydedildi: fiyat_adaleti_skorlu.csv")
 
 
# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────
if __name__ == "__main__":
    VERI_YOLU = r"C:\Users\lenova\OneDrive\Masaüstü\istanbul_apartment_prices.csv"
 
    df                 = yukle_ve_hazirla(VERI_YOLU)
    df                 = ilce_ici_knn(df)
    df, model, mfeats  = model_egit(df)
    df                 = skor_hesapla(df)
    shap_analiz(df, model, mfeats)
    gorselleştir(df)
    kaydet(df)
 
    print("\n✓ Pipeline tamamlandı.")
    print("  → fiyat_adaleti_skorlu.csv  (skorlu veri + güven + knn_mod)")
    print("  → shap_summary.png          (özellik önem grafiği)")
    print("  → fiyat_adaleti_gorsel.png  (dağılım + ilçe analizi)")