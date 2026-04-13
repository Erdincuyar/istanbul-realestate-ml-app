
# ─────────────────────────────────────────────
# CONFIG — Sütun adlarını buradan ayarlayın
# ─────────────────────────────────────────────
TARGET        = "price"
FEATURES_NUM  = ["gross_sqm", "net_sqm", "total_rooms", "floor",
                 "total_floors", "building_age", 
                 "maintenance_fee"]
FEATURES_CAT  = ["district", "neighborhood", "floor_category",
                 "building_type", "building_condition", "heating_type",
                 "furnished", "usage_status", "is_in_complex", "orientation"]
CLUSTER_FEATS = ["gross_sqm", "total_rooms", "district_enc",
                 "building_age", "floor", "is_in_complex"]   # kümeleme için kullanılacak özellikler
N_CLUSTERS    = 5                           # elbow analizi ile belirleyin
UCUZ_ESIK     = -0.15                       # −%15 altı → fırsat
PAHALI_ESIK   =  0.20                       # +%15 üstü → pahalı


# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME & ÖN İŞLEME
# ─────────────────────────────────────────────
def yukle_ve_hazirla(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Ham veri: {df.shape[0]:,} satır, {df.shape[1]} sütun")

    # Sayısal sütunlarda medyan ile eksik doldur
    for col in FEATURES_NUM:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Kategorik sütunlarda mod ile eksik doldur
    for col in FEATURES_CAT:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # IQR ile aşırı fiyat değerlerini çıkar
    q1, q3 = df[TARGET].quantile([0.02, 0.98])
    df = df[df[TARGET].between(q1, q3)].copy()
    print(f"Outlier temizleme sonrası: {df.shape[0]:,} satır")

    # Kategorik encode
    le = LabelEncoder()
    for col in FEATURES_CAT:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    # Türetilmiş özellik: oda başına m²
    if "gross_sqm" in df.columns and "total_rooms" in df.columns:
        df["sqm_per_room"] = df["gross_sqm"] / df["total_rooms"].replace(0, 1)

    # Log fiyat (tahmin için daha stabil)
    df["log_price"] = np.log1p(df[TARGET])

    return df


# ─────────────────────────────────────────────
# 2. KÜMELEME — Elbow + K-Means
# ─────────────────────────────────────────────
def kume_bul(df: pd.DataFrame) -> pd.DataFrame:
    feats = [f for f in CLUSTER_FEATS if f in df.columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])

    # Elbow grafiği
    inertias = []
    K_range = range(2, 12)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertias, "bo-", linewidth=2)
    plt.xlabel("Küme sayısı (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Grafiği — Optimal k seçimi")
    plt.axvline(N_CLUSTERS, color="red", linestyle="--", alpha=0.7,
                label=f"Seçilen k={N_CLUSTERS}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("elbow.png", dpi=150)
    plt.close()
    print(f"Elbow grafiği kaydedildi: elbow.png")

    # Final kümeleme
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)

    # Küme profilleri
    profil = df.groupby("cluster").agg(
        adet=("cluster", "count"),
        medyan_fiyat=(TARGET, "median"),
        medyan_m2=("gross_sqm", "median"),
        ort_oda=("total_rooms", "mean"),
    ).round(0)
    print("\n=== Küme Profilleri ===")
    print(profil.to_string())

    return df


# ─────────────────────────────────────────────
# 3. FİYAT TAHMİN MODELİ — Her küme için XGBoost
# ─────────────────────────────────────────────
def model_egit(df: pd.DataFrame) -> pd.DataFrame:
    enc_cols = [c for c in df.columns if c.endswith("_enc")]
    model_feats = FEATURES_NUM + enc_cols + ["sqm_per_room"]
    model_feats = [f for f in model_feats if f in df.columns]

    df["predicted_price"] = np.nan
    df["shap_values"]     = None

    for cluster_id in sorted(df["cluster"].unique()):
        idx = df["cluster"] == cluster_id
        sub = df[idx].copy()

        if len(sub) < 50:
            # Küçük kümeler için global medyanı kullan
            df.loc[idx, "predicted_price"] = sub[TARGET].median()
            continue

        X = sub[model_feats]
        y = sub["log_price"]

        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

        # Cross-validation skoru
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        print(f"Küme {cluster_id}: n={len(sub):,}  CV-R²={cv_r2:.3f}")

        model.fit(X, y)
        log_pred = model.predict(X)
        df.loc[idx, "predicted_price"] = np.expm1(log_pred)

    return df


# ─────────────────────────────────────────────
# 4. FİYAT ADALETİ SKORU
# ─────────────────────────────────────────────
def skor_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    df["price_gap"]  = (df[TARGET] - df["predicted_price"]) / df["predicted_price"]

    conditions = [
        df["price_gap"] < UCUZ_ESIK,
        df["price_gap"] > PAHALI_ESIK,
    ]
    choices = ["Ucuz / Fırsat", "Pahalı"]
    df["etiket"] = np.select(conditions, choices, default="Adil fiyat")

    # Özet
    print("\n=== Fiyat Adaleti Dağılımı ===")
    print(df["etiket"].value_counts().to_string())

    return df


# ─────────────────────────────────────────────
# 5. SHAP AÇIKLAMASI
# ─────────────────────────────────────────────
def shap_analiz(df: pd.DataFrame):
    """
    En pahalı 500 ilanı seçip SHAP summary plot üretir.
    'Neden pahalı?' sorusunu görselleştirir.
    """
    enc_cols = [c for c in df.columns if c.endswith("_enc")]
    model_feats = FEATURES_NUM + enc_cols + ["sqm_per_room"]
    model_feats = [f for f in model_feats if f in df.columns]

    pahali = df[df["etiket"] == "Pahalı"].sample(
        min(500, (df["etiket"] == "Pahalı").sum()), random_state=42
    )

    X = pahali[model_feats]
    y = pahali["log_price"]

    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X, y)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("'Pahalı' etiketli ilanlar — En etkili özellikler (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
    print("SHAP grafiği kaydedildi: shap_summary.png")

    # Tek ilan açıklaması (ilk pahalı ilan)
    idx_ornek = pahali.index[0]
    print(f"\n=== Tek İlan Açıklaması (listing_id: {pahali.loc[idx_ornek, 'listing_id'] if 'listing_id' in pahali.columns else idx_ornek}) ===")
    print(f"  Gerçek fiyat   : {df.loc[idx_ornek, TARGET]:,.0f} TL")
    print(f"  Tahmin fiyat   : {df.loc[idx_ornek, 'predicted_price']:,.0f} TL")
    print(f"  Sapma          : %{df.loc[idx_ornek, 'price_gap']*100:+.1f}")

    row_shap = pd.Series(shap_values[0], index=model_feats).abs().sort_values(ascending=False)
    print("  Etkili özellikler:")
    for feat, val in row_shap.head(5).items():
        print(f"    {feat:<25} etki={val:.4f}")


# ─────────────────────────────────────────────
# 6. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────
def gorselleştir(df: pd.DataFrame):
    palette = {
        "Ucuz / Fırsat": "#2ecc71",
        "Adil fiyat"   : "#95a5a6",
        "Pahalı"       : "#e74c3c",
    }

    # Fiyat dağılımı (etiket bazlı)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for etiket, renk in palette.items():
        sub = df[df["etiket"] == etiket]["price_gap"]
        axes[0].hist(sub, bins=50, alpha=0.65, color=renk, label=etiket)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fiyat sapması (gerçek − tahmin) / tahmin")
    axes[0].set_ylabel("İlan sayısı")
    axes[0].set_title("Fiyat adaleti dağılımı")
    axes[0].legend()

    # İlçe bazlı pahalı ilan oranı
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
# 7. ÇIKTI — Skorlu veri seti
# ─────────────────────────────────────────────
def kaydet(df: pd.DataFrame):
    cols_out = ["listing_id", TARGET, "predicted_price", "price_gap",
                "etiket", "cluster", "district", "neighborhood",
                "gross_sqm", "total_rooms", "building_age"]
    cols_out = [c for c in cols_out if c in df.columns]
    df[cols_out].to_csv("fiyat_adaleti_skorlu.csv", index=False)
    print("Çıktı kaydedildi: fiyat_adaleti_skorlu.csv")


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────
if __name__ == "__main__":
    VERI_YOLU = "istanbul_apartment_prices_2026.csv"   # ← kendi dosyanızın yolunu yazın

    df = yukle_ve_hazirla(VERI_YOLU)
    df = kume_bul(df)
    df = model_egit(df)
    df = skor_hesapla(df)
    shap_analiz(df)
    gorselleştir(df)
    kaydet(df)

    print("\n✓ Pipeline tamamlandı.")
    print("  → fiyat_adaleti_skorlu.csv  (skorlu veri)")
    print("  → elbow.png                 (optimal küme sayısı)")
    print("  → shap_summary.png          (özellik önem grafiği)")
    print("  → fiyat_adaleti_gorsel.png  (dağılım + ilçe analizi)")
