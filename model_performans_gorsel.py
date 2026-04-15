"""
Model Performans Gorsellestirme
================================
Egitilmis fiyat tahmin modelinin dogruluk analizini gorsel olarak sunar.

Kullanim:
    python model_performans_gorsel.py
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import xgboost as xgb

from fiyat_tahmin_pipeline import yukle_ve_temizle, feature_engineer, encode_transform

warnings.filterwarnings("ignore")

# ── Stil ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "figure.facecolor" : "#F8F9FA",
    "axes.facecolor"   : "#F8F9FA",
})

RENK_IYI   = "#2ECC71"
RENK_ORTA  = "#F39C12"
RENK_KOTU  = "#E74C3C"
RENK_MAVI  = "#3498DB"
RENK_MOR   = "#9B59B6"
RENK_KOYU  = "#2C3E50"

# ==============================================================================
# 1. VERİ & MODEL YUKLE
# ==============================================================================
print("Veri ve model yukleniyor...")

df, imp = yukle_ve_temizle("istanbul_apartment_prices_2026.csv")
df      = feature_engineer(df)

target_enc = joblib.load("model_artifacts/target_encoder.pkl")
ord_enc    = joblib.load("model_artifacts/ord_encoder.pkl")

with open("model_artifacts/feature_config.json", encoding="utf-8") as f:
    cfg = json.load(f)

df = encode_transform(df, target_enc, ord_enc,
                      cfg["features_target_enc"], cfg["features_ord_enc"])
df = df.reset_index(drop=True)

models = {}
for ql in [10, 50, 90]:
    m = xgb.XGBRegressor()
    m.load_model(f"model_artifacts/xgb_q{ql:02d}.json")
    models[ql] = m

X      = df[cfg["model_feats"]].values.astype(float)
preds  = {ql: np.expm1(m.predict(X)) for ql, m in models.items()}

y_true = df["price"].values
y_pred = preds[50]
y_lo   = preds[10]
y_hi   = preds[90]

hata_pct   = np.abs(y_pred - y_true) / y_true * 100
aralik_ici = (y_lo <= y_true) & (y_true <= y_hi)

df["tahmin"]     = y_pred
df["hata_pct"]   = hata_pct
df["aralik_ici"] = aralik_ici
df["y_lo"]       = y_lo
df["y_hi"]       = y_hi

global_mape   = mean_absolute_percentage_error(y_true, y_pred) * 100
global_medape = np.median(hata_pct)
global_r2     = r2_score(y_true, y_pred)
global_hit    = aralik_ici.mean() * 100

print(f"  R2={global_r2:.3f}  MAPE={global_mape:.1f}%  "
      f"MedAPE={global_medape:.1f}%  Aralik={global_hit:.1f}%")

# İlçe metrikleri
ilce_stats = (
    df.groupby("district")
    .apply(lambda g: pd.Series({
        "n"          : len(g),
        "mape"       : mean_absolute_percentage_error(g["price"], g["tahmin"]) * 100,
        "medape"     : np.median(np.abs(g["tahmin"] - g["price"]) / g["price"] * 100),
        "r2"         : r2_score(g["price"], g["tahmin"]) if len(g) >= 5 else np.nan,
        "hit_rate"   : g["aralik_ici"].mean() * 100,
        "aralik_genislik": ((g["y_hi"] - g["y_lo"]) / g["tahmin"] * 100).mean(),
    }))
    .query("n >= 30")
    .sort_values("mape")
    .reset_index()
)

# ==============================================================================
# 2. GRAFIK DUZENLEMESI — 3x3 izgara
# ==============================================================================
fig = plt.figure(figsize=(20, 27), facecolor="#F8F9FA")
fig.suptitle(
    "Istanbul Daire Fiyat Tahmin Modeli — Performans Raporu",
    fontsize=20, fontweight="bold", color=RENK_KOYU, y=0.992
)

# ── Üst özet bandı ────────────────────────────────────────────────────────────
ax_ozet = fig.add_axes([0.04, 0.935, 0.92, 0.048])
ax_ozet.set_xlim(0, 1)
ax_ozet.set_ylim(0, 1)
ax_ozet.axis("off")
ax_ozet.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1,
    boxstyle="round,pad=0.01",
    facecolor="#EBF5FB", edgecolor="#AED6F1", linewidth=1.2
))

cv_r2_val   = cfg.get("cv_r2", "N/A")
cv_mape_val = cfg.get("cv_mape_pct", "N/A")

ozet_metin = (
    f"Model:  XGBoost Kantil Regresyon  |  "
    f"Hedef:  log(Fiyat)  →  Q10 / Q50 / Q90 bant tahmini  |  "
    f"Egitim verisi:  {len(df):,} daire  (Istanbul, 2026)  |  "
    f"In-sample  R²={global_r2:.3f}   MAPE=%{global_mape:.1f}   MedAPE=%{global_medape:.1f}  |  "
    f"CV-5  R²={cv_r2_val}   MAPE=%{cv_mape_val}  |  "
    f"Q10–Q90 bant isabeti=%{global_hit:.1f}"
)
ax_ozet.text(
    0.5, 0.56, ozet_metin,
    ha="center", va="center", fontsize=9.5,
    color=RENK_KOYU, fontweight="bold"
)

panel_rehberi = (
    "A — Gercek vs Tahmin scatter  (renk = hata seviyesi)  |  "
    "B — Temel metrik karti  |  "
    "C — Hata yuzdesinin dagilimi  |  "
    "D — Kumulatif hata egrisi  (kac ev hangi hata altinda?)  |  "
    "E — Ornek evler icin Q10-Q90 fiyat bandi  |  "
    "F — Ilce bazli MAPE sirasi  |  "
    "G — En etkili 12 ozellik"
)
ax_ozet.text(
    0.5, 0.14, panel_rehberi,
    ha="center", va="center", fontsize=8.5,
    color="#555555", style="italic"
)

# ─────────────────────────────────────────────────────────────────────────────
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.45, wspace=0.35,
    left=0.07, right=0.97, top=0.925, bottom=0.04
)

# ── A: Gerçek vs Tahmin scatter ───────────────────────────────────────────────
ax_scatter = fig.add_subplot(gs[0, :2])

lim_lo = min(y_true.min(), y_pred.min()) * 0.9
lim_hi = max(y_true.max(), y_pred.max()) * 1.05

renkler = np.where(hata_pct < 15, RENK_IYI,
          np.where(hata_pct < 30, RENK_ORTA, RENK_KOTU))

ax_scatter.scatter(
    y_true / 1e6, y_pred / 1e6,
    c=renkler, alpha=0.4, s=12, linewidths=0
)
ax_scatter.plot(
    [lim_lo / 1e6, lim_hi / 1e6],
    [lim_lo / 1e6, lim_hi / 1e6],
    color=RENK_KOYU, lw=1.5, linestyle="--", label="Mukemmel tahmin"
)
ax_scatter.set_xlabel("Gercek Fiyat (M TL)", fontsize=11)
ax_scatter.set_ylabel("Tahmin Fiyat (M TL)", fontsize=11)
ax_scatter.set_title(
    "A  —  Gercek vs Tahmin Fiyat\n"
    "Diyagonale yakin nokta = daha dogru tahmin.  Yesil <15%  |  Turuncu 15-30%  |  Kirmizi >30%",
    fontsize=12, fontweight="bold", loc="left"
)
ax_scatter.set_xlabel("Gercek Fiyat (M TL)", fontsize=11)
ax_scatter.set_xlim(lim_lo / 1e6, lim_hi / 1e6)
ax_scatter.set_ylim(lim_lo / 1e6, lim_hi / 1e6)

legend_handles = [
    mpatches.Patch(color=RENK_IYI,  label="Hata < %15"),
    mpatches.Patch(color=RENK_ORTA, label="Hata %15–30"),
    mpatches.Patch(color=RENK_KOTU, label="Hata > %30"),
    Line2D([0], [0], color=RENK_KOYU, lw=1.5, ls="--", label="Mukemmel tahmin"),
]
ax_scatter.legend(handles=legend_handles, fontsize=9, loc="upper left")

ax_scatter.text(
    0.97, 0.05,
    f"R² = {global_r2:.3f}\nMAPE = %{global_mape:.1f}\nMedAPE = %{global_medape:.1f}",
    transform=ax_scatter.transAxes,
    ha="right", va="bottom", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8)
)

# ── B: Özet kart ─────────────────────────────────────────────────────────────
ax_kart = fig.add_subplot(gs[0, 2])
ax_kart.set_xlim(0, 1)
ax_kart.set_ylim(0, 1)
ax_kart.axis("off")
ax_kart.set_facecolor("white")
ax_kart.add_patch(mpatches.FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02",
    facecolor="white", edgecolor="#DEE2E6", linewidth=1.5
))

metriks = [
    ("R²  (in-sample)",   f"{global_r2:.3f}",   RENK_MAVI),
    ("R²  (CV 5-fold)",   f"{cfg.get('cv_r2', 'N/A')}",   RENK_MOR),
    ("MAPE  (in-sample)", f"%{global_mape:.1f}",  RENK_IYI),
    ("MAPE  (CV)",        f"%{cfg.get('cv_mape_pct', 'N/A')}",  RENK_ORTA),
    ("MedAPE",            f"%{global_medape:.1f}", RENK_IYI),
    ("Q10-Q90 isabet",    f"%{global_hit:.1f}",   RENK_MAVI),
    ("Egitim verisi",     f"{len(df):,} ev",       RENK_KOYU),
]

ax_kart.text(0.5, 0.99, "B  —  Model Ozeti", ha="center", va="top",
             fontsize=12, fontweight="bold", color=RENK_KOYU)
ax_kart.text(0.5, 0.91, "CV metrikleri holdout setten gelir;\nin-sample degerlerden daha gercekci.",
             ha="center", va="top", fontsize=7.5, color="#777777", style="italic")

for i, (etiket, deger, renk) in enumerate(metriks):
    y_pos = 0.82 - i * 0.115
    ax_kart.text(0.1,  y_pos, etiket, ha="left",  va="center",
                 fontsize=9.5, color="#555555")
    ax_kart.text(0.92, y_pos, deger,  ha="right", va="center",
                 fontsize=11, fontweight="bold", color=renk)
    ax_kart.axhline(y=y_pos - 0.045, xmin=0.06, xmax=0.94,
                    color="#DEE2E6", lw=0.7)

# ── C: Hata Dağılımı histogram ────────────────────────────────────────────────
ax_hist = fig.add_subplot(gs[1, 0])

bins  = np.linspace(0, 100, 51)
renk_hist = [
    RENK_IYI  if b < 15 else
    RENK_ORTA if b < 30 else
    RENK_KOTU
    for b in bins[:-1]
]
counts, _ = np.histogram(hata_pct, bins=bins)
for b_lo, b_hi, cnt, rk in zip(bins[:-1], bins[1:], counts, renk_hist):
    ax_hist.bar(b_lo, cnt, width=(b_hi - b_lo) * 0.95,
                color=rk, alpha=0.8, align="edge")

for p, ls in [(25, ":"), (50, "--"), (75, "-.")]:
    val = np.percentile(hata_pct, p)
    ax_hist.axvline(val, color=RENK_KOYU, lw=1.2, ls=ls,
                    label=f"P{p} = %{val:.0f}")

ax_hist.set_xlabel("Mutlak hata (%)", fontsize=10)
ax_hist.set_ylabel("Ev sayisi", fontsize=10)
ax_hist.set_title(
    "C  —  Hata Dagilimi\n"
    "Sol kayik = iyi.  P50 cizgisi medyan mutlak hatayi gosterir.",
    fontsize=11, fontweight="bold", loc="left"
)
ax_hist.set_xlim(0, 100)
ax_hist.legend(fontsize=8)

# ── D: Kümülatif hata eğrisi ──────────────────────────────────────────────────
ax_cum = fig.add_subplot(gs[1, 1])

esikler  = np.arange(0, 101, 1)
kumulatif = [(hata_pct <= e).mean() * 100 for e in esikler]

ax_cum.plot(esikler, kumulatif, color=RENK_MAVI, lw=2.5)
ax_cum.fill_between(esikler, kumulatif, alpha=0.15, color=RENK_MAVI)

for esik, renk in [(15, RENK_IYI), (30, RENK_ORTA), (50, RENK_KOTU)]:
    oran = (hata_pct <= esik).mean() * 100
    ax_cum.axvline(esik, color=renk, lw=1.2, ls="--")
    ax_cum.axhline(oran, color=renk, lw=1.2, ls="--")
    ax_cum.scatter([esik], [oran], color=renk, zorder=5, s=60)
    ax_cum.text(esik + 1, oran - 4, f"%{esik} hatada\n%{oran:.0f} ev",
                fontsize=8, color=renk, fontweight="bold")

ax_cum.set_xlabel("Maksimum hata esigi (%)", fontsize=10)
ax_cum.set_ylabel("Bu esik altindaki ev orani (%)", fontsize=10)
ax_cum.set_title(
    "D  —  Kumulatif Hata Egrisi\n"
    "X% hatada kac evin icinde oldugunu gosterir.  Egri yukari = model daha iyi.",
    fontsize=11, fontweight="bold", loc="left"
)
ax_cum.set_xlim(0, 100)
ax_cum.set_ylim(0, 102)

# ── E: Fiyat aralığı görselleştirme (örnek evler) ────────────────────────────
ax_aralik = fig.add_subplot(gs[1, 2])

# Her ilçeden en temsili 1 ev (medyana en yakın hata)
ornek_idx = []
for ilce, grp in df.groupby("district"):
    if len(grp) < 20:
        continue
    med_hata = grp["hata_pct"].median()
    en_yakin = (grp["hata_pct"] - med_hata).abs().idxmin()
    ornek_idx.append(en_yakin)

ornek = df.loc[ornek_idx].nlargest(12, "price").reset_index(drop=True)
ornek = ornek.sort_values("price").reset_index(drop=True)

y_pos    = np.arange(len(ornek))
gercek   = ornek["price"].values / 1e6
tahmin_v = ornek["tahmin"].values / 1e6
lo_v     = ornek["y_lo"].values / 1e6
hi_v     = ornek["y_hi"].values / 1e6

for i, (g, lo, hi, tah, ok) in enumerate(
        zip(gercek, lo_v, hi_v, tahmin_v, ornek["aralik_ici"])):
    renk_bar = RENK_IYI if ok else RENK_KOTU
    ax_aralik.barh(i, hi - lo, left=lo, height=0.5,
                   color=renk_bar, alpha=0.35, zorder=2)
    ax_aralik.scatter(tah, i, color=RENK_MAVI, s=40, zorder=4, marker="D")
    ax_aralik.scatter(g,   i, color=RENK_KOYU, s=50, zorder=5, marker="|",
                      linewidths=2.5)

isim_list = [
    f"{row['district'][:10]}  {int(row['gross_sqm'])}m²"
    for _, row in ornek.iterrows()
]
ax_aralik.set_yticks(y_pos)
ax_aralik.set_yticklabels(isim_list, fontsize=7.5)
ax_aralik.set_xlabel("Fiyat (M TL)", fontsize=10)
ax_aralik.set_title(
    "E  —  Fiyat Bandi — Ornek Evler\n"
    "Bant = Q10-Q90 aralik.  Karo(x) = Q50 tahmin.  Cubuk(|) = gercek fiyat.",
    fontsize=11, fontweight="bold", loc="left"
)

legend_h = [
    mpatches.Patch(color=RENK_IYI, alpha=0.5, label="Q10-Q90 aralik (EVET)"),
    mpatches.Patch(color=RENK_KOTU, alpha=0.5, label="Q10-Q90 aralik (HAYIR)"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor=RENK_MAVI,
           markersize=7, label="Tahmin (Q50)"),
    Line2D([0],[0], marker="|", color=RENK_KOYU, markersize=10,
           markeredgewidth=2.5, lw=0, label="Gercek fiyat"),
]
ax_aralik.legend(handles=legend_h, fontsize=7.5, loc="lower right")

# ── F: İlçe bazlı MAPE çubuğu ─────────────────────────────────────────────────
ax_ilce = fig.add_subplot(gs[2, :2])

n_goster  = min(20, len(ilce_stats))
ilce_plot = ilce_stats.head(n_goster).copy()

renkler_ilce = [
    RENK_IYI  if m < 15 else
    RENK_ORTA if m < 25 else
    RENK_KOTU
    for m in ilce_plot["mape"]
]

bars = ax_ilce.barh(
    ilce_plot["district"], ilce_plot["mape"],
    color=renkler_ilce, alpha=0.85, height=0.65
)

for bar, val, n_ev in zip(bars, ilce_plot["mape"], ilce_plot["n"]):
    ax_ilce.text(
        bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
        f"%{val:.1f}  (n={n_ev})",
        va="center", ha="left", fontsize=8, color="#555555"
    )

ax_ilce.axvline(15, color=RENK_IYI,  lw=1.2, ls="--", alpha=0.7, label="<15% iyi")
ax_ilce.axvline(25, color=RENK_ORTA, lw=1.2, ls="--", alpha=0.7, label="<25% orta")
ax_ilce.axvline(global_mape, color=RENK_MOR, lw=1.8, ls="-",
                label=f"Genel ort. %{global_mape:.1f}")

ax_ilce.set_xlabel("Ortalama Mutlak Hata Yuzdesi — MAPE (%)", fontsize=11)
ax_ilce.set_title(
    f"F  —  Ilce Bazli Model Performansi  (min 30 ilan, ilk {n_goster} ilce)\n"
    "Kisa cubuk = model o ilcede daha dogru.  Mor dikey cizgi = genel ortalama MAPE.",
    fontsize=12, fontweight="bold", loc="left"
)
ax_ilce.legend(fontsize=9, loc="lower right")
ax_ilce.set_xlim(0, ilce_plot["mape"].max() * 1.25)
ax_ilce.invert_yaxis()

# ── G: Feature importance ─────────────────────────────────────────────────────
ax_feat = fig.add_subplot(gs[2, 2])

m50 = models[50]
feat_names = cfg["model_feats"]
importances = pd.Series(m50.feature_importances_, index=feat_names)
top_feats   = importances.nlargest(12).sort_values()

ISIM_TR = {
    "gross_sqm"              : "Brut m2",
    "net_sqm"                : "Net m2",
    "total_rooms"            : "Oda sayisi",
    "floor"                  : "Kat",
    "total_floors"           : "Bina kat sayisi",
    "building_age"           : "Bina yasi",
    "bathroom_count"         : "Banyo sayisi",
    "sqm_per_room"           : "m2 / oda",
    "floor_ratio"            : "Kat orani",
    "yuksek_bina"            : "Yuksek bina?",
    "ust_kat"                : "Ust kat?",
    "alt_kat"                : "Alt kat?",
    "is_in_complex"          : "Site icinde?",
    "maintenance_fee"        : "Aidat",
    "district_te"            : "Ilce (hedef enc.)",
    "neighborhood_te"        : "Mahalle (hedef enc.)",
    "heating_type_enc"       : "Isinma tipi",
    "furnished_enc"          : "Esya durumu",
    "usage_status_enc"       : "Kullanim durumu",
    "building_type_enc"      : "Yapi tipi",
    "building_condition_enc" : "Yapi durumu",
    "floor_category_enc"     : "Kat kategorisi",
    "orientation_enc"        : "Cephe",
}

guzel_isimler = [ISIM_TR.get(f, f) for f in top_feats.index]

renk_feat = [
    RENK_MOR  if "te" in f else
    RENK_MAVI if f in ["gross_sqm", "net_sqm", "sqm_per_room", "maintenance_fee"] else
    RENK_ORTA
    for f in top_feats.index
]

ax_feat.barh(guzel_isimler, top_feats.values, color=renk_feat, alpha=0.85)
ax_feat.set_xlabel("Ozellik Onemi (XGBoost)", fontsize=10)
ax_feat.set_title(
    "G  —  En Etkili 12 Ozellik\n"
    "Mor = konum (ilce/mahalle)  |  Mavi = alan/fiyat  |  Turuncu = diger.",
    fontsize=11, fontweight="bold", loc="left"
)

legend_f = [
    mpatches.Patch(color=RENK_MOR,  alpha=0.85, label="Lokasyon (hedef enc.)"),
    mpatches.Patch(color=RENK_MAVI, alpha=0.85, label="Alan / fiyat"),
    mpatches.Patch(color=RENK_ORTA, alpha=0.85, label="Diger"),
]
ax_feat.legend(handles=legend_f, fontsize=8)

# ── Kaydet ───────────────────────────────────────────────────────────────────
plt.savefig("model_performans_raporu.png", dpi=150, bbox_inches="tight",
            facecolor="#F8F9FA")
plt.close()
print("\nGorsel kaydedildi: model_performans_raporu.png")
