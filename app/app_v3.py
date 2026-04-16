import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from fiyat_tahmin_pipeline import ARTIFACT_DIR

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI_YOLU = os.path.join(ROOT_DIR, "istanbul_apartment_prices_2026.csv")


# ══════════════════════════════════════════════════════════════════════════════
# KURUMSAL CSS
# ══════════════════════════════════════════════════════════════════════════════

CORPORATE_CSS = """
<style>
/* ── Genel arka plan & font ─────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background-color: #F0F4F8; }

/* ── Üst başlık ─────────────────────────────────────────────────────── */
.corp-header {
    background: linear-gradient(135deg, #0F2A4A 0%, #1A56DB 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    color: white;
}
.corp-header h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.3px; }
.corp-header p  { margin: 6px 0 0; font-size: 14px; opacity: 0.8; }

/* ── Sekmeler ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 14px;
    color: #64748B;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1A56DB !important;
    color: white !important;
}

/* ── Metrik kartlar ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
[data-testid="stMetricLabel"] { font-size: 12px; color: #64748B; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 22px; font-weight: 700; color: #0F2A4A; }

/* ── Butonlar ────────────────────────────────────────────────────────── */
.stButton > button {
    background: #1A56DB;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 20px;
    transition: background .2s;
}
.stButton > button:hover { background: #1348C0; color: white; }

/* ── Form elemanları ─────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div > div > input {
    border-radius: 8px;
    border-color: #CBD5E1;
}

/* ── Veri tablosu ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}

/* ── Bölücü çizgi ────────────────────────────────────────────────────── */
hr { border-color: #E2E8F0; margin: 20px 0; }

/* ── Bilgi kutuları ──────────────────────────────────────────────────── */
.stSuccess, .stWarning, .stInfo, .stError { border-radius: 8px; }

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* ── Alt caption ─────────────────────────────────────────────────────── */
.stCaption { color: #94A3B8; font-size: 12px; }

/* ── Kart bileşeni ───────────────────────────────────────────────────── */
.corp-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
    margin-bottom: 8px;
}
.corp-card-title {
    font-size: 11px;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: .8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Fırsat skoru rozetleri ──────────────────────────────────────────── */
.badge-pos {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 22px;
}
.badge-neg {
    display: inline-block;
    background: #FEE2E2;
    color: #991B1B;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 22px;
}

/* ── Analiz çıktısı ──────────────────────────────────────────────────── */
.analiz-box {
    background: #FFFFFF;
    border-left: 4px solid #1A56DB;
    border-radius: 0 10px 10px 0;
    padding: 20px 24px;
    margin-top: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}

/* ── Tablo başlık çizgisi ─────────────────────────────────────────────── */
.section-title {
    font-size: 15px;
    font-weight: 700;
    color: #0F2A4A;
    border-bottom: 2px solid #1A56DB;
    padding-bottom: 6px;
    margin: 20px 0 14px;
    display: inline-block;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def fmt_tl(v):
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M ₺"
    return f"{v / 1_000:.0f}K ₺"


def md2html(text: str) -> str:
    """**bold** → <strong>bold</strong>, satır sonu → <br>"""
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace("\n", "<br>")
    return text


def firsat_badge(pct):
    cls  = "badge-pos" if pct >= 0 else "badge-neg"
    sign = "+" if pct >= 0 else ""
    return f'<span class="{cls}">{sign}{pct:.1f}%</span>'


def corp_card(title, content_html):
    # Girintisiz tek satır — markdown parser'ın code-block algılamasını önler
    return (
        f'<div class="corp-card">'
        f'<div class="corp-card-title">{title}</div>'
        f'{content_html}'
        f'</div>'
    )


def fiyat_analizi_uret(oz, sonuc, ilce_ort, mahalle_ort):
    tahmin      = sonuc["tahmin"]
    ilce        = oz["district"]
    mahalle     = oz["neighborhood"]
    paragraflar = []

    ilce_fiyat    = ilce_ort.get(ilce)
    mahalle_fiyat = mahalle_ort.get(mahalle)
    bolge = []

    if mahalle_fiyat:
        oran = tahmin / mahalle_fiyat
        if oran > 1.25:
            bolge.append(f"{mahalle} mahallesi ortalamasının ({fmt_tl(mahalle_fiyat)}) **%{(oran-1)*100:.0f} üzerinde**")
        elif oran < 0.75:
            bolge.append(f"{mahalle} mahallesi ortalamasının ({fmt_tl(mahalle_fiyat)}) **%{(1-oran)*100:.0f} altında**")
        else:
            bolge.append(f"{mahalle} mahallesi ortalamasına ({fmt_tl(mahalle_fiyat)}) **yakın seviyede**")

    if ilce_fiyat:
        oran = tahmin / ilce_fiyat
        if oran > 1.25:
            bolge.append(f"{ilce} ilçesi ortalamasının ({fmt_tl(ilce_fiyat)}) **%{(oran-1)*100:.0f} üzerinde**")
        elif oran < 0.75:
            bolge.append(f"{ilce} ilçesi ortalamasının ({fmt_tl(ilce_fiyat)}) **%{(1-oran)*100:.0f} altında**")
        else:
            bolge.append(f"{ilce} ilçesi ortalamasına ({fmt_tl(ilce_fiyat)}) **yakın seviyede**")

    if bolge:
        paragraflar.append(
            f"**Bölge Karşılaştırması:** {ilce}, {mahalle} konumundaki bu dairenin "
            f"tahmini değeri **{sonuc['tahmin_fmt']}** olup " + " ve ".join(bolge) + " bir fiyat düzeyindedir."
        )
    else:
        paragraflar.append(
            f"**Fiyat Tahmini:** {ilce}, {mahalle} için tahmini değer "
            f"**{sonuc['tahmin_fmt']}** (aralık: {sonuc['aralik']})."
        )

    pozitif, negatif = [], []
    gross        = float(oz.get("gross_sqm") or 0)
    rooms        = float(oz.get("total_rooms") or 1)
    sqm_per_room = gross / max(rooms, 1)
    age          = float(oz.get("building_age") or 0)
    floor        = float(oz.get("floor") or 0)
    total_f      = float(oz.get("total_floors") or 1)

    if gross >= 180:   pozitif.append(f"çok geniş brüt alan ({gross:.0f} m²)")
    elif gross >= 130: pozitif.append(f"geniş brüt alan ({gross:.0f} m²)")
    elif gross < 65:   negatif.append(f"küçük brüt alan ({gross:.0f} m²)")

    if sqm_per_room >= 30:  pozitif.append(f"oda başına yüksek m² ({sqm_per_room:.0f} m²/oda)")
    elif sqm_per_room < 17: negatif.append(f"oda başına düşük m² ({sqm_per_room:.0f} m²/oda)")

    bc = float(oz.get("bathroom_count") or 1)
    if bc >= 2 and rooms >= 3: pozitif.append(f"{bc:.0f} banyo")
    elif bc == 1 and rooms >= 4: negatif.append("geniş daireye göre tek banyo")

    if age == 0:       pozitif.append("sıfır bina")
    elif age <= 5:     pozitif.append(f"yeni bina ({age:.0f} yaşında)")
    elif age <= 15:    pass
    elif age <= 25:    negatif.append(f"orta yaşlı bina ({age:.0f} yıl)")
    else:              negatif.append(f"eski bina ({age:.0f} yıl)")

    cond = str(oz.get("building_condition") or "").lower()
    if "new" in cond:                                    pozitif.append("yeni yapım")
    elif "construction" in cond or "under" in cond:     pozitif.append("inşaat halinde")

    btype = str(oz.get("building_type") or "").lower()
    if "reinforced" in btype or "concrete" in btype: pozitif.append("betonarme yapı")
    elif "masonry" in btype:                          negatif.append("yığma yapı")

    heat = str(oz.get("heating_type") or "").lower()
    if "combi" in heat:                             pozitif.append("kombi")
    elif "floor" in heat:                           pozitif.append("yerden ısıtma")
    elif "air" in heat or "conditioning" in heat:   negatif.append("klima ile ısınma")
    elif "stove" in heat:                           negatif.append("soba ısıtması")

    furn = str(oz.get("furnished") or "").lower()
    if furn and "unfurnished" not in furn and "semi" not in furn: pozitif.append("eşyalı")
    elif "semi" in furn:                                           pozitif.append("yarı eşyalı")

    if oz.get("is_in_complex"):
        aidat = float(oz.get("maintenance_fee") or 0)
        pozitif.append(f"site içi ({aidat:,.0f} ₺/ay aidat)" if aidat > 0 else "site içi")

    ust_kat = floor >= total_f - 1
    if ust_kat and total_f >= 5: pozitif.append(f"üst kat ({floor:.0f}. kat)")
    elif floor <= 0:             negatif.append("giriş/bodrum kat")
    elif floor == 1:             negatif.append("1. kat")

    orient = str(oz.get("orientation") or "").lower()
    if "south" in orient:                            pozitif.append("güney cephe")
    elif "north" in orient and "south" not in orient: negatif.append("kuzey cephe")

    usage = str(oz.get("usage_status") or "").lower()
    if "vacant" in usage or "empty" in usage: pozitif.append("boş daire (hemen teslim)")
    elif "tenant" in usage:                   negatif.append("kiracılı")

    if pozitif:
        paragraflar.append("**Fiyatı Artıran Özellikler:** " + ", ".join(pozitif) + ".")
    if negatif:
        paragraflar.append("**Fiyatı Düşüren Özellikler:** " + ", ".join(negatif) + ".")

    net = len(pozitif) - len(negatif)
    if net >= 4:
        ozet = f"Bu daire {ilce} ilçesinde güçlü bir profil sunmaktadır; olumlu özellikleri fiyatı desteklemektedir."
    elif net >= 1:
        ozet = f"Bu daire {ilce} ilçesi için dengeli bir profil sergilemekte; fiyat piyasayla uyumlu görünmektedir."
    elif not pozitif and not negatif:
        ozet = f"Özellikler ortalama değerlere yakın; fiyat {ilce} için tipik bir seviyededir."
    else:
        ozet = "Bazı zayıf özellikler fiyatı bölge ortalamasının altında tutmaktadır."

    paragraflar.append(f"**Genel Değerlendirme:** {ozet}")
    return "\n\n".join(paragraflar)


# ══════════════════════════════════════════════════════════════════════════════
# VERİ & MODEL YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    cfg_path = os.path.join(ARTIFACT_DIR, "feature_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    target_enc = joblib.load(os.path.join(ARTIFACT_DIR, "target_encoder.pkl"))
    ord_enc    = joblib.load(os.path.join(ARTIFACT_DIR, "ord_encoder.pkl"))
    models = {}
    for q in cfg["quantile_levels"]:
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(ARTIFACT_DIR, f"xgb_q{q:02d}.json"))
        models[q] = m
    return cfg, target_enc, ord_enc, models


@st.cache_data
def load_data():
    df = pd.read_csv(VERI_YOLU, low_memory=False)
    df = df[df["price"].notna() & df["district"].notna()].copy()
    for col in ["gross_sqm", "net_sqm", "total_rooms", "floor", "total_floors",
                "building_age", "bathroom_count", "is_in_complex", "maintenance_fee"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def yukle_ilce_mahalle(df):
    mapping = {}
    for dist, grp in df.groupby("district"):
        mapping[str(dist)] = sorted([str(m) for m in grp["neighborhood"].dropna().unique()])
    return mapping


@st.cache_resource
def yukle_bolge_fiyatlari(_te):
    ilce_ort, mahalle_ort = {}, {}
    for i, col in enumerate(["district", "neighborhood"]):
        cats = _te.categories_[i]
        encs = _te.encodings_[i]
        d = {str(k): float(np.expm1(v)) for k, v in zip(cats, encs)}
        if col == "district":  ilce_ort    = d
        else:                  mahalle_ort = d
    return ilce_ort, mahalle_ort


def batch_tahmin_q50(df_rows, cfg, target_enc, ord_enc, model_q50):
    imp = cfg["imputer_vals"]
    df  = df_rows.copy()
    for col in cfg["features_num"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce").fillna(imp.get(col, 0))
    for col in cfg["features_cat"]:
        df[col] = df[col].fillna(imp.get(col, "Unknown")).astype(str) if col in df.columns \
                  else str(imp.get(col, "Unknown"))
    df["is_in_complex"]   = pd.to_numeric(df.get("is_in_complex",   0), errors="coerce").fillna(0)
    df["maintenance_fee"] = pd.to_numeric(df.get("maintenance_fee", 0), errors="coerce").fillna(0)
    df["sqm_per_room"] = df["gross_sqm"] / df["total_rooms"].clip(lower=1)
    df["floor_ratio"]  = df["floor"] / df["total_floors"].clip(lower=1)
    df["yuksek_bina"]  = (df["total_floors"] >= 10).astype(int)
    df["ust_kat"]      = (df["floor"] >= df["total_floors"] - 1).astype(int)
    df["alt_kat"]      = (df["floor"] <= 0).astype(int)
    te_present  = [c for c in cfg["features_target_enc"] if c in df.columns]
    ord_present = [c for c in cfg["features_ord_enc"]    if c in df.columns]
    if te_present:
        te_vals = target_enc.transform(df[te_present].astype(str))
        for i, col in enumerate(te_present): df[f"{col}_te"] = te_vals[:, i]
    if ord_present:
        ord_vals = ord_enc.transform(df[ord_present].astype(str))
        for i, col in enumerate(ord_present): df[f"{col}_enc"] = ord_vals[:, i]
    for f in cfg["model_feats"]:
        if f not in df.columns: df[f] = 0.0
    X = df[cfg["model_feats"]].values.astype(np.float64)
    return np.expm1(model_q50.predict(X))


# ══════════════════════════════════════════════════════════════════════════════
# SAYFA BAŞLANGICI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="İstEmlak-AI | Kurumsal Emlak Paneli",
    page_icon="🏙️",
    layout="wide",
)

st.markdown(CORPORATE_CSS, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="corp-header">
    <h1>🏙️ İstEmlak-AI &nbsp;|&nbsp; Kurumsal Emlak Analiz Paneli</h1>
    <p>İstanbul konut piyasası için makine öğrenmesi destekli fiyat tahmini ve fırsat analizi</p>
</div>
""", unsafe_allow_html=True)

# Artifact kontrolü
cfg_path = os.path.join(ARTIFACT_DIR, "feature_config.json")
if not os.path.exists(cfg_path):
    st.error("Model artifacts bulunamadı. `python fiyat_tahmin_pipeline.py --egit` çalıştırın.")
    st.stop()

cfg, target_enc, ord_enc, models = load_artifacts()
cats                          = cfg.get("cat_categories", {})
imp                           = cfg.get("imputer_vals", {})
df_raw                        = load_data()
ilce_mahalle_map              = yukle_ilce_mahalle(df_raw)
ilce_ort, mahalle_ort         = yukle_bolge_fiyatlari(target_enc)

# ══════════════════════════════════════════════════════════════════════════════
# SEKMELER
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🔍  Fırsat Dedektörü", "💰  Fiyat Tahmini"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FIRSAT DEDEKTÖRÜ
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p style="color:#64748B; font-size:13px; margin-bottom:20px;">Gerçek ilanların piyasa değerini XGBoost Q50 modeli ile karşılaştırır; potansiyel fırsatları tespit eder.</p>', unsafe_allow_html=True)

    # Filtre satırı
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        ilce_t1 = st.selectbox("📍 İlçe", sorted(df_raw["district"].dropna().unique()), key="t1_ilce")
    with f2:
        mahalle_t1 = st.selectbox("🏘️ Mahalle", ilce_mahalle_map.get(ilce_t1, []), key="t1_mah")
    with f3:
        min_firsat = st.slider("Minimum Fırsat Skoru (%)", -30, 50, 5,
                               help="Pozitif değer: ilan fiyatı piyasa değerinin altında")

    filtreli = df_raw[
        (df_raw["district"]     == ilce_t1) &
        (df_raw["neighborhood"] == mahalle_t1)
    ].copy()

    if filtreli.empty:
        st.warning("Bu bölgede ilan bulunamadı.")
    else:
        with st.spinner(f"{len(filtreli)} ilan analiz ediliyor..."):
            filtreli["Tahmin"] = batch_tahmin_q50(filtreli, cfg, target_enc, ord_enc, models[50])

        filtreli["Fırsat %"] = (
            (filtreli["Tahmin"] - filtreli["price"]) / filtreli["Tahmin"] * 100
        ).round(1)
        filtreli = filtreli[filtreli["Fırsat %"] >= min_firsat].sort_values("Fırsat %", ascending=False)

        if filtreli.empty:
            st.warning(f"%{min_firsat} üzerinde fırsat skoru olan ilan bulunamadı.")
        else:
            # ── Özet metrikler ─────────────────────────────────────────────
            st.markdown('<div class="section-title">Bölge Özeti</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam İlan",         f"{len(filtreli):,}")
            m2.metric("En Yüksek Fırsat",    f"%{filtreli['Fırsat %'].max():.1f}")
            m3.metric("Ort. İlan Fiyatı",    fmt_tl(filtreli["price"].mean()))
            m4.metric("Ort. Piyasa Tahmini", fmt_tl(filtreli["Tahmin"].mean()))

            # ── İlan tablosu ───────────────────────────────────────────────
            st.markdown('<div class="section-title">İlan Listesi</div>', unsafe_allow_html=True)

            tablo = filtreli[[
                "listing_id", "price", "Tahmin", "Fırsat %",
                "gross_sqm", "total_rooms", "building_age"
            ]].copy()
            tablo.columns = ["İlan ID", "İlan Fiyatı (₺)", "Piyasa Tahmini (₺)",
                             "Fırsat %", "Brüt m²", "Oda", "Bina Yaşı"]
            tablo["İlan Fiyatı (₺)"]    = tablo["İlan Fiyatı (₺)"].map("{:,.0f}".format)
            tablo["Piyasa Tahmini (₺)"] = tablo["Piyasa Tahmini (₺)"].map("{:,.0f}".format)

            st.dataframe(tablo.reset_index(drop=True), use_container_width=True, height=260)

            # ── İlan detayı ────────────────────────────────────────────────
            st.markdown('<div class="section-title">İlan Detayı</div>', unsafe_allow_html=True)

            secim_listesi = filtreli.apply(
                lambda x: f"ID: {x['listing_id']}  |  {x['price']:,.0f} ₺  |  Fırsat: %{x['Fırsat %']}",
                axis=1
            ).tolist()
            secilen = st.selectbox("Detay görüntülemek istediğiniz ilanı seçin:", secim_listesi, key="t1_sec")

            if secilen:
                sel_id = secilen.split("|")[0].replace("ID:", "").strip()
                ev     = filtreli[filtreli["listing_id"] == sel_id].iloc[0]
                firsat = ev["Fırsat %"]

                d1, d2, d3 = st.columns([1.3, 1, 1])

                with d1:
                    durum_text = "📈 İlan fiyatı piyasa değerinin altında — potansiyel fırsat" \
                                 if firsat > 0 else "📉 İlan fiyatı piyasa değerinin üzerinde"
                    konum   = f"{ev['district']} / {ev['neighborhood']}"
                    icerik  = (
                        f'{firsat_badge(firsat)}'
                        f'<p style="margin:10px 0 4px;color:#374151;font-weight:600;font-size:14px;">{konum}</p>'
                        f'<p style="margin:0;font-size:12px;color:#6B7280;">{durum_text}</p>'
                    )
                    st.markdown(corp_card("ML FIRSAT ANALİZİ", icerik), unsafe_allow_html=True)

                with d2:
                    st.metric("İlan Fiyatı",          f"{ev['price']:,.0f} ₺")
                    st.metric("Piyasa Tahmini (Q50)",  fmt_tl(ev["Tahmin"]))

                with d3:
                    st.metric("Brüt m²",  f"{ev['gross_sqm']:.0f} m²"  if pd.notna(ev.get("gross_sqm"))   else "—")
                    st.metric("Kat",      f"{ev['floor']:.0f}/{ev['total_floors']:.0f}" if pd.notna(ev.get("floor")) else "—")

                if st.button("📊 Bu İlanın Fiyat Analizini Göster", use_container_width=True, key="t1_analiz_btn"):
                    sonuc_dict = {
                        "tahmin"        : ev["Tahmin"],
                        "tahmin_fmt"    : fmt_tl(ev["Tahmin"]),
                        "alt_sinir_fmt" : fmt_tl(ev["Tahmin"] * 0.85),
                        "ust_sinir_fmt" : fmt_tl(ev["Tahmin"] * 1.15),
                        "aralik"        : f"{fmt_tl(ev['Tahmin'] * 0.85)} — {fmt_tl(ev['Tahmin'] * 1.15)}",
                    }
                    analiz = fiyat_analizi_uret(ev.to_dict(), sonuc_dict, ilce_ort, mahalle_ort)
                    st.markdown(f'<div class="analiz-box">{md2html(analiz)}</div>',
                                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FİYAT TAHMİNİ
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p style="color:#64748B; font-size:13px; margin-bottom:20px;">Daire özelliklerini girerek XGBoost Quantile modeli ile %10 – %50 – %90 fiyat aralığı tahmin edin.</p>', unsafe_allow_html=True)

    if "sonuc" not in st.session_state:    st.session_state.sonuc = None
    if "ozellikler" not in st.session_state: st.session_state.ozellikler = None

    # İlçe (form dışı — mahalle filtrelemesi için)
    st.markdown('<div class="section-title">Konum</div>', unsafe_allow_html=True)
    ilce_col, _ = st.columns(2)
    with ilce_col:
        district_t2 = st.selectbox("İlçe", cats.get("district", []), index=0, key="t2_ilce")

    mahalle_sec_t2 = ilce_mahalle_map.get(district_t2, cats.get("neighborhood", []))

    st.markdown('<div class="section-title">Fiziksel Özellikler</div>', unsafe_allow_html=True)

    with st.form("tahmin_form_v3"):
        col1, col2 = st.columns(2)

        with col1:
            neighborhood_t2 = st.selectbox("Mahalle",            options=mahalle_sec_t2, index=0)
            gross_sqm       = st.number_input("Brüt m²",         min_value=20,  max_value=1000, value=int(imp.get("gross_sqm", 120)))
            net_sqm         = st.number_input("Net m²",          min_value=15,  max_value=900,  value=int(imp.get("net_sqm", 95)))
            total_rooms     = st.number_input("Oda Sayısı (3+1 → 4)", min_value=1, max_value=20, value=int(imp.get("total_rooms", 4)))
            bathroom_count  = st.number_input("Banyo Sayısı",    min_value=1,   max_value=10,   value=int(imp.get("bathroom_count", 1)))

        with col2:
            floor           = st.number_input("Bulunduğu Kat",   min_value=0,   max_value=100,  value=int(imp.get("floor", 3)))
            total_floors    = st.number_input("Toplam Kat",       min_value=1,   max_value=100,  value=int(imp.get("total_floors", 8)))
            building_age    = st.number_input("Bina Yaşı (yıl)", min_value=0,   max_value=100,  value=int(imp.get("building_age", 10)))
            is_in_complex   = st.radio("Site İçinde mi?", options=[("Evet", 1), ("Hayır", 0)],
                                       format_func=lambda x: x[0], index=0)
            maintenance_fee = st.number_input("Aidat (₺/ay)",    min_value=0,   max_value=150_000, value=int(imp.get("maintenance_fee", 0)))

        st.markdown('<div class="section-title" style="margin-top:12px;">Niteliksel Özellikler</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        def _sel(label, key):
            opts    = cats.get(key, [])
            default = imp.get(key, opts[0] if opts else "")
            idx     = opts.index(default) if default in opts else 0
            return st.selectbox(label, options=opts, index=idx)

        with col3:
            heating_type       = _sel("Isınma Tipi",     "heating_type")
            furnished          = _sel("Eşya Durumu",     "furnished")
            usage_status       = _sel("Kullanım Durumu", "usage_status")
            orientation        = _sel("Cephe",           "orientation")

        with col4:
            building_type      = _sel("Yapı Tipi",       "building_type")
            building_condition = _sel("Yapı Durumu",     "building_condition")
            floor_category     = _sel("Kat Kategorisi",  "floor_category")

        submitted_t2 = st.form_submit_button("💰  Fiyat Tahmin Et", use_container_width=True)

    if submitted_t2:
        from fiyat_tahmin_pipeline import tahmin_et
        ozellikler_t2 = {
            "district": district_t2,        "neighborhood": neighborhood_t2,
            "gross_sqm": gross_sqm,         "net_sqm": net_sqm,
            "total_rooms": total_rooms,      "floor": floor,
            "total_floors": total_floors,    "building_age": building_age,
            "bathroom_count": bathroom_count,
            "is_in_complex": is_in_complex[1],
            "maintenance_fee": maintenance_fee,
            "heating_type": heating_type,    "furnished": furnished,
            "usage_status": usage_status,    "building_type": building_type,
            "building_condition": building_condition,
            "floor_category": floor_category, "orientation": orientation,
        }
        with st.spinner("Model hesaplıyor..."):
            try:
                st.session_state.sonuc      = tahmin_et(ozellikler_t2)
                st.session_state.ozellikler = ozellikler_t2
            except Exception as e:
                st.error(f"Tahmin hatası: {e}")
                st.stop()

    if st.session_state.sonuc:
        sonuc      = st.session_state.sonuc
        ozellikler = st.session_state.ozellikler

        st.markdown('<div class="section-title">Tahmin Sonucu</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Alt Sınır (%10)", sonuc["alt_sinir_fmt"])
        c2.metric("Nokta Tahmin (%50)", sonuc["tahmin_fmt"])
        c3.metric("Üst Sınır (%90)", sonuc["ust_sinir_fmt"])

        st.markdown(f"""
        <div class="corp-card" style="margin-top:12px;">
            <div class="corp-card-title">TAHMİN ARALIGI</div>
            <span style="font-size:20px; font-weight:700; color:#1A56DB;">{sonuc['aralik']}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Model Performans Bilgisi"):
            e1, e2 = st.columns(2)
            e1.metric("CV-R²",    cfg.get("cv_r2", "—"))
            e2.metric("CV-MAPE",  f"%{cfg.get('cv_mape_pct', '—')}")

        if st.button("📊  Fiyat Analizi Göster", use_container_width=True, key="t2_analiz_btn"):
            analiz = fiyat_analizi_uret(ozellikler, sonuc, ilce_ort, mahalle_ort)
            st.markdown(f'<div class="analiz-box">{md2html(analiz)}</div>',
                        unsafe_allow_html=True)
