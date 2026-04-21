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

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI_YOLU      = os.path.join(ROOT_DIR, "istanbul_apartment_prices_2026.csv")
ULTIMATE_YOLU  = os.path.join(ROOT_DIR, "istanbul_emlak_ULTIMATE_2026.csv")


# ══════════════════════════════════════════════════════════════════════════════
# KURUMSAL CSS
# ══════════════════════════════════════════════════════════════════════════════

CORPORATE_CSS = """
<style>
/* ── Genel arka plan & font ─────────────────────────────────────────── */
html { font-size: 22px; }
body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; color: #EEEEEE; font-size: 22px; }
.stApp { background-color: #1E1E1E; }
.main .block-container { background-color: #1E1E1E; }

/* ── Global yazı boyutu & renk ───────────────────────────────────────── */
* { font-size: 22px !important; line-height: 1.8 !important; color: #EEEEEE !important; }
h1, h2, h3 { font-size: 32px !important; }
h4, h5, h6 { font-size: 26px !important; }
[data-testid="stMetricLabel"] { font-size: 16px !important; }
[data-testid="stMetricValue"] { font-size: 36px !important; font-weight: 800 !important; }
.stTabs [data-baseweb="tab"] { font-size: 20px !important; padding: 14px 30px !important; }
.section-title { font-size: 24px !important; }
.corp-card-title { font-size: 15px !important; }
.stButton > button { font-size: 22px !important; }
[data-testid="stDataFrame"] * { font-size: 18px !important; }

/* ── Üst başlık ─────────────────────────────────────────────────────── */
.corp-header {
    background: linear-gradient(135deg, #3A3A3A 0%, #5A5A5A 100%);
    padding: 32px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 4px 20px rgba(192,57,43,.35);
}
.corp-header h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; }
.corp-header p  { margin: 8px 0 0; font-size: 14px; opacity: 0.75; }

/* ── Sekmeler ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #2A2A2A;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #3A3A3A;
    box-shadow: 0 1px 4px rgba(0,0,0,.3);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 9px 22px;
    font-weight: 600;
    font-size: 14px;
    color: #AAAAAA;
    border: none !important;
    transition: background .15s;
}
.stTabs [aria-selected="true"] {
    background: #C0392B !important;
    color: white !important;
}

/* ── Metrik kartlar ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #2A2A2A;
    border: 1px solid #3A3A3A;
    border-top: 3px solid #C0392B;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,.3);
}
[data-testid="stMetricLabel"] { font-size: 12px; color: #AAAAAA; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; }
[data-testid="stMetricValue"] { font-size: 24px; font-weight: 800; color: #FFFFFF; }

/* ── Butonlar ────────────────────────────────────────────────────────── */
.stButton { display: flex; justify-content: center; }
.stButton > button {
    background: #C0392B;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 18px;
    padding: 12px 28px;
    width: fit-content !important;
    min-width: 0 !important;
    max-width: 360px !important;
    transition: background .2s, transform .1s;
    letter-spacing: .2px;
}
.stButton > button:hover { background: #A93226; color: white; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── Genel markdown yazı boyutu ──────────────────────────────────────── */
.stMarkdown p, .stMarkdown li { font-size: 15px; line-height: 1.75; }
[data-testid="stMarkdownContainer"] p { font-size: 15px; }

/* ── Selectbox & form elemanları ─────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] {
    background: #111111 !important;
    border-color: #3A3A3A !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div { color: #FFFFFF !important; }
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li,
[role="listbox"],
[role="option"] {
    background: #111111 !important;
    color: #FFFFFF !important;
}
[role="option"]:hover { background: #C0392B !important; }
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #111111 !important;
    color: #FFFFFF !important;
    border-color: #3A3A3A !important;
    border-radius: 10px !important;
}
/* ── Number input +/- butonları ──────────────────────────────────────── */
[data-testid="stNumberInput"] button {
    background: #C0392B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #A93226 !important;
}
[data-testid="stNumberInput"] button svg * {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}
/* ── Form submit butonu ──────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] > button,
button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"] {
    background: #C0392B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    width: fit-content !important;
    max-width: 360px !important;
    padding: 12px 32px !important;
    margin: 0 auto !important;
    display: block !important;
}
[data-testid="stFormSubmitButton"] button:hover { background: #A93226 !important; }

/* ── Veri tablosu ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #3A3A3A;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
}

/* ── Bölücü çizgi ────────────────────────────────────────────────────── */
hr { border-color: #3A3A3A; margin: 20px 0; }

/* ── Bilgi kutuları ──────────────────────────────────────────────────── */
.stSuccess, .stWarning, .stInfo, .stError { border-radius: 10px; }

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #111111;
    border: 1px solid #3A3A3A;
    border-radius: 12px;
    box-shadow: 0 1px 6px rgba(0,0,0,.2);
}
[data-testid="stExpander"] summary {
    background: #111111 !important;
    border-radius: 12px;
    padding: 14px 20px;
}
[data-testid="stExpander"] summary:hover {
    background: #1E1E1E !important;
}
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
}
[data-testid="stExpander"] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

/* ── Alt caption ─────────────────────────────────────────────────────── */
.stCaption { color: #777777; font-size: 12px; }

/* ── Kart bileşeni ───────────────────────────────────────────────────── */
.corp-card {
    background: #2A2A2A;
    border: 1px solid #3A3A3A;
    border-radius: 14px;
    padding: 22px 26px;
    box-shadow: 0 3px 14px rgba(0,0,0,.25);
    margin-bottom: 8px;
}
.corp-card-title {
    font-size: 11px;
    font-weight: 700;
    color: #C0392B;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Fırsat skoru rozetleri ──────────────────────────────────────────── */
.badge-pos {
    display: inline-block;
    background: #C0392B !important;
    color: #FFFFFF !important;
    border-radius: 24px;
    padding: 8px 22px;
    font-weight: 800 !important;
    font-size: 26px !important;
    border: none;
}
.badge-neg {
    display: inline-block;
    background: #2A2A2A !important;
    color: #AAAAAA !important;
    border-radius: 24px;
    padding: 8px 22px;
    font-weight: 800 !important;
    font-size: 26px !important;
    border: 2px solid #555555;
}

/* ── Analiz çıktısı ──────────────────────────────────────────────────── */
.analiz-box {
    background: #2A2A2A;
    border-left: 4px solid #C0392B;
    border-radius: 0 12px 12px 0;
    padding: 22px 26px;
    margin-top: 14px;
    color: #DDDDDD;
    box-shadow: 0 3px 12px rgba(0,0,0,.25);
}

/* ── Hikaye kutusu ───────────────────────────────────────────────────── */
.hikaye-box {
    background: #252525;
    border-left: 4px solid #888888;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-top: 8px;
    font-size: 13px;
    color: #CCCCCC;
    line-height: 1.8;
    box-shadow: 0 2px 6px rgba(0,0,0,.2);
}

/* ── Karne kutusu ────────────────────────────────────────────────────── */
.karne-box {
    background: #2C2020;
    border-left: 4px solid #C0392B;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-top: 8px;
    font-size: 13px;
    color: #DDDDDD;
    line-height: 1.8;
    white-space: pre-line;
    box-shadow: 0 2px 6px rgba(0,0,0,.2);
}

/* ── Bölüm başlığı ───────────────────────────────────────────────────── */
.section-title {
    font-size: 15px;
    font-weight: 800;
    color: #EEEEEE;
    border-bottom: 3px solid #C0392B;
    padding-bottom: 6px;
    margin: 22px 0 16px;
    display: inline-block;
    letter-spacing: -.2px;
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


def slugify(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("çğıöşü ", "cgiosu-"))
    text = re.sub(r'[^a-z0-9-]', '', text)
    return text


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


@st.cache_data
def load_ultimate_data():
    if not os.path.exists(ULTIMATE_YOLU):
        return pd.DataFrame()
    df = pd.read_csv(ULTIMATE_YOLU, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)
    for col in ["tahmini_kira", "mahalle_ort_m2_fiyat", "mahalle_favori_oda"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("listing_id")


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
import base64, io
from PIL import Image as _PILImage

_logo_path = os.path.join(ROOT_DIR, "yeni_header_transparent.png")
_logo_b64  = ""
if os.path.exists(_logo_path):
    _img = _PILImage.open(_logo_path)
    _img.thumbnail((300, 120), _PILImage.LANCZOS)
    _buf = io.BytesIO()
    _img.save(_buf, format="PNG", optimize=True)
    _logo_b64 = base64.b64encode(_buf.getvalue()).decode()

_logo_html = (
    f'<img src="data:image/png;base64,{_logo_b64}" style="height:72px; margin-right:24px;">'
    if _logo_b64 else ""
)

st.markdown(f"""
<div class="corp-header">
    <div style="display:flex; align-items:center;">
        {_logo_html}
        <div>
            <h1 style="margin:0;">İstEmlak-AI &nbsp;|&nbsp; Kurumsal Emlak Analiz Paneli</h1>
            <p style="margin:8px 0 0; font-size:14px; opacity:0.75;">İstanbul konut piyasası için makine öğrenmesi destekli fiyat tahmini ve fırsat analizi</p>
        </div>
    </div>
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
df_ultimate                   = load_ultimate_data()
ilce_mahalle_map              = yukle_ilce_mahalle(df_raw)
ilce_ort, mahalle_ort         = yukle_bolge_fiyatlari(target_enc)

# ══════════════════════════════════════════════════════════════════════════════
# SEKMELER
# ══════════════════════════════════════════════════════════════════════════════

tab1, = st.tabs(["🔍  Fırsat Dedektörü"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FIRSAT DEDEKTÖRÜ
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p style="color:#64748B; font-size:13px; margin-bottom:20px;">Gerçek ilanların piyasa değerini XGBoost Q50 modeli ile karşılaştırır; potansiyel fırsatları tespit eder.</p>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        ilce_t1 = st.selectbox("📍 İlçe", sorted(df_raw["district"].dropna().unique()), key="t1_ilce")
    with f2:
        mahalle_t1 = st.selectbox("🏘️ Mahalle", ilce_mahalle_map.get(ilce_t1, []), key="t1_mah")

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
        filtreli = filtreli.sort_values("Fırsat %", ascending=False)

        if filtreli.empty:
            st.warning("Bu bölgede ilan bulunamadı.")
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

            st.dataframe(
                tablo.reset_index(drop=True),
                use_container_width=True,
                height=260,
                hide_index=False,
            )

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
                    konum      = f"{ev['district']} / {ev['neighborhood']}"
                    ilan_url   = (
                        f"https://www.hepsiemlak.com/istanbul-"
                        f"{slugify(ev['district'])}-{slugify(ev['neighborhood'])}"
                        f"-satilik/daire/{ev['listing_id']}"
                    )
                    icerik  = (
                        f'{firsat_badge(firsat)}'
                        f'<p style="margin:10px 0 4px;color:#374151;font-weight:600;font-size:14px;">{konum}</p>'
                        f'<p style="margin:0;font-size:12px;color:#6B7280;">{durum_text}</p>'
                        f'<hr style="margin:12px 0;border-color:#E2E8F0;">'
                        f'<a href="{ilan_url}" target="_blank" '
                        f'style="color:#1A56DB;font-weight:600;font-size:13px;text-decoration:none;">'
                        f'🔗 İlanı Kaynağında Gör ↗</a>'
                    )
                    st.markdown(corp_card("ML FIRSAT ANALİZİ", icerik), unsafe_allow_html=True)

                with d2:
                    st.metric("İlan Fiyatı",          f"{ev['price']:,.0f} ₺")
                    st.metric("Piyasa Tahmini (Q50)",  fmt_tl(ev["Tahmin"]))

                with d3:
                    st.metric("Brüt m²",  f"{ev['gross_sqm']:.0f} m²"  if pd.notna(ev.get("gross_sqm"))   else "—")
                    st.metric("Kat",      f"{ev['floor']:.0f}/{ev['total_floors']:.0f}" if pd.notna(ev.get("floor")) else "—")

                # ── İlan Analizi (ULTIMATE zengin veriler) ────────────────
                ult_row = df_ultimate.loc[sel_id] if (not df_ultimate.empty and sel_id in df_ultimate.index) else None
                if ult_row is not None:
                    st.markdown('<div class="section-title">İlan Analizi</div>', unsafe_allow_html=True)

                    u1, u2 = st.columns(2)
                    tahmini_kira = ult_row.get("tahmini_kira", None)
                    amortisman   = ult_row.get("amortisman_yili", None)
                    if pd.notna(tahmini_kira):
                        u1.metric("Tahmini Kira/Ay", f"{tahmini_kira:,.0f} ₺")
                    if pd.notna(amortisman):
                        u2.metric("Amortisman Süresi", str(amortisman))

                    hikaye = ult_row.get("ilan_hikayesi", None)
                    if pd.notna(hikaye) and hikaye:
                        st.markdown("**İlan Hikayesi**")
                        st.markdown(f'<div class="hikaye-box">{hikaye}</div>', unsafe_allow_html=True)

                    karne = ult_row.get("mahalle_karnesi", None)
                    if pd.notna(karne) and karne:
                        st.markdown("**Bölge Karnesi**")
                        karne_temiz = str(karne).replace("📊 BÖLGE KARNESİ:", "").strip()
                        st.markdown(f'<div class="karne-box">{karne_temiz}</div>', unsafe_allow_html=True)

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


