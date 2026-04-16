import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from fiyat_tahmin_pipeline import tahmin_et, ARTIFACT_DIR

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI_YOLU = os.path.join(ROOT_DIR, "istanbul_apartment_prices_2026.csv")


# ── Kural tabanlı analiz fonksiyonu ───────────────────────────────────────────
def fiyat_analizi_uret(oz, sonuc, ilce_ort, mahalle_ort):
    def fmt(v):
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M TL"
        return f"{v / 1_000:.0f}K TL"

    tahmin      = sonuc["tahmin"]
    ilce        = oz["district"]
    mahalle     = oz["neighborhood"]
    paragraflar = []

    ilce_fiyat    = ilce_ort.get(ilce)
    mahalle_fiyat = mahalle_ort.get(mahalle)
    bolge_cumleler = []

    if mahalle_fiyat:
        oran = tahmin / mahalle_fiyat
        if oran > 1.25:
            bolge_cumleler.append(
                f"{mahalle} mahallesi ortalamasının ({fmt(mahalle_fiyat)}) "
                f"**%{(oran - 1) * 100:.0f} üzerinde**"
            )
        elif oran < 0.75:
            bolge_cumleler.append(
                f"{mahalle} mahallesi ortalamasının ({fmt(mahalle_fiyat)}) "
                f"**%{(1 - oran) * 100:.0f} altında**"
            )
        else:
            bolge_cumleler.append(
                f"{mahalle} mahallesi ortalamasına ({fmt(mahalle_fiyat)}) **yakın bir seviyede**"
            )

    if ilce_fiyat:
        oran = tahmin / ilce_fiyat
        if oran > 1.25:
            bolge_cumleler.append(
                f"{ilce} ilçesi ortalamasının ({fmt(ilce_fiyat)}) "
                f"**%{(oran - 1) * 100:.0f} üzerinde**"
            )
        elif oran < 0.75:
            bolge_cumleler.append(
                f"{ilce} ilçesi ortalamasının ({fmt(ilce_fiyat)}) "
                f"**%{(1 - oran) * 100:.0f} altında**"
            )
        else:
            bolge_cumleler.append(
                f"{ilce} ilçesi ortalamasına ({fmt(ilce_fiyat)}) **yakın bir seviyede**"
            )

    if bolge_cumleler:
        paragraflar.append(
            f"**Bölge Karşılaştırması:** {ilce}, {mahalle} konumundaki bu dairenin "
            f"tahmin edilen fiyatı **{sonuc['tahmin_fmt']}** olup "
            + " ve ".join(bolge_cumleler) + " bir fiyat düzeyindedir."
        )
    else:
        paragraflar.append(
            f"**Fiyat Tahmini:** {ilce}, {mahalle} konumundaki bu dairenin tahmini fiyatı "
            f"**{sonuc['tahmin_fmt']}** (aralık: {sonuc['aralik']})."
        )

    pozitif = []
    negatif = []

    gross        = oz["gross_sqm"]
    rooms        = oz["total_rooms"]
    sqm_per_room = gross / max(rooms, 1)
    age          = oz["building_age"]
    floor        = oz["floor"]
    total_f      = oz["total_floors"]
    ust_kat      = floor >= total_f - 1
    alt_kat      = floor <= 0

    if gross >= 180:
        pozitif.append(f"çok geniş brüt alan ({gross} m²)")
    elif gross >= 130:
        pozitif.append(f"geniş brüt alan ({gross} m²)")
    elif gross < 65:
        negatif.append(f"küçük brüt alan ({gross} m²)")

    if sqm_per_room >= 30:
        pozitif.append(f"oda başına yüksek m² ({sqm_per_room:.0f} m²/oda — ferah plan)")
    elif sqm_per_room < 17:
        negatif.append(f"oda başına düşük m² ({sqm_per_room:.0f} m²/oda — sıkışık plan)")

    if oz["bathroom_count"] >= 2 and rooms >= 3:
        pozitif.append(f"{oz['bathroom_count']} banyo (yüksek kullanım konforu)")
    elif oz["bathroom_count"] == 1 and rooms >= 4:
        negatif.append("geniş daireye göre tek banyo")

    if age == 0:
        pozitif.append("sıfır bina")
    elif age <= 5:
        pozitif.append(f"yeni bina ({age} yaşında)")
    elif age <= 15:
        pass
    elif age <= 25:
        negatif.append(f"orta yaşlı bina ({age} yıl)")
    else:
        negatif.append(f"eski bina ({age} yıl)")

    cond = oz["building_condition"].lower()
    if "new" in cond:
        pozitif.append("yeni yapım")
    elif "construction" in cond or "under" in cond:
        pozitif.append("inşaat halinde — yeni proje")

    btype = oz["building_type"].lower()
    if "reinforced" in btype or "concrete" in btype:
        pozitif.append("betonarme yapı")
    elif "masonry" in btype:
        negatif.append("yığma yapı (eski inşaat tekniği)")
    elif "wood" in btype or "ahşap" in btype:
        negatif.append("ahşap yapı")

    heat = oz["heating_type"].lower()
    if "combi" in heat:
        pozitif.append("kombi (bireysel ısınma kontrolü)")
    elif "floor" in heat:
        pozitif.append("yerden ısıtma (üst segment konfor)")
    elif "air" in heat or "conditioning" in heat:
        negatif.append("klima ile ısınma (merkezi sistem yok)")
    elif "stove" in heat or "soba" in heat:
        negatif.append("soba ısıtması")

    furn = oz["furnished"].lower()
    if "unfurnished" not in furn and "semi" not in furn:
        pozitif.append("eşyalı teslim")
    elif "semi" in furn:
        pozitif.append("yarı eşyalı")

    if oz["is_in_complex"]:
        aidat = oz["maintenance_fee"]
        if aidat > 5000:
            pozitif.append(f"site içi ({aidat:,} TL/ay — kapsamlı hizmet)")
        elif aidat > 0:
            pozitif.append(f"site içi ({aidat:,} TL/ay aidat)")
        else:
            pozitif.append("site içi konum")

    if ust_kat and total_f >= 5:
        pozitif.append(f"üst kat ({floor}. kat — manzara ve sessizlik avantajı)")
    elif alt_kat:
        negatif.append("giriş/bodrum kat (güvenlik ve nem riski)")
    elif floor == 1:
        negatif.append("1. kat (düşük kat tercih edilmez)")

    if total_f >= 12:
        pozitif.append(f"yüksek bina ({total_f} katlı — manzara potansiyeli)")

    orient = oz["orientation"].lower()
    if "south" in orient:
        pozitif.append("güney cephe (maksimum güneş ışığı)")
    elif "north" in orient:
        negatif.append("kuzey cephe (az güneş ışığı)")

    usage = oz["usage_status"].lower()
    if "vacant" in usage or "empty" in usage:
        pozitif.append("boş daire (hemen teslim)")
    elif "tenant" in usage:
        negatif.append("kiracılı (kiracı çıkarma süreci gerekebilir)")

    if pozitif:
        paragraflar.append(
            "**Fiyatı Artıran Özellikler:** "
            + ", ".join(pozitif)
            + " bu dairenin fiyatını destekleyen başlıca unsurlardır."
        )

    if negatif:
        paragraflar.append(
            "**Fiyatı Düşüren Özellikler:** "
            + ", ".join(negatif)
            + " ise fiyatı aşağı çeken etkenler arasında sayılabilir."
        )

    net = len(pozitif) - len(negatif)
    if net >= 4:
        ozet = (
            f"Bu daire, {ilce} ilçesindeki mevcut seçenekler arasında güçlü bir profil "
            f"sunmaktadır. Olumlu özelliklerinin ağırlığı, fiyatın bölge ortalamasına "
            f"yakın ya da üzerinde seyrelmesini açıklamaktadır."
        )
    elif net >= 1:
        ozet = (
            f"Bu daire, {ilce} ilçesi için dengeli bir profil sergilemektedir. "
            f"Güçlü ve zayıf yönleri birbirini dengelemekte; fiyat seviyesi piyasa "
            f"beklentileriyle uyumlu görünmektedir."
        )
    elif not pozitif and not negatif:
        ozet = (
            f"Girilen özellikler genel ortalama değerlere yakın olduğundan fiyat, "
            f"{ilce} ilçesi için tipik bir seviyede hesaplanmıştır."
        )
    else:
        ozet = (
            f"Bu dairenin bazı zayıf özellikleri, fiyatın bölge ortalamasının altında "
            f"kalmasına yol açmaktadır. Yatırım veya kullanım değerlendirmesinde bu "
            f"faktörlerin göz önünde bulundurulması önerilir."
        )

    paragraflar.append(f"**Genel Değerlendirme:** {ozet}")
    return "\n\n".join(paragraflar)


# ── Sayfa ayarları ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="İstanbul Daire Fiyat Tahmini",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 İstanbul Daire Fiyat Tahmini")
st.caption("XGBoost Quantile Regression • %10 – %50 – %90 fiyat aralığı")

# ── Model artifact yükle ───────────────────────────────────────────────────────
cfg_path = os.path.join(ARTIFACT_DIR, "feature_config.json")

if not os.path.exists(cfg_path):
    st.error(
        f"`{cfg_path}` bulunamadı. "
        "Önce `python fiyat_tahmin_pipeline.py --egit` komutuyla modeli eğitin."
    )
    st.stop()

with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)

cats = cfg.get("cat_categories", {})
imp  = cfg.get("imputer_vals", {})


# ── İlçe → mahalle eşlemesi ───────────────────────────────────────────────────
@st.cache_data
def yukle_ilce_mahalle():
    """CSV'den ilçe→mahalle eşlemesini yükler."""
    if os.path.exists(VERI_YOLU):
        df = pd.read_csv(VERI_YOLU, usecols=["district", "neighborhood"], low_memory=False)
        df = df.dropna(subset=["district", "neighborhood"])
        mapping = {}
        for dist, grp in df.groupby("district"):
            mapping[str(dist)] = sorted([str(m) for m in grp["neighborhood"].unique()])
        return mapping
    return {}


# ── TargetEncoder'dan bölge fiyatları ─────────────────────────────────────────
@st.cache_resource
def yukle_bolge_fiyatlari():
    try:
        te = joblib.load(os.path.join(ARTIFACT_DIR, "target_encoder.pkl"))
        ilce_ort    = {}
        mahalle_ort = {}
        for i, col in enumerate(["district", "neighborhood"]):
            kategoriler = te.categories_[i]
            kodlamalar  = te.encodings_[i]
            d = {str(k): float(np.expm1(v)) for k, v in zip(kategoriler, kodlamalar)}
            if col == "district":
                ilce_ort = d
            else:
                mahalle_ort = d
        return ilce_ort, mahalle_ort
    except Exception:
        return {}, {}


ilce_mahalle_map          = yukle_ilce_mahalle()
ilce_ort, mahalle_ort     = yukle_bolge_fiyatlari()

# ── Model metrikleri ───────────────────────────────────────────────────────────
with st.expander("Model Performansı", expanded=False):
    c1, c2 = st.columns(2)
    c1.metric("CV-R²", cfg.get("cv_r2", "—"))
    c2.metric("CV-MAPE", f"%{cfg.get('cv_mape_pct', '—')}")

st.divider()

# ── Session state ──────────────────────────────────────────────────────────────
if "sonuc" not in st.session_state:
    st.session_state.sonuc = None
if "ozellikler" not in st.session_state:
    st.session_state.ozellikler = None

# ── İlçe seçimi (form dışında — mahalle listesini anında filtreler) ────────────
st.subheader("Daire Özellikleri")
col_ilce, col_bos = st.columns(2)

with col_ilce:
    ilce_listesi = cats.get("district", [])
    district = st.selectbox("İlçe", options=ilce_listesi, index=0, key="district_select")

# Seçili ilçeye göre mahalle listesi
mahalle_secenekleri = ilce_mahalle_map.get(district, cats.get("neighborhood", []))

# ── Form ───────────────────────────────────────────────────────────────────────
with st.form("tahmin_form"):
    col1, col2 = st.columns(2)

    with col1:
        neighborhood = st.selectbox("Mahalle", options=mahalle_secenekleri, index=0)
        gross_sqm = st.number_input(
            "Brüt m²", min_value=20, max_value=1000,
            value=int(imp.get("gross_sqm", 120)),
        )
        net_sqm = st.number_input(
            "Net m²", min_value=15, max_value=900,
            value=int(imp.get("net_sqm", 95)),
        )
        total_rooms = st.number_input(
            "Oda Sayısı (3+1 → 4)", min_value=1, max_value=20,
            value=int(imp.get("total_rooms", 4)),
        )
        bathroom_count = st.number_input(
            "Banyo Sayısı", min_value=1, max_value=10,
            value=int(imp.get("bathroom_count", 1)),
        )

    with col2:
        floor = st.number_input(
            "Bulunduğu Kat", min_value=0, max_value=100,
            value=int(imp.get("floor", 3)),
        )
        total_floors = st.number_input(
            "Binadaki Toplam Kat", min_value=1, max_value=100,
            value=int(imp.get("total_floors", 8)),
        )
        building_age = st.number_input(
            "Bina Yaşı (yıl)", min_value=0, max_value=100,
            value=int(imp.get("building_age", 10)),
        )
        is_in_complex = st.radio(
            "Site İçinde mi?",
            options=[("Evet", 1), ("Hayır", 0)],
            format_func=lambda x: x[0],
            index=0,
        )
        maintenance_fee = st.number_input(
            "Aidat (TL/ay)", min_value=0, max_value=150_000,
            value=int(imp.get("maintenance_fee", 0)),
        )

    st.subheader("Kategorik Özellikler")
    col3, col4 = st.columns(2)

    def _select(label, key):
        opts    = cats.get(key, [])
        default = imp.get(key, opts[0] if opts else "")
        idx     = opts.index(default) if default in opts else 0
        return st.selectbox(label, options=opts, index=idx)

    with col3:
        heating_type       = _select("Isınma Tipi",     "heating_type")
        furnished          = _select("Eşya Durumu",     "furnished")
        usage_status       = _select("Kullanım Durumu", "usage_status")
        orientation        = _select("Cephe",           "orientation")

    with col4:
        building_type      = _select("Yapı Tipi",       "building_type")
        building_condition = _select("Yapı Durumu",     "building_condition")
        floor_category     = _select("Kat Kategorisi",  "floor_category")

    submitted = st.form_submit_button("💰 Fiyat Tahmin Et", use_container_width=True)

# ── Tahmin ─────────────────────────────────────────────────────────────────────
if submitted:
    ozellikler = {
        "district"          : district,
        "neighborhood"      : neighborhood,
        "gross_sqm"         : gross_sqm,
        "net_sqm"           : net_sqm,
        "total_rooms"       : total_rooms,
        "floor"             : floor,
        "total_floors"      : total_floors,
        "building_age"      : building_age,
        "bathroom_count"    : bathroom_count,
        "is_in_complex"     : is_in_complex[1],
        "maintenance_fee"   : maintenance_fee,
        "heating_type"      : heating_type,
        "furnished"         : furnished,
        "usage_status"      : usage_status,
        "building_type"     : building_type,
        "building_condition": building_condition,
        "floor_category"    : floor_category,
        "orientation"       : orientation,
    }
    with st.spinner("Hesaplanıyor..."):
        try:
            st.session_state.sonuc      = tahmin_et(ozellikler)
            st.session_state.ozellikler = ozellikler
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            st.stop()

# ── Sonuç göster ───────────────────────────────────────────────────────────────
if st.session_state.sonuc:
    sonuc      = st.session_state.sonuc
    ozellikler = st.session_state.ozellikler

    st.divider()
    st.subheader("Tahmin Sonucu")

    c1, c2, c3 = st.columns(3)
    c1.metric("Alt Sınır (%10)", sonuc["alt_sinir_fmt"])
    c2.metric("Nokta Tahmin (%50)", sonuc["tahmin_fmt"])
    c3.metric("Üst Sınır (%90)", sonuc["ust_sinir_fmt"])
    st.success(f"**Fiyat Aralığı:** {sonuc['aralik']}")

    st.divider()

    if st.button("📊 Fiyat Analizi Göster", use_container_width=True):
        analiz = fiyat_analizi_uret(ozellikler, sonuc, ilce_ort, mahalle_ort)
        st.subheader("📍 Fiyat & Özellik Analizi")
        st.caption("Analiz, girilen ev özellikleri ve eğitim veri seti istatistiklerine dayanmaktadır.")
        st.markdown(analiz)
