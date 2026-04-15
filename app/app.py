import sys
import os

# repo kökünü path'e ekle (fiyat_tahmin_pipeline import için)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
from fiyat_tahmin_pipeline import tahmin_et, ARTIFACT_DIR

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
        "Önce `python fiyat_tahmin_pipeline.py --egit` komutuyla modeli eğitin "
        "ve `model_artifacts/` klasörünü repoya ekleyin."
    )
    st.stop()

with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)

cats = cfg.get("cat_categories", {})
imp  = cfg.get("imputer_vals", {})

# ── Model metrikleri ───────────────────────────────────────────────────────────
with st.expander("Model Performansı", expanded=False):
    c1, c2 = st.columns(2)
    c1.metric("CV-R²", cfg.get("cv_r2", "—"))
    c2.metric("CV-MAPE", f"%{cfg.get('cv_mape_pct', '—')}")

st.divider()

# ── Form ───────────────────────────────────────────────────────────────────────
with st.form("tahmin_form"):
    st.subheader("Daire Özellikleri")

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox(
            "İlçe",
            options=cats.get("district", []),
            index=0,
        )
        neighborhood = st.selectbox(
            "Mahalle",
            options=cats.get("neighborhood", []),
            index=0,
        )
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
        opts = cats.get(key, [])
        default = imp.get(key, opts[0] if opts else "")
        idx = opts.index(default) if default in opts else 0
        return st.selectbox(label, options=opts, index=idx)

    with col3:
        heating_type       = _select("Isınma Tipi",       "heating_type")
        furnished          = _select("Eşya Durumu",       "furnished")
        usage_status       = _select("Kullanım Durumu",   "usage_status")
        orientation        = _select("Cephe",             "orientation")

    with col4:
        building_type      = _select("Yapı Tipi",         "building_type")
        building_condition = _select("Yapı Durumu",       "building_condition")
        floor_category     = _select("Kat Kategorisi",    "floor_category")

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
            sonuc = tahmin_et(ozellikler)
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            st.stop()

    st.divider()
    st.subheader("Tahmin Sonucu")

    c1, c2, c3 = st.columns(3)
    c1.metric("Alt Sınır (%10)", sonuc["alt_sinir_fmt"])
    c2.metric("Nokta Tahmin (%50)", sonuc["tahmin_fmt"])
    c3.metric("Üst Sınır (%90)", sonuc["ust_sinir_fmt"])

    st.success(f"**Fiyat Aralığı:** {sonuc['aralik']}")
