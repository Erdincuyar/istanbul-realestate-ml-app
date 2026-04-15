import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import anthropic
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

# ── Session state ──────────────────────────────────────────────────────────────
if "sonuc" not in st.session_state:
    st.session_state.sonuc = None
if "ozellikler" not in st.session_state:
    st.session_state.ozellikler = None

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

    # ── AI Analizi ─────────────────────────────────────────────────────────────
    if st.button("🤖 AI ile Konum & Fiyat Analizi Üret", use_container_width=True):

        # API anahtarları
        try:
            serper_key    = st.secrets["SERPER_API_KEY"]
            anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        except KeyError as e:
            st.error(f"Streamlit secrets içinde {e} bulunamadı.")
            st.stop()

        mahalle = ozellikler["neighborhood"]
        ilce    = ozellikler["district"]

        # ── Serper: mahalle → ilçe → boş waterfall ────────────────────────────
        def serper_ara(query: str) -> list:
            try:
                r = requests.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY"   : serper_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "gl": "tr", "hl": "tr", "num": 5},
                    timeout=10,
                )
                return [
                    item["snippet"]
                    for item in r.json().get("organic", [])
                    if item.get("snippet")
                ]
            except Exception:
                return []

        with st.spinner("Bölge bilgisi aranıyor..."):
            snippets = serper_ara(
                f"{mahalle} {ilce} İstanbul yatırım ulaşım değer proje"
            )
            kaynak = f"{mahalle}, {ilce}"

            if len(snippets) < 2:
                snippets = serper_ara(
                    f"{ilce} İstanbul emlak yatırım değer proje"
                )
                kaynak = ilce

        bolge_metni = "\n".join(snippets) if snippets else ""

        # ── Claude prompt ──────────────────────────────────────────────────────
        prompt = f"""İstanbul'da satılık bir daire için fiyat ve konum analizi yaz. Türkçe yaz. 3-4 paragraf olsun.

EV ÖZELLİKLERİ:
- Konum: {mahalle}, {ilce}
- Brüt m²: {ozellikler['gross_sqm']} | Net m²: {ozellikler['net_sqm']}
- Oda: {ozellikler['total_rooms']} | Banyo: {ozellikler['bathroom_count']}
- Kat: {ozellikler['floor']}/{ozellikler['total_floors']} | Bina yaşı: {ozellikler['building_age']} yıl
- Isınma: {ozellikler['heating_type']} | Eşya: {ozellikler['furnished']}
- Site: {"Evet" if ozellikler['is_in_complex'] else "Hayır"} | Aidat: {ozellikler['maintenance_fee']} TL/ay

FİYAT TAHMİNİ (XGBoost Quantile modeli):
- Alt sınır (%10): {sonuc['alt_sinir_fmt']}
- Nokta tahmin (%50): {sonuc['tahmin_fmt']}
- Üst sınır (%90): {sonuc['ust_sinir_fmt']}

BÖLGE BİLGİSİ ({kaynak} bazında web araması):
{bolge_metni if bolge_metni else "Web aramasından yeterli veri bulunamadı."}

KURALLAR:
- Fiyatın neden makul olduğunu açıkla
- Bölgenin yatırım potansiyelini ve değer artışını değerlendir
- Ulaşım, sosyal çevre ve çevredeki projelere değin (varsa)
- Bölge bilgisi yoksa sadece ev özelliklerinden yorum yap
- Bilmediğin ya da verilmeyen bilgileri uydurma
"""

        with st.spinner("AI analizi yazılıyor..."):
            try:
                client   = anthropic.Anthropic(api_key=anthropic_key)
                message  = client.messages.create(
                    model      = "claude-sonnet-4-6",
                    max_tokens = 1024,
                    messages   = [{"role": "user", "content": prompt}],
                )
                analiz = message.content[0].text
            except Exception as e:
                st.error(f"Claude API hatası: {e}")
                st.stop()

        st.subheader("📍 Konum & Fiyat Analizi")
        if bolge_metni:
            st.caption(f"Bölge verisi: {kaynak} bazında web araması")
        else:
            st.caption("Web verisi bulunamadı — ev özelliklerine göre analiz yapıldı")
        st.write(analiz)