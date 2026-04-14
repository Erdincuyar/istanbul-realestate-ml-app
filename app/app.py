import streamlit as st
import pandas as pd
import joblib
import os
import re

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İstEmlak-AI | ML Fırsat Dedektörü", layout="wide")

# --- DOSYA YOLLARI SİSTEMİ (Hata Önleyici) ---
# app.py'nin bulunduğu klasörü bulur (app/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Bir üst klasöre çıkarak models ve data klasörlerine gider
BASE_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoders.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "istanbul_apartment_prices_2026.csv")

# --- ASSET YÜKLEME ---
@st.cache_resource
def load_ml():
    # Eğer dosyalar bulunamazsa kullanıcıya net bir hata mesajı gösterir
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        st.error(f"Model dosyaları bulunamadı! Aranan konum: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

@st.cache_data
def get_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Veri dosyası bulunamadı! Aranan konum: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    int_cols = ['building_age', 'floor', 'total_floors', 'rooms', 'halls']
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

# Modeli ve Veriyi Yükle
model, encoders = load_ml()
df_raw = get_data()

# --- YARDIMCI FONKSİYON: SEO LİNK ---
def slugify(text):
    text = str(text).lower()
    tr_map = str.maketrans("çğıöşü ", "cgiosu-")
    text = text.translate(tr_map)
    text = re.sub(r'[^a-z0-9-]', '', text)
    return text

# --- TÜRKÇE ÇEVİRİ SÖZLÜĞÜ ---
translate = {
    'Furnished': 'Eşyalı', 'Unfurnished': 'Eşyasız',
    'Vacant': 'Boş', 'Occupied by Owner': 'Mülk Sahibi Oturuyor', 'Occupied by Tenant': 'Kiracılı',
    'Eligible': 'Uygun', 'Not Eligible': 'Uygun Değil',
    'Combi Boiler': 'Kombi', 'Air Conditioning': 'Klima', 'Central Heating': 'Merkezi Sistem',
    'Underfloor Heating': 'Yerden Isıtma', 'Natural Gas': 'Doğalgaz',
    'Land Title': 'Arsa Tapulu', 'Construction Easement': 'Kat İrtifaklı',
    'Condominium Title': 'Kat Mülkiyetli', 'No Title Deed': 'Tapu Yok',
    'Yes': 'Var', 'No': 'Yok'
}

# --- ANALİZ FONKSİYONU ---
def predict_price(row):
    input_df = pd.DataFrame([{
        'district': encoders['district'].transform([row['district']])[0],
        'neighborhood': encoders['neighborhood'].transform([row['neighborhood']])[0],
        'rooms': row['rooms'], 'halls': row['halls'], 'gross_sqm': row['gross_sqm'],
        'building_age': row['building_age'], 'floor': row['floor'], 'total_floors': row['total_floors']
    }])
    return model.predict(input_df)[0]

# --- ARAYÜZ ---
st.title("🏠 İstEmlak-AI | Akıllı Emlak Paneli")

with st.sidebar:
    st.header("📍 Bölge Seçimi")
    ilce = st.selectbox("İlçe", sorted(df_raw['district'].unique()))
    mahalle = st.selectbox("Mahalle", sorted(df_raw[df_raw['district'] == ilce]['neighborhood'].unique()))
    st.divider()
    min_firsat = st.slider("Minimum Fırsat Skoru (%)", -50, 50, 0)

filtered = df_raw[(df_raw['district'] == ilce) & (df_raw['neighborhood'] == mahalle)].copy()

if not filtered.empty:
    filtered['Tahmin'] = filtered.apply(predict_price, axis=1)
    filtered['Fırsat %'] = ((filtered['Tahmin'] - filtered['price']) / filtered['Tahmin'] * 100).round(1)
    filtered = filtered[filtered['Fırsat %'] >= min_firsat].sort_values('Fırsat %', ascending=False)

    if not filtered.empty:
        secim_listesi = filtered.apply(lambda x: f"ID: {x['listing_id']} | {x['price']:,} TL | Fırsat: %{x['Fırsat %']}", axis=1).tolist()
        secilen_metin = st.selectbox("👉 Detaylarını görmek istediğiniz evi seçin:", secim_listesi)

        sel_id = secilen_metin.split(" | ")[0].split(": ")[1]
        ev = filtered[filtered['listing_id'] == sel_id].iloc[0]

        st.markdown("---")

        col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:
            ilce_slug = slugify(ev['district'])
            mahalle_slug = slugify(ev['neighborhood'])
            hepsiemlak_url = f"https://www.hepsiemlak.com/istanbul-{ilce_slug}-{mahalle_slug}-satilik/daire/{ev['listing_id']}"

            st.markdown(f"""
            <div style="background-color: #f0f7ff; border-radius: 12px; padding: 20px; border: 1px solid #cfe2ff;">
                <h4 style='margin-top:0; color: #555;'>🌟 Yapay Zeka Analizi</h4>
                <h2 style='color: {"#198754" if ev['Fırsat %'] > 0 else "#dc3545"}; margin: 5px 0;'>
                    Fırsat Skoru: %{ev['Fırsat %']}
                </h2>
                <p style='font-size: 14px; color: #666;'>{ev['district']} / {ev['neighborhood']}</p>
                <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 10px;">
                    <a href="{hepsiemlak_url}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold; font-size: 15px;">
                        🔗 İlanı Kaynağında Gör ↗
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric("İlan Fiyatı", f"{ev['price']:,} TL")
            st.metric("Net Metrekare", f"{int(ev['net_sqm'])} m²")

        with col3:
            st.metric("Oda Sayısı", f"{ev['rooms']}+{ev['halls']}")
            st.metric("ML Piyasa Tahmini", f"{ev['Tahmin']:,.0f} TL")

        st.write("### 🛠️ Evin Teknik Özellikleri")
        specs = {
            "Oda Sayısı": f"{ev['rooms']}+{ev['halls']}",
            "Net Metrekare": f"{int(ev['net_sqm'])} m²",
            "Bulunduğu Kat": str(ev['floor']),
            "Bina Kat Sayısı": str(ev['total_floors']),
            "Bina Yaşı": str(ev['building_age']),
            "Isınma Tipi": translate.get(ev['heating_type'], "Belirtilmemiş"),
            "Eşya Durumu": translate.get(ev['furnished'], "Belirtilmemiş"),
            "Kullanım Durumu": translate.get(ev['usage_status'], "Belirtilmemiş"),
            "Krediye Uygunluk": translate.get(ev['credit_eligible'], "Belirtilmemiş"),
            "Tapu Durumu": translate.get(ev['deed_status'], "Belirtilmemiş")
        }
        st.table(pd.DataFrame([specs]).T.rename(columns={0: 'Değer'}))
    else:
        st.warning("Fırsat skoruna uygun ilan bulunamadı.")
else:
    st.warning("Bu bölgede ilan bulunamadı.")
