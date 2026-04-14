import streamlit as st
import pandas as pd
import joblib
import os
import re
import pydeck as pdk

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İstEmlak-AI | ML Fırsat Analizi", layout="wide")

# --- DOSYA YOLLARI SİSTEMİ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(base_dir, "models", "xgboost_model.joblib")
ENCODER_PATH = os.path.join(base_dir, "models", "encoders.joblib")
DATA_PATH = os.path.join(base_dir, "data", "istanbul_apartment_prices_2026.csv")

# --- ASSET YÜKLEME ---
@st.cache_resource
def load_ml():
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

@st.cache_data
def get_data():
    df = pd.read_csv(DATA_PATH)
    # Koordinat sütunları yoksa demo amaçlı merkez noktalar ekliyoruz
    if 'lat' not in df.columns:
        # Bu kısım sadece görselleştirme içindir
        df['lat'] = 41.0082
        df['lon'] = 28.9784
    return df

model, encoders = load_ml()
df_raw = get_data()

# --- YAN MENÜ (NAVİGASYON) ---
st.sidebar.title("🧭 İstEmlak Navigasyon")
page = st.sidebar.radio("Gitmek istediğiniz sayfa:", ["🏠 Fırsat Analizi", "📍 Fiyat Haritası"])

# --- TÜRKÇE SÖZLÜK ---
translate = {'Furnished': 'Eşyalı', 'Unfurnished': 'Eşyasız', 'Yes': 'Var', 'No': 'Yok'}

def slugify(text):
    tr_map = str.maketrans("çğıöşü ", "cgiosu-")
    return str(text).lower().translate(tr_map)

# ---------------------------------------------------------
# SAYFA 1: FIRSAT ANALİZİ
# ---------------------------------------------------------
if page == "🏠 Fırsat Analizi":
    st.title("🔍 Yapay Zeka ile Fırsat Analizi")

    ilce = st.sidebar.selectbox("İlçe Seçin", sorted(df_raw['district'].unique()))
    mahalle = st.sidebar.selectbox("Mahalle Seçin", sorted(df_raw[df_raw['district'] == ilce]['neighborhood'].unique()))

    filtered = df_raw[(df_raw['district'] == ilce) & (df_raw['neighborhood'] == mahalle)].copy()

    if not filtered.empty:
        # Tahmin ve Fırsat Hesaplama
        def quick_predict(row):
            try:
                in_df = pd.DataFrame([{
                    'district': encoders['district'].transform([row['district']])[0],
                    'neighborhood': encoders['neighborhood'].transform([row['neighborhood']])[0],
                    'rooms': row['rooms'], 'halls': row['halls'], 'gross_sqm': row['gross_sqm'],
                    'building_age': row['building_age'], 'floor': row['floor'], 'total_floors': row['total_floors']
                }])
                return float(model.predict(in_df)[0])
            except: return 0.0

        filtered['Tahmin'] = filtered.apply(quick_predict, axis=1)
        filtered['Fırsat %'] = ((filtered['Tahmin'] - filtered['price']) / filtered['Tahmin'] * 100).round(1)

        # En yüksek fırsatları göster
        firsatlar = filtered.sort_values('Fırsat %', ascending=False)
        secim = st.selectbox("İlanları İncele:", firsatlar.apply(lambda x: f"ID: {x['listing_id']} - %{x['Fırsat %']} Fırsat", axis=1))

        ev = firsatlar[firsatlar['listing_id'] == secim.split(" - ")[0].split(": ")[1]].iloc[0]

        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            st.success(f"Bu ev piyasa değerinin %{ev['Fırsat %']} altında olabilir!")
            st.metric("Piyasa Tahmini", f"{ev['Tahmin']:,.0f} TL")
            st.write(f"**İlçe/Mahalle:** {ev['district']} / {ev['neighborhood']}")
        with c2:
            st.metric("İlan Fiyatı", f"{ev['price']:,} TL")
        with c3:
            st.metric("Net Alan", f"{ev['net_sqm']} m²")

# ---------------------------------------------------------
# SAYFA 2: HARİTA GÖRÜNÜMÜ
# ---------------------------------------------------------
else:
    st.title("📍 İstanbul İnteraktif Fiyat Haritası")

    fiyat_filtresi = st.sidebar.slider("Fiyat Aralığı (Milyon TL)", 0, 100, (1, 20))
    map_data = df_raw[(df_raw['price'] >= fiyat_filtresi[0]*1e6) & (df_raw['price'] <= fiyat_filtresi[1]*1e6)]

    # Pydeck Harita Ayarları
    view_state = pdk.ViewState(latitude=41.0082, longitude=28.9784, zoom=10, pitch=45)

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["lon", "lat"],
        get_color="[200, 30, 0, 160]",
        get_radius=200,
        pickable=True,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Fiyat: {price} TL\nMahalle: {neighborhood}"}
    ))
