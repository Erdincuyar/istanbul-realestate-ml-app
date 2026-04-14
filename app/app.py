import streamlit as st
import pandas as pd
import joblib
import os
import re
import pydeck as pdk
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İstEmlak-AI | Analiz", layout="wide")

# --- DOSYA YOLLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(base_dir, "models", "xgboost_model.joblib")
ENCODER_PATH = os.path.join(base_dir, "models", "encoders.joblib")
DATA_PATH = os.path.join(base_dir, "data", "istanbul_apartment_prices_2026.csv")

# --- KOORDİNAT VERİSİ ---
ISTANBUL_COORDS = {
    'Adalar': [40.8732, 29.1278], 'Arnavutköy': [41.1852, 28.7400], 'Ataşehir': [40.9928, 29.1244],
    'Avcılar': [40.9801, 28.7175], 'Bağcılar': [41.0343, 28.8336], 'Bahçelievler': [40.9990, 28.8637],
    'Bakırköy': [40.9782, 28.8741], 'Başakşehir': [41.1091, 28.7884], 'Bayrampaşa': [41.0347, 28.9118],
    'Beşiktaş': [41.0428, 29.0075], 'Beykoz': [41.1171, 29.0970], 'Beylikdüzü': [41.0016, 28.6419],
    'Beyoğlu': [41.0369, 28.9774], 'Büyükçekmece': [41.0210, 28.5830], 'Çatalca': [41.1430, 28.4619],
    'Çekmeköy': [41.0336, 29.1751], 'Esenler': [41.0357, 28.8911], 'Esenyurt': [41.0343, 28.6800],
    'Eyüpsultan': [41.0475, 28.9329], 'Fatih': [41.0112, 28.9416], 'Gaziosmanpaşa': [41.0566, 28.9130],
    'Güngören': [41.0219, 28.8718], 'Kadıköy': [40.9910, 29.0270], 'Kağıthane': [41.0818, 28.9723],
    'Kartal': [40.8885, 29.1854], 'Küçükçekmece': [40.9918, 28.7719], 'Maltepe': [40.9255, 29.1333],
    'Pendik': [40.8769, 29.2338], 'Sancaktepe': [41.0063, 29.2274], 'Sarıyer': [41.1687, 29.0463],
    'Silivri': [41.0736, 28.2464], 'Sultanbeyli': [40.9631, 29.2680], 'Sultangazi': [41.1044, 28.8837],
    'Şile': [41.1754, 29.6133], 'Şişli': [41.0600, 28.9870], 'Tuzla': [40.8163, 29.3031],
    'Ümraniye': [41.0166, 29.1171], 'Üsküdar': [41.0200, 29.0200], 'Zeytinburnu': [40.9880, 28.8950]
}

@st.cache_data
def get_data():
    df = pd.read_csv(DATA_PATH)
    if 'lat' not in df.columns:
        df['lat'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9416])[0])
        df['lon'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9416])[1])
        df['lat'] += np.random.uniform(-0.02, 0.02, len(df))
        df['lon'] += np.random.uniform(-0.02, 0.02, len(df))
    return df

model, encoders = joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)
df_raw = get_data()

# --- NAVİGASYON ---
page = st.sidebar.radio("Sayfa:", ["🏠 Fırsat Analizi", "📍 3D Isı Haritası"])

if page == "🏠 Fırsat Analizi":
    st.title("🔍 Akıllı İlan Analizi")
    ilce = st.sidebar.selectbox("İlçe", sorted(df_raw['district'].unique()))
    mahalle = st.sidebar.selectbox("Mahalle", sorted(df_raw[df_raw['district'] == ilce]['neighborhood'].unique()))
    filtered = df_raw[(df_raw['district'] == ilce) & (df_raw['neighborhood'] == mahalle)].copy()

    if not filtered.empty:
        def quick_predict(row):
            in_df = pd.DataFrame([{
                'district': encoders['district'].transform([row['district']])[0],
                'neighborhood': encoders['neighborhood'].transform([row['neighborhood']])[0],
                'rooms': row['rooms'], 'halls': row['halls'], 'gross_sqm': row['gross_sqm'],
                'building_age': row['building_age'], 'floor': row['floor'], 'total_floors': row['total_floors']
            }])
            return abs(float(model.predict(in_df)[0]))

        filtered['Tahmin'] = filtered.apply(quick_predict, axis=1)
        filtered['Fırsat %'] = ((filtered['Tahmin'] - filtered['price']) / filtered['Tahmin'] * 100).round(1)
        secim = st.selectbox("İlanlar:", filtered.apply(lambda x: f"ID: {x['listing_id']} | {x['price']:,} TL", axis=1))
        ev = filtered[filtered['listing_id'] == secim.split(" | ")[0].split(": ")[1]].iloc[0]

        c1, c2 = st.columns(2)
        c1.metric("Piyasa Tahmini", f"{ev['Tahmin']:,.0f} TL")
        c2.metric("Fırsat Skoru", f"%{ev['Fırsat %']}")

else:
    st.title("📍 İstanbul 3D Fiyat Isı Haritası")
    fiyat_range = st.sidebar.slider("Fiyat (TL)", int(df_raw.price.min()), int(df_raw.price.max()), (1000000, 15000000))
    map_data = df_raw[(df_raw.price >= fiyat_range[0]) & (df_raw.price <= fiyat_range[1])]

    # --- KESİN ÇÖZÜM KATMANI ---
    layer = pdk.Layer(
        "HexagonLayer",
        map_data,
        get_position=["lon", "lat"],
        radius=700,
        elevation_scale=15,
        elevation_range=[0, 1000],
        extruded=True,
        pickable=True,
        get_weight="price",
        aggregation=pdk.types.String("MEAN")
    )

    # Arka planı koyu (dark) yaparak sütunların parlamasını sağlıyoruz
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(latitude=41.01, longitude=28.97, zoom=10, pitch=50),
        layers=[layer],
        tooltip={
            "html": "<b>Yoğunluk:</b> {count} ilan<br><b>Ort. Fiyat:</b> {elevationValue} TL",
            "style": {"color": "white"}
        }
    ))
