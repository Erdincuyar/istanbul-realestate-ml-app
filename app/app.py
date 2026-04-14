import streamlit as st
import pandas as pd
import os
import pydeck as pdk
import numpy as np

st.set_page_config(page_title="İstEmlak-AI | İstanbul Haritası", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
DATA_PATH = os.path.join(base_dir, "data", "istanbul_apartment_prices_2026.csv")

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
def load_data():
    df = pd.read_csv(DATA_PATH)

    df['lat_base'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.01, 28.97])[0])
    df['lon_base'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.01, 28.97])[1])

    # Daha geniş jitter (üst üste binmeyi azaltır)
    df['lat'] = df['lat_base'] + np.random.normal(0, 0.01, len(df))
    df['lon'] = df['lon_base'] + np.random.normal(0, 0.01, len(df))

    df['fiyat_str'] = df['price'].apply(lambda x: f"{x:,} TL")

    # Yüksekliği normalize et (çok uzun olmasın)
    df['height'] = (df['price'] / df['price'].max()) * 10000

    return df

df = load_data()

st.title("📍 İstanbul 3D Emlak Haritası")

with st.sidebar:
    min_f, max_f = int(df.price.min()), int(df.price.max())
    fiyat_range = st.slider("Fiyat Aralığı", min_f, max_f, (1500000, 12000000))

filtered_df = df[(df.price >= fiyat_range[0]) & (df.price <= fiyat_range[1])]

# ✅ 3D COLUMN LAYER
layer = pdk.Layer(
    "ColumnLayer",
    data=filtered_df,
    get_position='[lon, lat]',
    get_elevation='height',
    elevation_scale=5,   # 🔥 artırdık
    radius=200,          # 🔥 artırdık
    get_fill_color='[255, 100, 0, 200]',
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=41.01,
    longitude=28.97,
    zoom=10,
    pitch=60,   # 🔥 artır (45 → 60)
    bearing=20  # hafif açı ver
)


deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v9',
    tooltip={
        "html": """
        <b>İlçe:</b> {district} <br/>
        <b>Mahalle:</b> {neighborhood} <br/>
        <b>Fiyat:</b> {fiyat_str}
        """
    }
)

st.pydeck_chart(deck)

st.info(f"{len(filtered_df):,} ilan gösteriliyor")
