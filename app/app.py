import streamlit as st
import pandas as pd
import joblib
import os
import re
import pydeck as pdk
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İstEmlak-AI | Pro Analiz", layout="wide")

# --- DOSYA YOLLARI SİSTEMİ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(base_dir, "models", "xgboost_model.joblib")
ENCODER_PATH = os.path.join(base_dir, "models", "encoders.joblib")
DATA_PATH = os.path.join(base_dir, "data", "istanbul_apartment_prices_2026.csv")

# --- İSTANBUL İLÇE KOORDİNAT VERİ TABANI ---
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

# --- ASSET YÜKLEME ---
@st.cache_resource
def load_ml():
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

@st.cache_data
def get_data():
    df = pd.read_csv(DATA_PATH)
    if 'lat' not in df.columns or 'lon' not in df.columns:
        df['lat'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9416])[0])
        df['lon'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9416])[1])
        # İlçe içinde hafif dağılım (Sütunlar çok uzun olmasın diye 0.02 yaptık)
        df['lat'] += np.random.uniform(-0.02, 0.02, len(df))
        df['lon'] += np.random.uniform(-0.02, 0.02, len(df))
    return df

try:
    model, encoders = load_ml()
    df_raw = get_data()
except Exception as e:
    st.error(f"Hata: {e}")
    st.stop()

# --- YARDIMCI ARAÇLAR ---
def slugify(text):
    tr_map = str.maketrans("çğıöşü ", "cgiosu-")
    return str(text).lower().translate(tr_map)

translate = {'Furnished': 'Eşyalı', 'Unfurnished': 'Eşyasız', 'Yes': 'Var', 'No': 'Yok'}

# --- NAVİGASYON ---
st.sidebar.title("🧭 İstEmlak AI")
page = st.sidebar.radio("Sayfa Seçin:", ["🏠 Fırsat Analizi", "📍 3D Fiyat Haritası"])

# ---------------------------------------------------------
# SAYFA 1: ANALİZ
# ---------------------------------------------------------
if page == "🏠 Fırsat Analizi":
    st.title("🔍 Akıllı İlan Analizi")
    ilce = st.sidebar.selectbox("İlçe", sorted(df_raw['district'].unique()))
    mahalle = st.sidebar.selectbox("Mahalle", sorted(df_raw[df_raw['district'] == ilce]['neighborhood'].unique()))

    filtered = df_raw[(df_raw['district'] == ilce) & (df_raw['neighborhood'] == mahalle)].copy()

    if not filtered.empty:
        def predict_price(row):
            try:
                in_df = pd.DataFrame([{
                    'district': encoders['district'].transform([row['district']])[0],
                    'neighborhood': encoders['neighborhood'].transform([row['neighborhood']])[0],
                    'rooms': row['rooms'], 'halls': row['halls'], 'gross_sqm': row['gross_sqm'],
                    'building_age': row['building_age'], 'floor': row['floor'], 'total_floors': row['total_floors']
                }])
                return abs(float(model.predict(in_df)[0]))
            except: return 0.0

        filtered['Tahmin'] = filtered.apply(predict_price, axis=1)
        filtered['Fırsat %'] = ((filtered['Tahmin'] - filtered['price']) / filtered['Tahmin'] * 100).round(1)

        secilen_ilan = st.selectbox("İlan seçin:", filtered.apply(lambda x: f"ID: {x['listing_id']} | {x['price']:,} TL", axis=1))
        ev = filtered[filtered['listing_id'] == secilen_ilan.split(" | ")[0].split(": ")[1]].iloc[0]

        st.divider()
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            st.metric("Fırsat Skoru", f"%{ev['Fırsat %']}")
            st.write(f"**Konum:** {ev['district']} / {ev['neighborhood']}")
        with c2:
            st.metric("İlan Fiyatı", f"{ev['price']:,} TL")
            st.metric("Piyasa Tahmini", f"{ev['Tahmin']:,.0f} TL")
        with c3:
            st.metric("Alan", f"{ev['net_sqm']} m²")
            st.metric("Oda", f"{ev['rooms']}+{ev['halls']}")

# ---------------------------------------------------------
# SAYFA 2: HARİTA
# ---------------------------------------------------------
else:
    st.title("📍 İstanbul 3D Fiyat Isı Haritası")

    min_p, max_p = int(df_raw['price'].min()), int(df_raw['price'].max())
    fiyat_range = st.sidebar.slider("Fiyat Filtresi (TL)", min_p, max_p, (min_p, int(max_p/4)))

    map_data = df_raw[(df_raw['price'] >= fiyat_range[0]) & (df_raw['price'] <= fiyat_range[1])]

    # 3D Hexagon Katmanı
    hexagon_layer = pdk.Layer(
        "HexagonLayer",
        map_data,
        get_position=["lon", "lat"],
        radius=800,
        elevation_scale=12,
        elevation_range=[0, 1500],
        extruded=True,
        pickable=True,
        get_weight="price",
        aggregation=pdk.types.String("MEAN"), # Fiyatların ortalamasını al
        color_range=[
            [254, 240, 217], [253, 204, 138], [252, 141, 89], [227, 74, 51], [179, 0, 0]
        ],
    )

    st.pydeck_chart(pdk.Deck(
        map_style='https://basemaps.cartocdn.com/gl/light-v10/style.json',
        initial_view_state=pdk.ViewState(latitude=41.0112, longitude=28.9784, zoom=10, pitch=60),
        layers=[hexagon_layer],
        tooltip={
            "html": """
                <div style="font-family: sans-serif;">
                    <b>📊 Bölge Analizi</b><br/>
                    <b>Ortalama Fiyat:</b> {elevationValue:,.0f} TL<br/>
                    <b>İlan Sayısı:</b> {count}<br/>
                </div>
            """,
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))
    st.success(f"Haritada {len(map_data):,} ilan analiz ediliyor.")
