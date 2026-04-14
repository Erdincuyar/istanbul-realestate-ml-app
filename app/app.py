import streamlit as st
import pandas as pd
import os
import pydeck as pdk
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İstEmlak-AI | İstanbul Haritası", layout="wide")

# --- DOSYA YOLLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
DATA_PATH = os.path.join(base_dir, "data", "istanbul_apartment_prices_2026.csv")

# --- KOORDİNAT SİSTEMİ ---
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
    if not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Veri dosyası bulunamadı! Yol: {DATA_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            st.error("⚠️ Veri dosyası boş!")
            return pd.DataFrame()

        # Koordinat ataması ve noktaların hafif dağıtılması
        df['lat_base'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9784])[0])
        df['lon_base'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9784])[1])

        # Noktaların üst üste binmemesi için jitter (rastgele dağılım) ekliyoruz
        df['lat'] = df['lat_base'] + np.random.uniform(-0.012, 0.012, len(df))
        df['lon'] = df['lon_base'] + np.random.uniform(-0.012, 0.012, len(df))

        df['fiyat_str'] = df['price'].apply(lambda x: f"{x:,} TL")
        return df
    except Exception as e:
        st.error(f"❌ Hata oluştu: {e}")
        return pd.DataFrame()

# Veriyi yükle
df = load_data()

if df.empty:
    st.stop()

# --- HARİTA ARAYÜZÜ ---
st.title("📍 İstanbul İnteraktif Emlak Fiyat Haritası")
st.markdown("Noktaların üzerine gelerek **İlçe**, **Mahalle** ve **Fiyat** bilgilerini görebilirsiniz.")

# Filtreler Sidebar'da
with st.sidebar:
    st.header("⚙️ Ayarlar")
    min_f, max_f = int(df.price.min()), int(df.price.max())
    fiyat_range = st.slider("Fiyat Aralığı", min_f, max_f, (1500000, 12000000))

filtered_df = df[(df.price >= fiyat_range[0]) & (df.price <= fiyat_range[1])]

# Katman Ayarı (Scatterplot - Nokta Görünümü)
layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_df,
    get_position=["lon", "lat"],
    get_color="[225, 40, 0, 160]", # Kırmızı tonu
    get_radius=180,
    pickable=True,
)

# Harita Gövdesi
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v10',
    initial_view_state=pdk.ViewState(
        latitude=41.01,
        longitude=28.97,
        zoom=10,
        pitch=0
    ),
    layers=[layer],
    tooltip={
        "html": """
            <div style="font-family: sans-serif; font-size: 14px;">
                <b>İlçe:</b> {district} <br/>
                <b>Mahalle:</b> {neighborhood} <br/>
                <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ccc;">
                <b>Fiyat:</b> <span style="color: #ff4b4b;">{fiyat_str}</span> <br/>
                <b>Oda Sayısı:</b> {rooms}+{halls}
            </div>
        """,
        "style": {"backgroundColor": "white", "color": "black", "borderRadius": "5px", "padding": "10px"}
    }
))

st.info(f"💡 Filtrelere göre **{len(filtered_df):,}** ilan gösteriliyor.")
