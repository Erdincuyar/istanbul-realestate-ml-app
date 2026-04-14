import streamlit as st
import pandas as pd
import joblib
import os
import re
import pydeck as pdk
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İstEmlak-AI | Akıllı Panel", layout="wide")

# --- DOSYA YOLLARI SİSTEMİ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(base_dir, "models", "xgboost_model.joblib")
ENCODER_PATH = os.path.join(base_dir, "models", "encoders.joblib")
DATA_PATH = os.path.join(base_dir, "data", "istanbul_apartment_prices_2026.csv")

# --- İSTANBUL İLÇE KOORDİNAT VERİ TABANI ---
# İlçe isimlerine göre haritada görünecekleri merkez noktalar
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
    if not os.path.exists(DATA_PATH):
        st.error(f"VERİ DOSYASI BULUNAMADI! Aranan yer: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)

    # İLÇE İSMİNE GÖRE KOORDİNAT ATAMA (Düzeltme Burası)
    if 'lat' not in df.columns or 'lon' not in df.columns:
        # Verideki ilçe ismini koordinat sözlüğüyle eşleştir
        # Eğer sözlükte olmayan bir ilçe varsa İstanbul'un merkezine (Fatih) koyar
        df['lat'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9416])[0])
        df['lon'] = df['district'].map(lambda x: ISTANBUL_COORDS.get(x, [41.0112, 28.9416])[1])

        # Noktalar tam üst üste binmesin diye çok küçük rastgele dağılım ekleyelim
        # Bu, aynı ilçedeki ilanların bir bulut gibi görünmesini sağlar
        df['lat'] += np.random.uniform(-0.005, 0.005, len(df))
        df['lon'] += np.random.uniform(-0.005, 0.005, len(df))

    int_cols = ['building_age', 'floor', 'total_floors', 'rooms', 'halls']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

# Modeli ve Veriyi Yükle
try:
    model, encoders = load_ml()
    df_raw = get_data()
except Exception as e:
    st.error(f"Yükleme Hatası: {e}")
    st.stop()

# --- YARDIMCI FONKSİYONLAR ---
def slugify(text):
    text = str(text).lower()
    tr_map = str.maketrans("çğıöşü ", "cgiosu-")
    text = text.translate(tr_map)
    text = re.sub(r'[^a-z0-9-]', '', text)
    return text

translate = {
    'Furnished': 'Eşyalı', 'Unfurnished': 'Eşyasız', 'Vacant': 'Boş',
    'Occupied by Owner': 'Mülk Sahibi Oturuyor', 'Occupied by Tenant': 'Kiracılı',
    'Eligible': 'Uygun', 'Not Eligible': 'Uygun Değil', 'Combi Boiler': 'Kombi',
    'Air Conditioning': 'Klima', 'Central Heating': 'Merkezi Sistem',
    'Underfloor Heating': 'Yerden Isıtma', 'Natural Gas': 'Doğalgaz',
    'Land Title': 'Arsa Tapulu', 'Construction Easement': 'Kat İrtifaklı',
    'Condominium Title': 'Kat Mülkiyetli', 'No Title Deed': 'Tapu Yok', 'Yes': 'Var', 'No': 'Yok'
}

# --- TAHMİN FONKSİYONU (Hata Ayıklamalı) ---
def predict_price(row):
    try:
        input_df = pd.DataFrame([{
            'district': encoders['district'].transform([row['district']])[0],
            'neighborhood': encoders['neighborhood'].transform([row['neighborhood']])[0],
            'rooms': row['rooms'], 'halls': row['halls'], 'gross_sqm': row['gross_sqm'],
            'building_age': row['building_age'], 'floor': row['floor'], 'total_floors': row['total_floors']
        }])
        res = model.predict(input_df)[0]
        # Eğer tahmin eksi çıkarsa mutlak değerini alarak düzelt (Sınırlı Çözüm)
        return abs(float(res))
    except Exception as e:
        st.warning(f"Tahmin Hatası (ID {row['listing_id']}): {e}")
        return 0.0

# --- YAN MENÜ (NAVİGASYON) ---
st.sidebar.title("🧭 İstEmlak Navigasyon")
page = st.sidebar.radio("Gitmek istediğiniz sayfa:", ["🏠 Fırsat Analizi", "📍 İnteraktif Fiyat Haritası"])

# ---------------------------------------------------------
# SAYFA 1: FIRSAT ANALİZİ
# ---------------------------------------------------------
if page == "🏠 Fırsat Analizi":
    st.title("🔍 Yapay Zeka ile Fırsat Analizi")
    st.write("Seçtiğiniz bölgedeki ilanların yapay zeka tarafından hesaplanan piyasa değerine göre fırsat skorlarını inceleyebilirsiniz.")

    with st.sidebar:
        st.header("📍 Bölge Seçimi")
        # İlçeleri veri setinden çek
        available_districts = sorted(df_raw['district'].unique())
        ilce = st.selectbox("İlçe Seçin", available_districts)

        # Seçilen ilçedeki mahalleleri çek
        available_neighborhoods = sorted(df_raw[df_raw['district'] == ilce]['neighborhood'].unique())
        mahalle = st.selectbox("Mahalle Seçin", available_neighborhoods)

        st.divider()
        min_firsat = st.slider("Minimum Fırsat Skoru (%)", -100, 100, 0)

    # Veriyi filtrele
    filtered = df_raw[(df_raw['district'] == ilce) & (df_raw['neighborhood'] == mahalle)].copy()

    if not filtered.empty:
        # Tahminleri Hesapla
        filtered['Tahmin'] = filtered.apply(predict_price, axis=1)
        # Fırsat Skorunu Hesapla ( (Tahmin - Fiyat) / Tahmin )
        filtered['Fırsat %'] = ((filtered['Tahmin'] - filtered['price']) / filtered['Tahmin'] * 100).round(1)

        # Fırsat skoruna göre filtrele ve sırala
        filtered = filtered[filtered['Fırsat %'] >= min_firsat].sort_values('Fırsat %', ascending=False)

        if not filtered.empty:
            # Seçim Listesi Oluştur
            secim_listesi = filtered.apply(lambda x: f"ID: {x['listing_id']} | {x['price']:,} TL | Fırsat: %{x['Fırsat %']}", axis=1).tolist()
            secilen_metin = st.selectbox("👉 İncelemek istediğiniz ilanı seçin:", secim_listesi)

            # Seçilen ilanın ID'sini ayıkla
            sel_id = secilen_metin.split(" | ")[0].split(": ")[1]
            ev = filtered[filtered['listing_id'] == sel_id].iloc[0]

            # Analiz Sonuçlarını Göster
            st.markdown("---")
            c1, c2, c3 = st.columns([1.5, 1, 1])

            with c1:
                # Hepsiemlak linkini oluştur
                i_slug, m_slug = slugify(ev['district']), slugify(ev['neighborhood'])
                url = f"https://www.hepsiemlak.com/istanbul-{i_slug}-{m_slug}-satilik/daire/{ev['listing_id']}"
                st.markdown(f"""
                <div style="background-color: #f0f7ff; border-radius: 12px; padding: 20px; border: 1px solid #cfe2ff;">
                    <h4>🌟 Yapay Zeka Analizi</h4>
                    <h2 style='color: {"#198754" if ev['Fırsat %'] > 0 else "#dc3545"};'>Fırsat Skoru: %{ev['Fırsat %']}</h2>
                    <p style="color: #666; font-size: 0.9em;">(Tahmin edilen piyasa değerinin %{ev['Fırsat %']} altında fiyata sahip.)</p>
                    <a href="{url}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">🔗 İlanı Kaynağında Gör ↗</a>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.metric("İlan Fiyatı", f"{ev['price']:,} TL")
                st.metric("Net Metrekare", f"{int(ev['net_sqm'])} m²")

            with c3:
                st.metric("Oda Sayısı", f"{ev['rooms']}+{ev['halls']}")
                st.metric("ML Piyasa Tahmini", f"{ev['Tahmin']:,.0f} TL")

            # Teknik Detaylar Tablosu
            st.write("### 🛠️ Evin Teknik Özellikleri")
            specs = {
                "Oda": f"{ev['rooms']}+{ev['halls']}",
                "Kat Bilgisi": f"{ev['floor']} / {ev['total_floors']}",
                "Bina Yaşı": str(ev['building_age']),
                "Isınma Tipi": translate.get(ev['heating_type'], ev['heating_type']),
                "Eşya Durumu": translate.get(ev['furnished'], ev['furnished']),
                "Kredi Uygunluğu": translate.get(ev['credit_eligible'], ev['credit_eligible']),
                "Tapu Durumu": translate.get(ev['title_status'], ev['title_status']),
                "Kullanım Durumu": translate.get(ev['usage_status'], ev['usage_status'])
            }
            st.table(pd.DataFrame([specs]).T.rename(columns={0: 'Değer'}))
        else:
            st.warning("Seçtiğiniz minimum fırsat skoruna uygun ilan bulunamadı.")
    else:
        st.info("Bu mahallede şu an aktif ilan bulunmuyor.")

# ---------------------------------------------------------
# SAYFA 2: İNTERAKTİF HARİTA
# ---------------------------------------------------------
else:
    st.title("📍 İstanbul İnteraktif Fiyat Haritası")
    st.write("Haritadaki noktaların üzerine gelerek fiyat ve oda bilgisini görebilirsiniz.")

    with st.sidebar:
        st.header("🗺️ Harita Filtreleri")

        # Fiyat Filtresi
        min_price = int(df_raw['price'].min())
        max_price = int(df_raw['price'].max())
        fiyat_filtresi = st.slider("Fiyat Aralığı (TL)", min_price, max_price, (min_price, int(max_price/2)))

        st.divider()
        st.write(f"Şu an {fiyat_filtresi[0]:,} TL - {fiyat_filtresi[1]:,} TL arası ilanlar gösteriliyor.")

    # Veriyi filtrele
    map_data = df_raw[(df_raw['price'] >= fiyat_filtresi[0]) & (df_raw['price'] <= fiyat_filtresi[1])]

    # Pydeck Harita Ayarları
    view_state = pdk.ViewState(
        latitude=41.0112,
        longitude=28.9784,
        zoom=10,
        pitch=45,
    )

    # Scatterplot Katmanı
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["lon", "lat"], # get_data fonksiyonunda oluşturduğumuz sütunlar
        get_color="[200, 30, 0, 160]", # Kırmızı renk
        get_radius=200, # Nokta boyutu
        pickable=True, # Bilgi kutusu için şart
    )

    # Haritayı Çiz
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10', # Açık renk harita teması
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "html": "<b>Fiyat:</b> {price} TL <br/> <b>Oda:</b> {rooms}+{halls} <br/> <b>İlçe:</b> {district}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))

    # İstatistiksel Bilgi
    st.success(f"Haritada şu an {len(map_data)} ilan gösteriliyor.")
