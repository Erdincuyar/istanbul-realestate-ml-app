import pydeck as pdk
import pandas as pd

def istanbul_3d_harita_ciz(df_harita, secilen_ilce=None):
    # Eğer Tümü seçilmediyse veriyi sadece o ilçeye göre filtrele
    if secilen_ilce and secilen_ilce != "Tümü":
        df_map = df_harita[df_harita['district'] == secilen_ilce].copy()
    else:
        df_map = df_harita.copy()

    # Mahalle bazında gruplayarak medyan fiyat ve gerçek koordinatları al
    df_map = df_map.groupby(['district', 'neighborhood']).agg({
        'price_per_sqm': 'median',
        'lat': 'first', # Mahallenin gerçek enlemi
        'lon': 'first'  # Mahallenin gerçek boylamı
    }).reset_index()

    df_map.columns = ['ilce', 'mahalle', 'fiyat', 'lat', 'lng']

    # Renk skalası (Tüm İstanbul'a göre hesapla ki renkler tutarlı olsun)
    q1 = df_map['fiyat'].quantile(0.33)
    q2 = df_map['fiyat'].quantile(0.66)

    def get_color(f):
        if f <= q1: return [46, 204, 113, 200]   # YEŞİL
        if f <= q2: return [241, 196, 15, 200]   # SARI
        return [231, 76, 60, 200]                # KIRMIZI

    df_map['renk'] = df_map['fiyat'].apply(get_color)
    df_map['yukseklik'] = df_map['fiyat'] / 100

    # Kamera açısı: Seçilen ilçenin tam ortasına otomatik zoom yap!
    if len(df_map) > 0:
        merkez_lat = df_map['lat'].mean()
        merkez_lon = df_map['lng'].mean()
        v_state = pdk.ViewState(latitude=merkez_lat, longitude=merkez_lon, zoom=12 if secilen_ilce != "Tümü" else 10, pitch=45)
    else:
        v_state = pdk.ViewState(latitude=41.0082, longitude=28.9784, zoom=10, pitch=45)

    # 3D Sütun katmanı
    layer = pdk.Layer(
        "ColumnLayer",
        df_map,
        get_position=["lng", "lat"],
        get_elevation="yukseklik",
        elevation_scale=1,
        radius=200, # Sütunları incelttik ki mahalleler birbirine girmesin
        get_fill_color="renk",
        pickable=True,
        auto_highlight=True,
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=v_state,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": "<b>İlçe:</b> {ilce}<br/><b>Mahalle:</b> {mahalle}<br/><b>m² Fiyatı:</b> {fiyat} TL",
            "style": {"color": "white", "backgroundColor": "#2c3e50"}
        }
    )
