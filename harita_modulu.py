import pydeck as pdk
import pandas as pd
import numpy as np

# İlçe bazlı koordinat sözlüğü
ilce_koordinatlari = {
    'Adalar': [40.8700, 29.1200], 'Arnavutköy': [41.1819, 28.7402], 'Ataşehir': [40.9926, 29.1228],
    'Avcılar': [40.9901, 28.7118], 'Bağcılar': [41.0344, 28.8336], 'Bahçelievler': [40.9991, 28.8633],
    'Bakırköy': [40.9786, 28.8722], 'Başakşehir': [41.0963, 28.7885], 'Bayrampaşa': [41.0471, 28.8967],
    'Beşiktaş': [41.0428, 29.0076], 'Beykoz': [41.1179, 29.0955], 'Beylikdüzü': [40.9904, 28.6493],
    'Beyoğlu': [41.0369, 28.9776], 'Büyükçekmece': [41.0216, 28.5901], 'Çatalca': [41.1428, 28.4608],
    'Çekmeköy': [41.0407, 29.1724], 'Esenler': [41.0363, 28.8872], 'Esenyurt': [41.0343, 28.6811],
    'Eyüpsultan': [41.0435, 28.9340], 'Fatih': [41.0092, 28.9413], 'Gaziosmanpaşa': [41.0566, 28.9130],
    'Güngören': [41.0253, 28.8723], 'Kadıköy': [40.9910, 29.0270], 'Kağıthane': [41.0809, 28.9734],
    'Kartal': [40.8994, 29.1917], 'Küçükçekmece': [41.0006, 28.7806], 'Maltepe': [40.9416, 29.1417],
    'Pendik': [40.8767, 29.2319], 'Sancaktepe': [41.0002, 29.2310], 'Sarıyer': [41.1661, 29.0494],
    'Silivri': [41.0742, 28.2481], 'Sultanbeyli': [40.9663, 29.2678], 'Sultangazi': [41.1044, 28.8617],
    'Şile': [41.1764, 29.6101], 'Şişli': [41.0600, 28.9870], 'Tuzla': [40.8163, 29.3039],
    'Ümraniye': [41.0232, 29.1023], 'Üsküdar': [41.0274, 29.0175], 'Zeytinburnu': [40.9881, 28.8950]
}

def istanbul_3d_harita_ciz(df_harita, secilen_ilce=None):
    # Veriyi harita için hazırla
    df_map = df_harita.groupby('district')['price_per_sqm'].median().reset_index()
    df_map.columns = ['ilce', 'fiyat']

    # Koordinatları ekle
    coords = pd.DataFrame.from_dict(ilce_koordinatlari, orient='index', columns=['lat', 'lng']).reset_index()
    coords.columns = ['ilce', 'lat', 'lng']
    df_map = pd.merge(df_map, coords, on='ilce')

    # Renk skalası ve yükseklik ayarı
    q1 = df_map['fiyat'].quantile(0.33)
    q2 = df_map['fiyat'].quantile(0.66)

    def get_color(f):
        if f <= q1: return [46, 204, 113, 200]   # YEŞİL
        if f <= q2: return [241, 196, 15, 200]  # SARI
        return [231, 76, 60, 200]              # KIRMIZI

    df_map['renk'] = df_map['fiyat'].apply(get_color)
    df_map['yukseklik'] = df_map['fiyat'] / 100

    # Kamera açısı (Fly-to) kontrolü
    if secilen_ilce and secilen_ilce != "Tümü":
        lat, lng = ilce_koordinatlari[secilen_ilce]
        v_state = pdk.ViewState(latitude=lat, longitude=lng, zoom=12, pitch=60)
    else:
        v_state = pdk.ViewState(latitude=41.0082, longitude=28.9784, zoom=10, pitch=45)

    # 3D Sütun katmanı
    layer = pdk.Layer(
        "ColumnLayer",
        df_map,
        get_position=["lng", "lat"],
        get_elevation="yukseklik",
        elevation_scale=1,
        radius=400,
        get_fill_color="renk",
        pickable=True,
        auto_highlight=True,
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=v_state,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": "<b>İlçe:</b> {ilce}<br/><b>m² Fiyatı:</b> {fiyat} TL",
            "style": {"color": "white", "backgroundColor": "#2c3e50"}
        }
    )
