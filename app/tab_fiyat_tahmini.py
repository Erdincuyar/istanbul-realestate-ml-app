import streamlit as st


def render(cfg, cats, imp, ilce_mahalle_map, ilce_ort, mahalle_ort, fmt_tl, fiyat_analizi_uret, md2html):
    st.markdown('<p style="color:#AAAAAA; font-size:15px; margin-bottom:20px;">Daire özelliklerini girerek XGBoost Quantile modeli ile %10 – %50 – %90 fiyat aralığı tahmin edin.</p>', unsafe_allow_html=True)

    if "sonuc" not in st.session_state:
        st.session_state.sonuc = None
    if "ozellikler" not in st.session_state:
        st.session_state.ozellikler = None

    st.markdown('<div class="section-title">Konum</div>', unsafe_allow_html=True)
    ilce_col, _ = st.columns(2)
    with ilce_col:
        district_t2 = st.selectbox("İlçe", cats.get("district", []), index=0, key="t2_ilce")

    mahalle_sec_t2 = ilce_mahalle_map.get(district_t2, cats.get("neighborhood", []))

    st.markdown('<div class="section-title">Fiziksel Özellikler</div>', unsafe_allow_html=True)

    with st.form("tahmin_form_v3"):
        col1, col2 = st.columns(2)

        with col1:
            neighborhood_t2 = st.selectbox("Mahalle",               options=mahalle_sec_t2, index=0)
            gross_sqm       = st.number_input("Brüt m²",            min_value=20,  max_value=1000, value=int(imp.get("gross_sqm", 120)))
            net_sqm         = st.number_input("Net m²",             min_value=15,  max_value=900,  value=int(imp.get("net_sqm", 95)))
            total_rooms     = st.number_input("Oda Sayısı (3+1→4)", min_value=1,   max_value=20,   value=int(imp.get("total_rooms", 4)))
            bathroom_count  = st.number_input("Banyo Sayısı",       min_value=1,   max_value=10,   value=int(imp.get("bathroom_count", 1)))

        with col2:
            floor           = st.number_input("Bulunduğu Kat",      min_value=0,   max_value=100,  value=int(imp.get("floor", 3)))
            total_floors    = st.number_input("Toplam Kat",          min_value=1,   max_value=100,  value=int(imp.get("total_floors", 8)))
            building_age    = st.number_input("Bina Yaşı (yıl)",    min_value=0,   max_value=100,  value=int(imp.get("building_age", 10)))
            is_in_complex   = st.radio("Site İçinde mi?", options=[("Evet", 1), ("Hayır", 0)],
                                       format_func=lambda x: x[0], index=0)
            maintenance_fee = st.number_input("Aidat (₺/ay)",       min_value=0,   max_value=150_000, value=int(imp.get("maintenance_fee", 0)))

        st.markdown('<div class="section-title" style="margin-top:12px;">Niteliksel Özellikler</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        def _sel(label, key):
            opts    = cats.get(key, [])
            default = imp.get(key, opts[0] if opts else "")
            idx     = opts.index(default) if default in opts else 0
            return st.selectbox(label, options=opts, index=idx)

        with col3:
            heating_type       = _sel("Isınma Tipi",     "heating_type")
            furnished          = _sel("Eşya Durumu",     "furnished")
            usage_status       = _sel("Kullanım Durumu", "usage_status")
            orientation        = _sel("Cephe",           "orientation")

        with col4:
            building_type      = _sel("Yapı Tipi",       "building_type")
            building_condition = _sel("Yapı Durumu",     "building_condition")
            floor_category     = _sel("Kat Kategorisi",  "floor_category")

        submitted = st.form_submit_button("💰  Fiyat Tahmin Et", use_container_width=True)

    if submitted:
        from fiyat_tahmin_pipeline import tahmin_et
        ozellikler_t2 = {
            "district": district_t2,         "neighborhood": neighborhood_t2,
            "gross_sqm": gross_sqm,          "net_sqm": net_sqm,
            "total_rooms": total_rooms,       "floor": floor,
            "total_floors": total_floors,     "building_age": building_age,
            "bathroom_count": bathroom_count,
            "is_in_complex": is_in_complex[1],
            "maintenance_fee": maintenance_fee,
            "heating_type": heating_type,     "furnished": furnished,
            "usage_status": usage_status,     "building_type": building_type,
            "building_condition": building_condition,
            "floor_category": floor_category, "orientation": orientation,
        }
        with st.spinner("Model hesaplıyor..."):
            try:
                st.session_state.sonuc      = tahmin_et(ozellikler_t2)
                st.session_state.ozellikler = ozellikler_t2
            except Exception as e:
                st.error(f"Tahmin hatası: {e}")
                return

    if st.session_state.sonuc:
        sonuc      = st.session_state.sonuc
        ozellikler = st.session_state.ozellikler

        st.markdown('<div class="section-title">Tahmin Sonucu</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Alt Sınır (%10)",     sonuc["alt_sinir_fmt"])
        c2.metric("Nokta Tahmin (%50)",  sonuc["tahmin_fmt"])
        c3.metric("Üst Sınır (%90)",     sonuc["ust_sinir_fmt"])

        st.markdown(f"""
        <div class="corp-card" style="margin-top:12px;">
            <div class="corp-card-title">TAHMİN ARALIGI</div>
            <span style="font-size:24px; font-weight:800; color:#C0392B;">{sonuc['aralik']}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Model Performans Bilgisi"):
            e1, e2 = st.columns(2)
            e1.metric("CV-R²",   cfg.get("cv_r2", "—"))
            e2.metric("CV-MAPE", f"%{cfg.get('cv_mape_pct', '—')}")

        if st.button("📊  Fiyat Analizi Göster", use_container_width=True, key="t2_analiz_btn"):
            analiz = fiyat_analizi_uret(ozellikler, sonuc, ilce_ort, mahalle_ort)
            st.markdown(f'<div class="analiz-box">{md2html(analiz)}</div>', unsafe_allow_html=True)
