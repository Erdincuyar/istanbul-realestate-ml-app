import re
import streamlit as st
import pandas as pd
import plotly.express as px


def render(df_ultimate):
    if df_ultimate.empty:
        st.warning("istanbul_emlak_ULTIMATE_2026.csv bulunamadı.")
        return

    st.markdown('<p style="color:#AAAAAA; font-size:15px; margin-bottom:20px;">ULTIMATE veri setiyle mahalle bazlı piyasa karnesi, kira getirisi ve amortisman analizi.</p>', unsafe_allow_html=True)

    df = df_ultimate.reset_index()

    t3c1, t3c2 = st.columns(2)
    with t3c1:
        ilce_t3 = st.selectbox("📍 İlçe", sorted(df["district"].dropna().unique()), key="t3_ilce")
    with t3c2:
        mah_opts = sorted(df[df["district"] == ilce_t3]["neighborhood"].dropna().unique())
        mahalle_t3 = st.selectbox("🏘️ Mahalle", mah_opts, key="t3_mah")

    bolge = df[(df["district"] == ilce_t3) & (df["neighborhood"] == mahalle_t3)].copy()

    if bolge.empty:
        st.warning("Bu bölgede ULTIMATE verisinde ilan bulunamadı.")
        return

    # ── Özet metrikler ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Bölge Özeti</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam İlan", f"{len(bolge):,}")
    ort_m2 = bolge["mahalle_ort_m2_fiyat"].dropna().mean()
    m2.metric("Ort. m² Fiyatı", f"{ort_m2:,.0f} ₺" if pd.notna(ort_m2) else "—")
    ort_kira = bolge["tahmini_kira"].dropna().mean()
    m3.metric("Ort. Tahmini Kira", f"{ort_kira:,.0f} ₺" if pd.notna(ort_kira) else "—")
    favori_oda = bolge["mahalle_favori_oda"].dropna().mode()
    m4.metric("Tercih Edilen Oda", f"{int(favori_oda.iloc[0])}+1" if len(favori_oda) else "—")

    # ── Kira dağılım grafiği ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Kira & Amortisman Dağılımı</div>', unsafe_allow_html=True)
    kira_data = bolge["tahmini_kira"].dropna()
    if not kira_data.empty:
        fig = px.histogram(
            kira_data, nbins=20,
            labels={"value": "Tahmini Kira (₺)", "count": "İlan Sayısı"},
            title=f"{mahalle_t3} — Tahmini Kira Dağılımı",
            color_discrete_sequence=["#C0392B"],
        )
        fig.update_layout(
            paper_bgcolor="#1E1E1E", plot_bgcolor="#2A2A2A",
            font_family="Inter", showlegend=False,
            font_color="#EEEEEE", title_font_color="#EEEEEE",
            margin=dict(t=40, b=20, l=10, r=10), height=300,
            xaxis=dict(gridcolor="#3A3A3A", linecolor="#3A3A3A", tickfont=dict(color="#EEEEEE")),
            yaxis=dict(gridcolor="#3A3A3A", linecolor="#3A3A3A", tickfont=dict(color="#EEEEEE")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Amortisman özeti ────────────────────────────────────────────────────
    amor_data = bolge["amortisman_yili"].dropna()
    if not amor_data.empty:
        def parse_amor(s):
            m = re.match(r"(\d+)\s*Yıl", str(s))
            return int(m.group(1)) if m else None
        amor_nums = amor_data.map(parse_amor).dropna()
        if not amor_nums.empty:
            ac, _, _, _ = st.columns(4)
            ac.metric("Ort. Amortisman", f"{int(round(amor_nums.mean()))} yıl")

    # ── İlan hikayeleri + karne ─────────────────────────────────────────────
    st.markdown('<div class="section-title">İlan Hikayeleri</div>', unsafe_allow_html=True)
    for _, row in bolge[bolge["ilan_hikayesi"].notna()].head(5).iterrows():
        fiyat_str = f"{row['price']:,.0f} ₺" if pd.notna(row.get("price")) else "—"
        kira_str  = f"{row['tahmini_kira']:,.0f} ₺/ay" if pd.notna(row.get("tahmini_kira")) else "—"
        with st.expander(f"İlan {row.name}  |  {fiyat_str}  |  Kira: {kira_str}"):
            st.markdown(f'<div class="hikaye-box">{row["ilan_hikayesi"]}</div>', unsafe_allow_html=True)
            karne = row.get("mahalle_karnesi")
            if pd.notna(karne) and karne:
                karne_txt = str(karne).replace("📊 BÖLGE KARNESİ:", "").strip()
                st.markdown(f'<div class="karne-box" style="margin-top:8px;">{karne_txt}</div>', unsafe_allow_html=True)
            amor = row.get("amortisman_yili")
            if pd.notna(amor):
                st.caption(f"Amortisman: {amor}")
