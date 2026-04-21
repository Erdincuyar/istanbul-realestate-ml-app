def kredi_hesapla(fiyat, pesinat_orani=0.20, yillik_faiz=0.45, vade=120):
    """
    Selen'in Finansal Analiz Modülü:
    Girilen fiyat ve parametrelere göre peşinat ve aylık taksiti hesaplar.
    """
    gerekli_pesinat = fiyat * pesinat_orani
    kredi_miktari = fiyat - gerekli_pesinat
    aylik_faiz = yillik_faiz / 12

    if aylik_faiz > 0:
        taksit = (kredi_miktari * aylik_faiz * (1 + aylik_faiz)**vade) / ((1 + aylik_faiz)**vade - 1)
    else:
        taksit = kredi_miktari / vade

    return {
        "pesinat": int(gerekli_pesinat),
        "aylik_taksit": int(taksit)
    }
