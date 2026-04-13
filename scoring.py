"""
scoring.py — Çok Boyutlu Yazı Kalitesi Skorlama Modülü

DTW tabanlı karşılaştırma ile harf yazım kalitesini değerlendirir.
ML kullanılmıyor — tüm metrikler matematiksel hesaplamalarla elde edilir.
Manyometre kaldırıldı, buton tabanlı kayıt sistemi kullanılıyor.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# ── Ağırlıklar ──
AGIRLIKLAR = {
    "shape_similarity": 0.35,
    "stroke_accuracy": 0.25,
    "smoothness": 0.15,
    "speed_consistency": 0.15,
    "confidence": 0.10,
}

# ── Eşik Değerleri ──
JERK_MUKEMMEL = 50.0
JERK_KOTU = 500.0
CV_MUKEMMEL_ESIK = 0.3
CV_KOTU_ESIK = 2.0
GECME_ESIGI = 7.0

# ── Not Aralıkları ──
NOT_ARALIKLARI = [
    (9.0, 10.0, "Mükemmel"),
    (7.0, 9.0, "Çok İyi"),
    (5.0, 7.0, "İyi"),
    (3.0, 5.0, "Orta"),
    (0.0, 3.0, "Zayıf"),
]


def _klamp(deger: float, alt: float = 0.0, ust: float = 10.0) -> float:
    """Değeri belirtilen aralığa sıkıştırır."""
    return max(alt, min(ust, deger))


def _dogrusal_harita(
    deger: float, kaynak_min: float, kaynak_max: float,
    hedef_min: float, hedef_max: float, ters: bool = False,
) -> float:
    """Doğrusal interpolasyon ile değer eşleme."""
    if abs(kaynak_max - kaynak_min) < 1e-10:
        return (hedef_min + hedef_max) / 2

    oran = (deger - kaynak_min) / (kaynak_max - kaynak_min)
    oran = max(0.0, min(1.0, oran))

    if ters:
        oran = 1.0 - oran

    return hedef_min + oran * (hedef_max - hedef_min)


def _sekil_benzerligi(dtw_mesafe: float) -> float:
    """Kriter 1 — DTW mesafesini 0-10 skora dönüştürür."""
    from dtw_analysis import PERFECT_DISTANCE, TERRIBLE_DISTANCE
    skor = _dogrusal_harita(
        dtw_mesafe, PERFECT_DISTANCE, TERRIBLE_DISTANCE,
        10.0, 0.0, ters=False
    )
    return round(_klamp(skor), 2)


def _akicilik_skoru(user_data: np.ndarray, dt: float = 0.01) -> float:
    """Kriter 3 — Jerk analizi ile akıcılık skoru."""
    if user_data.shape[0] < 4:
        return 5.0

    imu = user_data[:, :6] if user_data.shape[1] > 6 else user_data
    ivme_mag = np.sqrt(np.sum(imu[:, :3] ** 2, axis=1))
    jerk = np.diff(ivme_mag) / dt
    ortalama_jerk = float(np.mean(np.abs(jerk)))

    skor = _dogrusal_harita(
        ortalama_jerk, JERK_MUKEMMEL, JERK_KOTU, 10.0, 0.0, ters=False
    )
    return round(_klamp(skor), 2)


def _hiz_tutarliligi(user_data: np.ndarray) -> float:
    """Kriter 4 — Hız tutarlılığı (varyasyon katsayısı)."""
    if user_data.shape[0] < 3:
        return 5.0

    imu = user_data[:, :6] if user_data.shape[1] > 6 else user_data
    ivme_mag = np.sqrt(np.sum(imu[:, :3] ** 2, axis=1))
    ortalama = float(np.mean(ivme_mag))

    if ortalama < 1e-8:
        return 5.0

    std = float(np.std(ivme_mag))
    cv = std / ortalama

    skor = _dogrusal_harita(cv, CV_MUKEMMEL_ESIK, CV_KOTU_ESIK, 10.0, 0.0, ters=False)
    return round(_klamp(skor), 2)


def _guven_skoru(tanimlama_guveni: float) -> float:
    """Kriter 5 — DTW güven değerini 0-10'a eşler."""
    skor = _klamp(tanimlama_guveni, 0.0, 1.0) * 10.0
    return round(skor, 2)


def _not_belirle(skor: float) -> str:
    """Skora göre not belirler."""
    for alt, ust, not_adi in NOT_ARALIKLARI:
        if alt <= skor <= ust:
            return not_adi
    return "Zayıf"


def _geri_bildirim_uret(detaylar: Dict[str, float], toplam: float) -> List[str]:
    """Skor detaylarına göre Türkçe geri bildirim üretir."""
    mesajlar = []

    if toplam >= 9.0:
        mesajlar.append("🎉 Mükemmel! Harfi çok güzel yazdın!")
    elif toplam >= 7.0:
        mesajlar.append("👍 Çok iyi! Harika gidiyorsun!")
    elif toplam >= 5.0:
        mesajlar.append("💪 İyi! Biraz daha pratikle daha iyi olacak.")
    elif toplam >= 3.0:
        mesajlar.append("📝 Fena değil, ama geliştirilmesi gereken noktalar var.")
    else:
        mesajlar.append("🔄 Tekrar dene! Yavaş ve dikkatli yazmayı dene.")

    if detaylar.get("shape_similarity", 10) < 5:
        mesajlar.append("Harfin genel şekli referanstan farklı. Yavaşça tekrar dene.")

    if detaylar.get("smoothness", 10) < 5:
        mesajlar.append("Kalem hareketlerin kesik kesik. Daha akıcı çizmeyi dene.")

    if detaylar.get("speed_consistency", 10) < 5:
        mesajlar.append("Yazma hızın tutarsız. Sabit hızda yazmaya çalış.")

    if detaylar.get("confidence", 10) < 4:
        mesajlar.append("Harf yeterince net tanınamadı. Daha belirgin hareketler yap.")

    return mesajlar


def calculate_quality_score(
    user_data: np.ndarray,
    ref_data: np.ndarray,
    dtw_distance: float,
    identification_confidence: float = 0.5,
) -> Dict[str, Any]:
    """Kapsamlı yazı kalitesi skorlaması yapar.

    Args:
        user_data: Kullanıcı ham verisi, shape (N, 6).
        ref_data: Referans ham verisi, shape (M, 6).
        dtw_distance: DTW mesafesi (dtw_analysis'ten).
        identification_confidence: Harf tanımlama güveni (0-1).

    Returns:
        dict: Skor detayları.
    """
    # Her kriteri hesapla
    shape_score = _sekil_benzerligi(dtw_distance)
    stroke_score = 5.0  # Vuruş analizi manyometre olmadan temel seviyede
    smooth_score = _akicilik_skoru(user_data)
    speed_score = _hiz_tutarliligi(user_data)
    conf_score = _guven_skoru(identification_confidence)

    detaylar = {
        "shape_similarity": shape_score,
        "stroke_accuracy": stroke_score,
        "smoothness": smooth_score,
        "speed_consistency": speed_score,
        "confidence": conf_score,
    }

    # Ağırlıklı toplam
    toplam = sum(detaylar[k] * AGIRLIKLAR[k] for k in AGIRLIKLAR)
    toplam = round(_klamp(toplam), 1)

    return {
        "total_score": toplam,
        "details": detaylar,
        "feedback": _geri_bildirim_uret(detaylar, toplam),
        "pass": toplam >= GECME_ESIGI,
        "grade": _not_belirle(toplam),
    }