"""
scoring.py — Çok Boyutlu El Yazısı Kalite Skorlama Modülü

IMU sensör verilerinden elde edilen kullanıcı yazısını referans
verisiyle karşılaştırarak 5 farklı kriter üzerinden 0-10 arası
skor hesaplar. Tüm karşılaştırmalar DTW ve matematiksel metriklerle
yapılır — makine öğrenmesi KULLANILMAZ.

Kriterler ve ağırlıklar:
    1. Şekil benzerliği       → %35
    2. Vuruş yapısı doğruluğu → %25
    3. Akıcılık (jerk)        → %15
    4. Hız tutarlılığı (cv)   → %15
    5. DTW güven skoru        → %10

Bağımlılıklar:
    - numpy, scipy
    - dtw_analysis.py (aynı projede)

Veri formatı: (N, 6+) → [ax, ay, az, gx, gy, gz]
Not: Manyometre (FSR 402) kaldırıldı, basınç kanalı kullanılmamaktadır.
"""

import numpy as np
from scipy.signal import medfilt
from typing import Dict, List, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────

# Kriter ağırlıkları (toplam = 1.0)
AGIRLIKLAR: Dict[str, float] = {
    "shape_similarity": 0.35,
    "stroke_accuracy": 0.25,
    "smoothness": 0.15,
    "speed_consistency": 0.15,
    "confidence": 0.10,
}

# DTW mesafesi → skor dönüşüm eşiği
# Bu eşiğin üstündeki mesafeler skor 0 olarak değerlendirilir.
# Proje referans veritabanına göre kalibre edilebilir.
DTW_MESAFE_ESIK: float = 150.0

# Jerk analizi — bu eşiğin altındaki ortalama jerk "mükemmel akıcılık"
JERK_MUKEMMEL_ESIK: float = 0.5
# Bu eşiğin üstündeki ortalama jerk "sıfır akıcılık"
JERK_KOTU_ESIK: float = 15.0

# Hız tutarlılığı — varyasyon katsayısı eşikleri
CV_MUKEMMEL_ESIK: float = 0.1
CV_KOTU_ESIK: float = 1.5

# Geçme eşiği
GECME_ESIGI: float = 7.0

# Not aralıkları
NOT_ARALIKLARI: List[Tuple[float, float, str]] = [
    (0.0, 3.0, "Zayıf"),
    (3.0, 5.0, "Orta"),
    (5.0, 7.0, "İyi"),
    (7.0, 9.0, "Çok İyi"),
    (9.0, 10.01, "Mükemmel"),
]


# ─────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────

def _klamp(deger: float, alt: float = 0.0, ust: float = 10.0) -> float:
    """Değeri [alt, üst] aralığına sıkıştırır.

    Args:
        deger: Sıkıştırılacak sayı.
        alt: Alt sınır (varsayılan 0.0).
        ust: Üst sınır (varsayılan 10.0).

    Returns:
        Sıkıştırılmış değer.
    """
    return max(alt, min(ust, deger))


def _dogrusal_harita(
    deger: float,
    kaynak_min: float,
    kaynak_max: float,
    hedef_min: float = 0.0,
    hedef_max: float = 10.0,
    ters: bool = False,
) -> float:
    """Bir değeri kaynak aralıktan hedef aralığa doğrusal olarak eşler.

    Args:
        deger: Dönüştürülecek değer.
        kaynak_min: Kaynak aralığın alt sınırı.
        kaynak_max: Kaynak aralığın üst sınırı.
        hedef_min: Hedef aralığın alt sınırı.
        hedef_max: Hedef aralığın üst sınırı.
        ters: True ise yüksek kaynak → düşük hedef (ters orantı).

    Returns:
        Hedef aralığa eşlenmiş ve sıkıştırılmış değer.
    """
    if kaynak_max == kaynak_min:
        return hedef_max if not ters else hedef_min

    # Normalize [0, 1]
    oran = (deger - kaynak_min) / (kaynak_max - kaynak_min)
    oran = max(0.0, min(1.0, oran))

    if ters:
        oran = 1.0 - oran

    sonuc = hedef_min + oran * (hedef_max - hedef_min)
    return _klamp(sonuc, hedef_min, hedef_max)


def _ivme_buyuklugu(veri: np.ndarray) -> np.ndarray:
    """İvme kanallarından (ax, ay, az) büyüklük vektörünü hesaplar.

    Args:
        veri: (N, 3+) şeklinde dizi; ilk 3 sütun ax, ay, az.

    Returns:
        (N,) şeklinde ivme büyüklükleri.
    """
    ax = veri[:, 0]
    ay = veri[:, 1]
    az = veri[:, 2]
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)


def _jiroskop_buyuklugu(veri: np.ndarray) -> np.ndarray:
    """Jiroskop kanallarından (gx, gy, gz) büyüklük vektörünü hesaplar.

    Args:
        veri: (N, 6+) şeklinde dizi; sütun 3-5 gx, gy, gz.

    Returns:
        (N,) şeklinde jiroskop büyüklükleri.
    """
    gx = veri[:, 3]
    gy = veri[:, 4]
    gz = veri[:, 5]
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def _vurus_sure_oranlari(vuruslar: List[Tuple[int, int]]) -> np.ndarray:
    """Vuruş listesinden her vuruşun toplama oranla süre oranını hesaplar.

    Args:
        vuruslar: [(baslangic_idx, bitis_idx), ...] listesi.

    Returns:
        Normalize edilmiş süre oranları dizisi (toplam = 1.0).
    """
    if not vuruslar:
        return np.array([])

    sureler = np.array([float(bitis - baslangic) for baslangic, bitis in vuruslar])
    toplam = sureler.sum()

    if toplam == 0:
        return np.ones(len(sureler)) / len(sureler)

    return sureler / toplam


# ─────────────────────────────────────────────────────────────────────
# Kriter hesaplama fonksiyonları
# ─────────────────────────────────────────────────────────────────────

def _sekil_benzerligi_skoru(
    dtw_mesafe: float,
    esik: float = DTW_MESAFE_ESIK,
) -> float:
    """Kriter 1 — DTW mesafesini 0-10 arası şekil benzerlik skoruna dönüştürür.

    DTW mesafesi 0'a yaklaştıkça skor 10'a yaklaşır.
    Eşik değerini aşan mesafeler 0 skor alır.
    Dönüşüm üstel (exponential) eğriyle yapılır — küçük farklara
    daha hassas, büyük farklara daha toleranslı.

    Args:
        dtw_mesafe: Ham DTW mesafesi (≥ 0).
        esik: Sıfır skor eşiği.

    Returns:
        0.0 – 10.0 arası şekil benzerlik skoru.
    """
    if dtw_mesafe <= 0:
        return 10.0

    if dtw_mesafe >= esik:
        return 0.0

    # Üstel dönüşüm: skor = 10 * exp(-k * mesafe)
    # k, esik noktasında skorun ~0.01 olmasını sağlayacak şekilde ayarlanır.
    # exp(-k * esik) ≈ 0.001 ⇒ k = ln(1000) / esik ≈ 6.9 / esik
    k = 6.9 / esik
    skor = 10.0 * np.exp(-k * dtw_mesafe)

    return round(_klamp(skor), 2)


def _vurus_yapisi_skoru(
    kullanici_vuruslari: List[Tuple[int, int]],
    referans_vuruslari: List[Tuple[int, int]],
) -> float:
    """Kriter 2 — Vuruş sayısı ve süre oranı uyumunu değerlendirir.

    Puanlama:
        - Sayı bileşeni (0-10, ağırlık %60):
            • Tam eşleşme → 10
            • ±1 fark     → 5
            • ±2+ fark    → 0
        - Süre oranı bileşeni (0-10, ağırlık %40):
            • Her vuruşun süre oranı referansla karşılaştırılır.
            • Ortalama mutlak fark 0 → 10, fark ≥ 0.5 → 0.

    Args:
        kullanici_vuruslari: Kullanıcı vuruş aralıkları listesi.
        referans_vuruslari: Referans vuruş aralıkları listesi.

    Returns:
        0.0 – 10.0 arası vuruş yapısı skoru.
    """
    ref_sayi = len(referans_vuruslari)
    kul_sayi = len(kullanici_vuruslari)
    sayi_farki = abs(ref_sayi - kul_sayi)

    # --- Sayı bileşeni ---
    if sayi_farki == 0:
        sayi_skoru = 10.0
    elif sayi_farki == 1:
        sayi_skoru = 5.0
    else:
        sayi_skoru = 0.0

    # --- Süre oranı bileşeni ---
    ref_oranlar = _vurus_sure_oranlari(referans_vuruslari)
    kul_oranlar = _vurus_sure_oranlari(kullanici_vuruslari)

    if ref_oranlar.size == 0 or kul_oranlar.size == 0:
        # Vuruş yoksa sadece sayı bileşenini kullan
        return round(_klamp(sayi_skoru), 2)

    # Eşit uzunluğa getir — kısa olanı sıfırlarla doldur
    maks_uzunluk = max(ref_oranlar.size, kul_oranlar.size)
    ref_padded = np.zeros(maks_uzunluk)
    kul_padded = np.zeros(maks_uzunluk)
    ref_padded[: ref_oranlar.size] = ref_oranlar
    kul_padded[: kul_oranlar.size] = kul_oranlar

    # Ortalama mutlak oran farkı
    ort_fark = np.mean(np.abs(ref_padded - kul_padded))
    sure_skoru = _dogrusal_harita(ort_fark, 0.0, 0.5, 0.0, 10.0, ters=True)

    # Ağırlıklı birleştirme
    toplam = 0.60 * sayi_skoru + 0.40 * sure_skoru
    return round(_klamp(toplam), 2)


def _akicilik_skoru(
    kullanici_veri: np.ndarray,
    dt: float = 0.01,
) -> float:
    """Kriter 3 — Jerk analizi ile yazı akıcılığını değerlendirir.

    Jerk, ivmenin zamana göre türevidir (da/dt). Düşük jerk büyüklüğü
    pürüzsüz, akıcı hareketlere karşılık gelir.

    İşlem:
        1. İvme büyüklüğünü hesapla: ||a|| = sqrt(ax² + ay² + az²)
        2. Jerk = np.diff(||a||) / dt
        3. Ortalama |jerk| → skora dönüştür

    Args:
        kullanici_veri: (N, 3+) kullanıcı IMU verisi.
        dt: Örnekleme periyodu (saniye). Varsayılan 10 ms (100 Hz).

    Returns:
        0.0 – 10.0 arası akıcılık skoru.
    """
    if kullanici_veri.shape[0] < 3:
        # Çok kısa sinyal — değerlendirme yapılamaz, nötr skor
        return 5.0

    # İvme büyüklüğü
    ivme_mag = _ivme_buyuklugu(kullanici_veri)

    # Medyan filtre (gürültü azaltma) — çekirdek boyutu tek olmalı
    cekirdek = min(5, kullanici_veri.shape[0])
    if cekirdek % 2 == 0:
        cekirdek -= 1
    if cekirdek >= 3:
        ivme_mag = medfilt(ivme_mag, kernel_size=cekirdek)

    # Jerk hesapla
    jerk = np.diff(ivme_mag) / dt
    jerk_buyukluk = np.abs(jerk)
    ort_jerk = float(np.mean(jerk_buyukluk))

    # Skora dönüştür (düşük jerk → yüksek skor)
    skor = _dogrusal_harita(
        ort_jerk,
        JERK_MUKEMMEL_ESIK,
        JERK_KOTU_ESIK,
        0.0,
        10.0,
        ters=True,
    )
    return round(_klamp(skor), 2)


def _hiz_tutarliligi_skoru(
    kullanici_veri: np.ndarray,
) -> float:
    """Kriter 4 — İvme büyüklüğünün varyasyon katsayısı ile hız tutarlılığı.

    Varyasyon katsayısı (cv) = std(||a||) / mean(||a||)
    Düşük cv → tutarlı hız → yüksek skor.

    Args:
        kullanici_veri: (N, 3+) kullanıcı IMU verisi.

    Returns:
        0.0 – 10.0 arası hız tutarlılığı skoru.
    """
    if kullanici_veri.shape[0] < 2:
        return 5.0

    ivme_mag = _ivme_buyuklugu(kullanici_veri)
    ortalama = float(np.mean(ivme_mag))

    if ortalama < 1e-9:
        # Neredeyse hiç hareket yok — skor nötr
        return 5.0

    std = float(np.std(ivme_mag))
    cv = std / ortalama

    skor = _dogrusal_harita(
        cv,
        CV_MUKEMMEL_ESIK,
        CV_KOTU_ESIK,
        0.0,
        10.0,
        ters=True,
    )
    return round(_klamp(skor), 2)


def _guven_skoru(
    tanimlama_guveni: float,
) -> float:
    """Kriter 5 — DTW tabanlı harf tanımlama güven değerini 0-10'a eşler.

    identify_letter fonksiyonundan gelen güven değeri genellikle
    0.0 – 1.0 arasıdır.

    Args:
        tanimlama_guveni: 0.0 – 1.0 arası güven değeri.

    Returns:
        0.0 – 10.0 arası güven skoru.
    """
    skor = _klamp(tanimlama_guveni, 0.0, 1.0) * 10.0
    return round(skor, 2)


# ─────────────────────────────────────────────────────────────────────
# Not / derece belirleme
# ─────────────────────────────────────────────────────────────────────

def _derece_belirle(skor: float) -> str:
    """Toplam skora göre metin notu döndürür.

    Args:
        skor: 0.0 – 10.0 arası toplam skor.

    Returns:
        "Zayıf", "Orta", "İyi", "Çok İyi" veya "Mükemmel"
    """
    for alt, ust, etiket in NOT_ARALIKLARI:
        if alt <= skor < ust:
            return etiket
    return "Mükemmel"  # 10.0 tam skor


# ─────────────────────────────────────────────────────────────────────
# Geri bildirim üretimi (Türkçe)
# ─────────────────────────────────────────────────────────────────────

def _geri_bildirim_uret(detaylar: Dict[str, float]) -> List[str]:
    """Kriter skorlarına göre Türkçe geri bildirim cümleleri üretir.

    Her kriter için skor aralığına uygun ton kullanılır:
        0-3  : teşvik edici ama düzeltici
        4-6  : olumlu ama geliştirici
        7-8  : övgü + ince ayar önerileri
        9-10 : kutlama

    Args:
        detaylar: Kriter adı → skor eşlemesi.

    Returns:
        Geri bildirim cümlelerinin listesi.
    """
    geri_bildirimler: List[str] = []

    # ── Şekil benzerliği ──
    s = detaylar["shape_similarity"]
    if s >= 9.0:
        geri_bildirimler.append("Harfin şekli neredeyse mükemmel! Harika bir çizim. 🎉")
    elif s >= 7.0:
        geri_bildirimler.append(
            "Harfin genel şekli çok iyi! "
            "Köşeleri biraz daha belirgin yaparak daha da geliştirebilirsin."
        )
    elif s >= 4.0:
        geri_bildirimler.append(
            "Harfin genel hatları doğru yolda. "
            "Oranları referansa daha yakın tutmaya çalış."
        )
    else:
        geri_bildirimler.append(
            "Harfin şekli referanstan biraz uzak görünüyor. "
            "Referans harfi inceleyip yeniden denemeye ne dersin?"
        )

    # ── Vuruş yapısı ──
    s = detaylar["stroke_accuracy"]
    if s >= 9.0:
        geri_bildirimler.append("Vuruş sayın ve yapın tam doğru! 👏")
    elif s >= 7.0:
        geri_bildirimler.append(
            "Vuruş yapısı oldukça iyi. "
            "Her vuruşun süresini referansla eşleştirmeye dikkat et."
        )
    elif s >= 4.0:
        geri_bildirimler.append(
            "Vuruş sayısı ya da sürelerinde ufak farklılıklar var. "
            "Kalemi kaldırma noktalarına dikkat etmeyi dene."
        )
    else:
        geri_bildirimler.append(
            "Vuruş sayısı referanstan farklı. "
            "Bu harfin kaç vuruşla yazıldığını tekrar gözden geçir."
        )

    # ── Akıcılık ──
    s = detaylar["smoothness"]
    if s >= 9.0:
        geri_bildirimler.append("Yazın çok akıcı, el hareketlerin pürüzsüz! ✨")
    elif s >= 7.0:
        geri_bildirimler.append(
            "Güzel bir akıcılık var. "
            "Ani hızlanma ve yavaşlamaları biraz daha azaltabilirsin."
        )
    elif s >= 4.0:
        geri_bildirimler.append(
            "Hareketlerinde biraz sarsıntı var. "
            "Daha sakin ve sürekli bir hareketle yazmayı dene."
        )
    else:
        geri_bildirimler.append(
            "Yazı sırasında çok fazla titreme/sarsıntı algılandı. "
            "Kolunu sabitlemeye çalış ve kalemi rahat tut."
        )

    # ── Hız tutarlılığı ──
    s = detaylar["speed_consistency"]
    if s >= 9.0:
        geri_bildirimler.append("Hızın baştan sona çok tutarlı — harika kontrol! 🏆")
    elif s >= 7.0:
        geri_bildirimler.append(
            "Hız tutarlılığın iyi. "
            "Harfin başı ve sonu arasında küçük hız farkları var, onları dengelemeye çalış."
        )
    elif s >= 4.0:
        geri_bildirimler.append(
            "Yazı hızında dalgalanmalar var. "
            "Eşit tempoda yazmaya odaklanmayı dene."
        )
    else:
        geri_bildirimler.append(
            "Hızın çok değişken — bazen çok hızlı, bazen çok yavaş. "
            "Sabit bir ritim tutturmaya çalış."
        )

    # ── Güven skoru ──
    s = detaylar["confidence"]
    if s >= 9.0:
        geri_bildirimler.append("Yazdığın harf çok net tanınıyor! 🔍")
    elif s >= 7.0:
        geri_bildirimler.append("Harf tanıma güveni yüksek, doğru harfi yazıyorsun.")
    elif s >= 4.0:
        geri_bildirimler.append(
            "Harf tanınabiliyor ama başka harflerle karışabilir. "
            "Ayırt edici detaylara dikkat et."
        )
    else:
        geri_bildirimler.append(
            "Harf tanıma güveni düşük — harf başka harflerle karıştırılıyor. "
            "Harfin karakteristik özelliklerini vurgulayarak tekrar dene."
        )

    return geri_bildirimler


def _genel_degerlendirme(toplam_skor: float, derece: str) -> str:
    """Toplam skora göre 1-2 cümlelik genel değerlendirme üretir.

    Args:
        toplam_skor: 0-10 arası toplam skor.
        derece: Harf notu (Zayıf/Orta/İyi/Çok İyi/Mükemmel).

    Returns:
        Genel değerlendirme cümlesi.
    """
    if toplam_skor >= 9.0:
        return (
            f"Tebrikler! 🎉 Skorun {toplam_skor:.1f}/10 ({derece}). "
            "Olağanüstü bir yazım performansı sergiliyorsun!"
        )
    elif toplam_skor >= 7.0:
        return (
            f"Güzel iş! Skorun {toplam_skor:.1f}/10 ({derece}). "
            "Birkaç ince detayı geliştirerek mükemmele ulaşabilirsin."
        )
    elif toplam_skor >= 5.0:
        return (
            f"İyi gidiyorsun! Skorun {toplam_skor:.1f}/10 ({derece}). "
            "Biraz daha pratikle çok daha iyi olacak."
        )
    elif toplam_skor >= 3.0:
        return (
            f"Skorun {toplam_skor:.1f}/10 ({derece}). "
            "Gelişim alanların var ama endişelenme — pratik yapmaya devam et!"
        )
    else:
        return (
            f"Skorun {toplam_skor:.1f}/10 ({derece}). "
            "Referans harfi dikkatlice incele ve tekrar dene. Her denemede daha iyi olacaksın! 💪"
        )


# ─────────────────────────────────────────────────────────────────────
# Ana skorlama fonksiyonu
# ─────────────────────────────────────────────────────────────────────

def calculate_quality_score(
    user_data: np.ndarray,
    ref_data: np.ndarray,
    user_strokes: List[Tuple[int, int]],
    ref_strokes: List[Tuple[int, int]],
    dtw_distance: float,
    identification_confidence: float,
    *,
    dt: float = 0.01,
    dtw_esik: float = DTW_MESAFE_ESIK,
) -> Dict[str, Any]:
    """Kullanıcının yazım kalitesini çok boyutlu olarak değerlendirir.

    5 kriteri bağımsız olarak hesaplar, ağırlıklı ortalamayla toplam
    skor üretir ve Türkçe geri bildirim döndürür.

    Bu fonksiyon dtw_analysis.py modülündeki DTW mesafesi ve vuruş
    tespiti sonuçlarını girdi olarak alır [1].

    Args:
        user_data:                Kullanıcı ham IMU verisi, shape (M, 6+).
        ref_data:                 Referans IMU verisi, shape (N, 6+).
        user_strokes:             Kullanıcı vuruş aralıkları [(start, end), ...].
        ref_strokes:              Referans vuruş aralıkları [(start, end), ...].
        dtw_distance:             dtw_analysis.py'den gelen ham DTW mesafesi.
        identification_confidence: identify_letter'dan gelen güven (0.0-1.0).
        dt:                       Örnekleme periyodu (saniye). Varsayılan 10 ms.
        dtw_esik:                 DTW sıfır-skor eşiği. Kalibre edilebilir.

    Returns:
        Dict yapısı:
        {
            "total_score": float,       # 0-10 arası ağırlıklı toplam
            "details": {
                "shape_similarity": float,
                "stroke_accuracy": float,
                "smoothness": float,
                "speed_consistency": float,
                "confidence": float,
            },
            "feedback": List[str],      # Kriter bazlı + genel değerlendirme
            "pass": bool,               # total_score >= 7.0
            "grade": str,               # Zayıf / Orta / İyi / Çok İyi / Mükemmel
        }

    Raises:
        ValueError: Veri boyutu yetersizse (en az 3 sütun gerekli).
    """
    # ── Girdi doğrulama ──
    if user_data.ndim != 2 or user_data.shape[1] < 3:
        raise ValueError(
            f"user_data en az (N, 3) boyutunda olmalı; alınan şekil: {user_data.shape}"
        )
    if ref_data.ndim != 2 or ref_data.shape[1] < 3:
        raise ValueError(
            f"ref_data en az (N, 3) boyutunda olmalı; alınan şekil: {ref_data.shape}"
        )

    # ── Kriter skorlarını hesapla ──
    sekil_skor = _sekil_benzerligi_skoru(dtw_distance, esik=dtw_esik)
    vurus_skor = _vurus_yapisi_skoru(user_strokes, ref_strokes)
    akicilik_skor = _akicilik_skoru(user_data, dt=dt)
    hiz_skor = _hiz_tutarliligi_skoru(user_data)
    guven_skor = _guven_skoru(identification_confidence)

    detaylar: Dict[str, float] = {
        "shape_similarity": sekil_skor,
        "stroke_accuracy": vurus_skor,
        "smoothness": akicilik_skor,
        "speed_consistency": hiz_skor,
        "confidence": guven_skor,
    }

    # ── Ağırlıklı toplam skor ──
    toplam_skor = sum(
        detaylar[kriter] * agirlik
        for kriter, agirlik in AGIRLIKLAR.items()
    )
    toplam_skor = round(_klamp(toplam_skor), 2)

    # ── Derece ──
    derece = _derece_belirle(toplam_skor)

    # ── Geri bildirim ──
    geri_bildirim_listesi = _geri_bildirim_uret(detaylar)
    genel = _genel_degerlendirme(toplam_skor, derece)
    geri_bildirim_listesi.append(genel)

    # ── Geçti mi? ──
    gecti = toplam_skor >= GECME_ESIGI

    return {
        "total_score": toplam_skor,
        "details": detaylar,
        "feedback": geri_bildirim_listesi,
        "pass": gecti,
        "grade": derece,
    }


# ─────────────────────────────────────────────────────────────────────
# Kolaylık fonksiyonu — sadece veri ve mesafeyle hızlı skorlama
# ─────────────────────────────────────────────────────────────────────

def quick_score(
    user_data: np.ndarray,
    ref_data: np.ndarray,
    dtw_distance: float,
    identification_confidence: float = 0.5,
    # pressure_threshold: float = 0.3,  # manyometre kaldırıldı
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Vuruş bilgisi olmadan hızlı skorlama yapar.

    Not: Manyometre kaldırıldığı için tüm veri tek vuruş olarak kabul edilir.
    # ESKİ: Basınç kanalı (7. sütun) varsa otomatik vuruş tespiti yapıyordu.

    Args:
        user_data:                Kullanıcı IMU verisi, shape (M, 6+).
        ref_data:                 Referans IMU verisi, shape (N, 6+).
        dtw_distance:             DTW mesafesi.
        identification_confidence: Güven değeri (0-1).
        # pressure_threshold:       Basınç eşiği — manyometre kaldırıldı.
        dt:                       Örnekleme periyodu.

    Returns:
        calculate_quality_score ile aynı dict yapısı.
    """

    def _basit_vurus_tespit(
        veri: np.ndarray, esik: float
    ) -> List[Tuple[int, int]]:
        """Tüm veriyi tek vuruş olarak döndürür.

        Not: Manyometre kaldırıldığı için basınç tabanlı tespit devre dışı.
        # ESKİ: Basınç > eşik olan ardışık bölgeleri vuruş olarak algılıyordu.

        Args:
            veri: (N, 6+) IMU verisi.
            esik: Basınç eşik değeri (kullanılmıyor).

        Returns:
            Vuruş aralıkları listesi (tek vuruş).
        """
        # Manyometre kaldırıldı — tüm veri tek vuruş
        return [(0, veri.shape[0] - 1)]

        # --- ESKİ KOD: Basınç tabanlı vuruş tespiti ---
        # if veri.shape[1] < 7:
        #     return [(0, veri.shape[0] - 1)]
        # basinc = veri[:, 6]
        # aktif = basinc > esik
        # vuruslar: List[Tuple[int, int]] = []
        # baslangic: Optional[int] = None
        # for i, durum in enumerate(aktif):
        #     if durum and baslangic is None:
        #         baslangic = i
        #     elif not durum and baslangic is not None:
        #         vuruslar.append((baslangic, i - 1))
        #         baslangic = None
        # if baslangic is not None:
        #     vuruslar.append((baslangic, len(aktif) - 1))
        # if not vuruslar:
        #     vuruslar = [(0, veri.shape[0] - 1)]
        # return vuruslar

    user_strokes = _basit_vurus_tespit(user_data, 0)  # esik kullanılmıyor
    ref_strokes = _basit_vurus_tespit(ref_data, 0)    # esik kullanılmıyor
    # ESKİ: _basit_vurus_tespit(user_data, pressure_threshold)

    return calculate_quality_score(
        user_data=user_data,
        ref_data=ref_data,
        user_strokes=user_strokes,
        ref_strokes=ref_strokes,
        dtw_distance=dtw_distance,
        identification_confidence=identification_confidence,
        dt=dt,
    )


# ─────────────────────────────────────────────────────────────────────
# Modül testi (doğrudan çalıştırılırsa)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sentetik test verisi oluştur (manyometre kaldırıldı — 6 kanal)
    np.random.seed(42)
    N = 200  # örnek sayısı

    # Referans: düzgün sinüzoidal hareket
    t = np.linspace(0, 2 * np.pi, N)
    ref = np.column_stack([
        np.sin(t),              # ax
        np.cos(t),              # ay
        np.zeros(N),            # az
        np.sin(t) * 0.5,       # gx
        np.cos(t) * 0.5,       # gy
        np.zeros(N),            # gz
        # np.where(t < 4.5, 1.0, 0.0),  # pressure — MANYOMETRE KALDIRILDI
    ])

    # Kullanıcı: hafif gürültülü referans kopyası
    kullanici = ref.copy() + np.random.normal(0, 0.05, ref.shape)

    # Örnek DTW mesafesi ve güven
    ornek_dtw_mesafe = 12.5
    ornek_guven = 0.85

    sonuc = quick_score(
        user_data=kullanici,
        ref_data=ref,
        dtw_distance=ornek_dtw_mesafe,
        identification_confidence=ornek_guven,
    )

    print("=" * 60)
    print("  EL YAZISI KALİTE DEĞERLENDİRME SONUCU")
    print("=" * 60)
    print(f"  Toplam Skor : {sonuc['total_score']:.1f} / 10")
    print(f"  Derece      : {sonuc['grade']}")
    print(f"  Geçti mi?   : {'✅ Evet' if sonuc['pass'] else '❌ Hayır'}")
    print("-" * 60)
    print("  Kriter Detayları:")
    for kriter, skor in sonuc["details"].items():
        agirlik = AGIRLIKLAR[kriter]
        print(f"    {kriter:<22s} : {skor:>5.1f}  (ağırlık: %{agirlik*100:.0f})")
    print("-" * 60)
    print("  Geri Bildirim:")
    for i, fb in enumerate(sonuc["feedback"], 1):
        print(f"    {i}. {fb}")
    print("=" * 60)