"""
dtw_analysis.py — DTW Tabanlı Harf Tanıma ve Sinyal Karşılaştırma Modülü

IMU sensör verilerini Dynamic Time Warping (DTW) algoritmasıyla
karşılaştırarak harf tanıma, skor hesaplama ve 2D yörünge
çıkarma işlemlerini yapar.

Referans veritabanı: letters_db/<HARF>/ref_01.npy, ref_02.npy, ...
Not: Manyometre (FSR 402) kaldırıldı.
Veri formatı: (N, 6) → [ax, ay, az, gx, gy, gz]
"""

import os
import glob
import numpy as np
from fastdtw import fastdtw
from scipy.signal import medfilt, detrend
from scipy.integrate import cumulative_trapezoid
from typing import Optional, Dict, List, Tuple

# ==========================================
# SABİTLER
# ==========================================

# Proje kök dizini ve referans veritabanı yolu
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
LETTERS_DB_DIR: str = os.path.join(BASE_DIR, "letters_db")

# DTW parametreleri
DTW_RADIUS: int = 10  # fastdtw arama yarıçapı (Sakoe-Chiba bandı)

# Eksen ağırlıkları — çok eksenli DTW birleştirme
AXIS_WEIGHTS: Dict[str, float] = {
    'ax': 0.2,
    'ay': 0.2,
    'az': 0.1,
    'gx': 0.2,
    'gy': 0.2,
    'gz': 0.1,
}

# Skor kalibrasyonu — DTW mesafesini 0-100 arası skora çevirme
PERFECT_DISTANCE: float = 1200.0   # Bu mesafe = 100 puan
TERRIBLE_DISTANCE: float = 2400.0  # Bu mesafe = 0 puan

# Giyotin eşikleri — bunların altında direkt 0 puan
MIN_ENERGY_RATIO: float = 0.60       # Minimum enerji oranı
MIN_COMPLEXITY_RATIO: float = 0.35   # Minimum karmaşıklık oranı

# Vuruş karşılaştırma
STROKE_COUNT_PENALTY: float = 0.15   # Vuruş sayısı farkı başına ceza (0-1)
# PRESSURE_THRESHOLD: int = 50         # Vuruş algılama basınç eşiği — manyometre kaldırıldı

# Yörünge hesaplama
ZUPT_THRESHOLD: float = 500.0        # Zero Velocity Update eşiği (ham birim)
SAMPLE_RATE: int = 100                # Varsayılan örnekleme hızı (Hz)

# Eksen isimleri
AXIS_NAMES: List[str] = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']


# ==========================================
# 1. REFERANS VERİ YÖNETİMİ
# ==========================================

def load_references(db_dir: str = LETTERS_DB_DIR) -> Dict[str, List[np.ndarray]]:
    """Tüm referans harfleri veritabanından yükler.

    letters_db/ altındaki her alt klasördeki .npy dosyalarını okur.
    Eski düz dosya formatını (letter_A.npy) da destekler.

    Args:
        db_dir: Veritabanı dizini. Varsayılan: letters_db/

    Returns:
        Dict[str, List[np.ndarray]]: Harf → referans listesi.
            Örnek: {'A': [array1, array2], 'B': [array1]}
    """
    references: Dict[str, List[np.ndarray]] = {}

    if not os.path.exists(db_dir):
        print(f"[Referans] UYARI: {db_dir} dizini bulunamadı.")
        os.makedirs(db_dir, exist_ok=True)
        return references

    # --- Yeni format: letters_db/A/ref_01.npy ---
    for entry in sorted(os.listdir(db_dir)):
        entry_path = os.path.join(db_dir, entry)

        if os.path.isdir(entry_path):
            harf = entry.upper()
            dosyalar = sorted(glob.glob(os.path.join(entry_path, "*.npy")))

            if dosyalar:
                references[harf] = []
                for dosya in dosyalar:
                    try:
                        veri = np.load(dosya)
                        references[harf].append(veri)
                    except Exception as e:
                        print(f"[Referans] UYARI: {dosya} yüklenemedi: {e}")

                print(f"[Referans] '{harf}' → {len(references[harf])} referans")

    # --- Eski format uyumluluğu: letters_db/letter_A.npy ---
    eski_dosyalar = glob.glob(os.path.join(db_dir, "letter_*.npy"))
    for dosya in sorted(eski_dosyalar):
        dosya_adi = os.path.basename(dosya)
        # "letter_A.npy" → "A"
        harf = dosya_adi.replace("letter_", "").replace(".npy", "").upper()

        if harf not in references:
            try:
                veri = np.load(dosya)
                references[harf] = [veri]
                print(f"[Referans] '{harf}' → 1 referans (eski format)")
            except Exception as e:
                print(f"[Referans] UYARI: {dosya} yüklenemedi: {e}")

    print(f"[Referans] Toplam: {len(references)} harf, "
          f"{sum(len(v) for v in references.values())} referans")

    return references


def save_reference(
    letter: str,
    data: np.ndarray,
    db_dir: str = LETTERS_DB_DIR
) -> str:
    """Yeni referans verisini veritabanına kaydeder.

    Otomatik numaralandırma yapar: ref_01.npy, ref_02.npy, ...
    Klasör yoksa oluşturur.

    Args:
        letter: Harf adı (örn: 'A', 'B').
        data: Referans verisi, shape (N, 7) veya (N, 6).
        db_dir: Veritabanı dizini. Varsayılan: letters_db/

    Returns:
        str: Kaydedilen dosyanın tam yolu.
    """
    harf = letter.upper()
    harf_dir = os.path.join(db_dir, harf)
    os.makedirs(harf_dir, exist_ok=True)

    # Otomatik numaralandır — mevcut dosyaları say
    mevcut = glob.glob(os.path.join(harf_dir, "ref_*.npy"))
    sonraki_no = len(mevcut) + 1
    dosya_adi = f"ref_{sonraki_no:02d}.npy"
    tam_yol = os.path.join(harf_dir, dosya_adi)

    np.save(tam_yol, data)
    print(f"[Referans] '{harf}' → {dosya_adi} kaydedildi ({data.shape})")

    return tam_yol


# ==========================================
# 2. ÖN İŞLEME (geriye uyumluluk)
# ==========================================

def preprocess_data(data: np.ndarray) -> np.ndarray:
    """Ham sensör verisini medyan filtre ile temizler.

    Donanım spike'larını kaldırır. Geriye uyumluluk için korunmuştur.
    Daha kapsamlı filtreleme için preprocessing.py modülünü kullanın.

    Args:
        data: Ham sensör verisi, shape (N, M).

    Returns:
        np.ndarray: Temizlenmiş veri, aynı shape.
    """
    clean_data = np.zeros_like(data, dtype=np.float64)
    for i in range(data.shape[1]):
        clean_data[:, i] = medfilt(data[:, i], kernel_size=5)
    return clean_data


# ==========================================
# 3. ÇOK EKSENLİ DTW KARŞILAŞTIRMA
# ==========================================

def _single_axis_dtw(ref: np.ndarray, user: np.ndarray) -> float:
    """Tek eksen için DTW mesafesi hesaplar.

    Args:
        ref: Referans ekseni, shape (N,).
        user: Kullanıcı ekseni, shape (M,).

    Returns:
        float: DTW mesafesi.
    """
    # fastdtw 2D array bekler → reshape
    ref_2d = ref.reshape(-1, 1)
    user_2d = user.reshape(-1, 1)

    distance, _ = fastdtw(
        ref_2d, user_2d,
        radius=DTW_RADIUS,
        dist=lambda x, y: np.linalg.norm(x - y)
    )
    return float(distance)


def multiaxis_dtw(
    ref_data: np.ndarray,
    user_data: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """6 eksen için ağırlıklı DTW mesafesi hesaplar.

    Her eksene ayrı DTW uygulanır, sonra ağırlıklı birleştirilir.

    Args:
        ref_data: Referans verisi, shape (N, 6+).
        user_data: Kullanıcı verisi, shape (M, 6+).
        weights: Eksen ağırlıkları dict'i. None ise varsayılan kullanılır.

    Returns:
        Tuple[float, Dict[str, float]]:
            - Ağırlıklı toplam DTW mesafesi
            - Her eksenin bireysel DTW mesafesi
    """
    if weights is None:
        weights = AXIS_WEIGHTS

    # Normalizasyon (mevcut kodla uyumlu)
    ref_norm = ref_data[:, :6].astype(np.float64) / 1000.0
    user_norm = user_data[:, :6].astype(np.float64) / 1000.0

    eksen_mesafeleri: Dict[str, float] = {}
    agirlikli_toplam = 0.0

    for i, eksen in enumerate(AXIS_NAMES):
        mesafe = _single_axis_dtw(ref_norm[:, i], user_norm[:, i])
        eksen_mesafeleri[eksen] = mesafe
        agirlik = weights.get(eksen, 1.0 / 6.0)
        agirlikli_toplam += mesafe * agirlik

    return agirlikli_toplam, eksen_mesafeleri


# ==========================================
# 4. ENERJİ VE KARMAŞIKLIK ANALİZİ
# ==========================================

def _energy_ratio(ref_data: np.ndarray, user_data: np.ndarray) -> float:
    """İki verinin kinetik enerji oranını hesaplar.

    Args:
        ref_data: Referans IMU verisi (N, 6+), normalize edilmiş.
        user_data: Kullanıcı IMU verisi (M, 6+), normalize edilmiş.

    Returns:
        float: Enerji oranı (0-1 arası, 1 = eşleşiyor).
    """
    ref_energy = np.sum(np.abs(ref_data[:, :6]))
    user_energy = np.sum(np.abs(user_data[:, :6]))
    return float(min(ref_energy, user_energy) / (max(ref_energy, user_energy) + 1e-5))


def _complexity_ratio(ref_data: np.ndarray, user_data: np.ndarray) -> float:
    """Eksen bazında karmaşıklık oranı — vuruş sayısı/şekil benzerliği.

    Her eksenin diferansiyel toplamını karşılaştırır.

    Args:
        ref_data: Referans IMU verisi (N, 6+), normalize edilmiş.
        user_data: Kullanıcı IMU verisi (M, 6+), normalize edilmiş.

    Returns:
        float: Ortalama karmaşıklık oranı (0-1 arası).
    """
    ref_complexity = np.sum(np.abs(np.diff(ref_data[:, :6], axis=0)), axis=0)
    user_complexity = np.sum(np.abs(np.diff(user_data[:, :6], axis=0)), axis=0)

    ratios = np.minimum(ref_complexity, user_complexity) / \
             (np.maximum(ref_complexity, user_complexity) + 1e-5)
    return float(np.mean(ratios))


# ==========================================
# 5. VURUŞ BAZLI KARŞILAŞTIRMA
# ==========================================

def _detect_strokes(
    data: np.ndarray,
    # threshold: int = PRESSURE_THRESHOLD  # manyometre kaldırıldı
) -> List[Tuple[int, int]]:
    """Vuruş bölgelerini tespit eder.

    Not: Manyometre kaldırıldığı için tüm veri tek vuruş olarak kabul edilir.
    # ESKİ: Basınç kanalından vuruş bölgelerini tespit ediyordu.

    Args:
        data: Sensör verisi, shape (N, 6).

    Returns:
        List[Tuple[int, int]]: (başlangıç, bitiş) indeks çiftleri.
    """
    # Manyometre kaldırıldı — tek vuruş varsay
    return [(0, len(data) - 1)]

    # --- ESKİ KOD: Basınç tabanlı vuruş tespiti ---
    # if data.shape[1] < 7:
    #     return [(0, len(data) - 1)]
    # pressure = data[:, 6]
    # temas = pressure > threshold
    # bolgeler: List[Tuple[int, int]] = []
    # i = 0
    # n = len(pressure)
    # while i < n:
    #     if temas[i]:
    #         baslangic = i
    #         while i < n and temas[i]:
    #             i += 1
    #         bolgeler.append((baslangic, i - 1))
    #     else:
    #         i += 1
    # return bolgeler if bolgeler else [(0, len(data) - 1)]


def _stroke_comparison_penalty(
    ref_data: np.ndarray,
    user_data: np.ndarray,
    # threshold: int = PRESSURE_THRESHOLD  # manyometre kaldırıldı
) -> Tuple[float, Dict]:
    """Vuruş bazlı karşılaştırma — sayı ve süre oranı farkına ceza uygular.

    Args:
        ref_data: Referans verisi, shape (N, 6).
        user_data: Kullanıcı verisi, shape (M, 6).
        threshold: Basınç eşiği.

    Returns:
        Tuple[float, Dict]:
            - Ceza çarpanı (0-1 arası, 1 = ceza yok)
            - Detay bilgisi dict'i
    """
    ref_strokes = _detect_strokes(ref_data)
    user_strokes = _detect_strokes(user_data)

    ref_count = len(ref_strokes)
    user_count = len(user_strokes)
    sayi_farki = abs(ref_count - user_count)

    detay = {
        'ref_stroke_count': ref_count,
        'user_stroke_count': user_count,
        'count_diff': sayi_farki,
        'duration_similarity': 1.0,
        'per_stroke_dtw': []
    }

    # Vuruş sayısı farkına göre ceza
    sayi_cezasi = max(0.0, 1.0 - sayi_farki * STROKE_COUNT_PENALTY)

    # Vuruş sayısı aynıysa — her vuruşu eşleştirip detaylı karşılaştır
    sure_benzerlik = 1.0
    if ref_count == user_count and ref_count > 0:
        ref_sureleri = [(b - a + 1) for a, b in ref_strokes]
        user_sureleri = [(b - a + 1) for a, b in user_strokes]
        toplam_ref = sum(ref_sureleri)
        toplam_user = sum(user_sureleri)

        # Her vuruşun oransal süresini karşılaştır
        if toplam_ref > 0 and toplam_user > 0:
            ref_oranlari = [s / toplam_ref for s in ref_sureleri]
            user_oranlari = [s / toplam_user for s in user_sureleri]
            farklar = [abs(r - u) for r, u in zip(ref_oranlari, user_oranlari)]
            sure_benzerlik = max(0.0, 1.0 - sum(farklar))

        detay['duration_similarity'] = round(sure_benzerlik, 3)

    # Toplam ceza çarpanı
    carpan = sayi_cezasi * (0.7 + 0.3 * sure_benzerlik)
    return max(0.0, min(1.0, carpan)), detay


# ==========================================
# 6. ANA KARŞILAŞTIRMA FONKSİYONLARI
# ==========================================

def compare_with_reference(
    user_data: np.ndarray,
    ref_data: np.ndarray
) -> Tuple[float, Dict]:
    """Kullanıcı verisini tek bir referansla detaylı karşılaştırır.

    Çok eksenli DTW, enerji, karmaşıklık ve vuruş analizini birleştirir.

    Args:
        user_data: Kullanıcı verisi, shape (M, 6+).
        ref_data: Referans verisi, shape (N, 6+).

    Returns:
        Tuple[float, Dict]:
            - Ağırlıklı DTW mesafesi
            - Detay dictionary'si:
                - axis_distances: Her eksenin DTW mesafesi
                - energy_ratio: Enerji oranı
                - complexity_ratio: Karmaşıklık oranı
                - stroke_penalty: Vuruş ceza çarpanı
                - stroke_details: Vuruş detayları
    """
    # Normalize et (uyumluluk için)
    ref_norm = ref_data[:, :6].astype(np.float64) / 1000.0
    user_norm = user_data[:, :6].astype(np.float64) / 1000.0

    # Çok eksenli DTW
    dtw_mesafe, eksen_mesafeleri = multiaxis_dtw(ref_data, user_data)

    # Enerji ve karmaşıklık
    enerji = _energy_ratio(ref_norm, user_norm)
    karmasiklik = _complexity_ratio(ref_norm, user_norm)

    # Vuruş karşılaştırma (manyometre kaldırıldı — tek vuruş varsayılır)
    vurus_carpan = 1.0
    vurus_detay = {}
    # ESKİ: Basınç kanalı varsa vuruş karşılaştırması yapılıyordu
    # if ref_data.shape[1] >= 7 and user_data.shape[1] >= 7:
    #     vurus_carpan, vurus_detay = _stroke_comparison_penalty(ref_data, user_data)
    vurus_carpan, vurus_detay = _stroke_comparison_penalty(ref_data, user_data)

    detay = {
        'axis_distances': eksen_mesafeleri,
        'energy_ratio': round(enerji, 3),
        'complexity_ratio': round(karmasiklik, 3),
        'stroke_penalty': round(vurus_carpan, 3),
        'stroke_details': vurus_detay,
    }

    return dtw_mesafe, detay


def calculate_score(
    ref_data: np.ndarray,
    user_data: np.ndarray
) -> Tuple[float, float]:
    """Referans ve kullanıcı verisini karşılaştırarak 0-100 arası skor hesaplar.

    Mevcut fonksiyon imzası korunmuştur (geriye uyumluluk).
    Dahili olarak çok eksenli DTW, enerji, karmaşıklık ve vuruş
    analizini kullanır.

    Args:
        ref_data: Referans verisi, shape (N, 6+).
        user_data: Kullanıcı verisi, shape (M, 6+).

    Returns:
        Tuple[float, float]:
            - final_score: 0-100 arası skor
            - distance: Ham DTW mesafesi
    """
    # Normalize (mevcut kodla aynı şekilde)
    ref_norm = ref_data[:, :6].astype(np.float64) / 1000.0
    user_norm = user_data[:, :6].astype(np.float64) / 1000.0

    # Enerji oranı
    energy_ratio = _energy_ratio(ref_norm, user_norm)

    # Karmaşıklık oranı
    complexity_ratio = _complexity_ratio(ref_norm, user_norm)

    # Çok eksenli DTW mesafesi
    distance, eksen_mesafeleri = multiaxis_dtw(ref_data, user_data)

    # Vuruş bazlı ceza (manyometre kaldırıldı — tek vuruş varsayılır)
    stroke_mult = 1.0
    # ESKİ: Basınç kanalı varsa vuruş cezası uygulanıyordu
    # if ref_data.shape[1] >= 7 and user_data.shape[1] >= 7:
    #     stroke_mult, _ = _stroke_comparison_penalty(ref_data, user_data)
    stroke_mult, _ = _stroke_comparison_penalty(ref_data, user_data)

    print(f"\n[Debug] DTW: {distance:.1f} | Enerji: {energy_ratio * 100:.0f}% | "
          f"Karmaşıklık: {complexity_ratio * 100:.0f}% | "
          f"Vuruş çarpanı: {stroke_mult:.2f}")
    print(f"  → Eksen DTW: " + ", ".join(
        f"{k}={v:.1f}" for k, v in eksen_mesafeleri.items()))

    # DTW mesafesinden temel skor hesapla
    base_score = float(np.interp(
        distance,
        [PERFECT_DISTANCE, TERRIBLE_DISTANCE],
        [100.0, 0.0]
    ))

    # Karmaşıklık cezası — kök fonksiyonuyla yumuşat (mevcut kaldla aynı)
    gentle_complexity_penalty = np.sqrt(complexity_ratio)

    # Final skor — tüm faktörleri birleştir
    final_score = base_score * energy_ratio * gentle_complexity_penalty * stroke_mult

    # ==========================================
    # GİYOTİN — tamamen yanlış harf tespiti
    # ==========================================
    # Enerji çok düşük, karmaşıklık çok farklı veya DTW çok yüksekse → 0 puan
    if (distance >= TERRIBLE_DISTANCE or
            energy_ratio < MIN_ENERGY_RATIO or
            complexity_ratio < MIN_COMPLEXITY_RATIO):
        final_score = 0.0

    return float(final_score), float(distance)


# ==========================================
# 7. HARF TANIMLAMA
# ==========================================

def identify_letter(
    user_data: np.ndarray,
    all_references: Optional[Dict[str, List[np.ndarray]]] = None
) -> Tuple[str, float, Dict[str, float]]:
    """Kullanıcının çizdiği harfi tüm referanslarla karşılaştırarak tanımlar.

    Her harf için tüm referanslarla DTW hesaplanır, minimum alınır.
    En düşük mesafeli harf = tanınan harf.

    Args:
        user_data: Kullanıcı verisi, shape (M, 6+).
        all_references: Harf → referans listesi dict'i.
                        None ise veritabanından yüklenir.

    Returns:
        Tuple[str, float, Dict[str, float]]:
            - taninan_harf: En iyi eşleşen harf (örn: 'A')
            - guven: Güven skoru (0-1 arası)
            - mesafeler: Her harf için en düşük DTW mesafesi
    """
    if all_references is None:
        all_references = load_references()

    if not all_references:
        print("[Tanıma] UYARI: Referans veritabanı boş!")
        return "?", 0.0, {}

    # Her harf için en düşük DTW mesafesini hesapla
    mesafeler: Dict[str, float] = {}

    print("[Tanıma] Harf karşılaştırması başlatılıyor...")
    for harf, referanslar in all_references.items():
        min_mesafe = float('inf')

        for ref in referanslar:
            try:
                mesafe, _ = multiaxis_dtw(ref, user_data)
                min_mesafe = min(min_mesafe, mesafe)
            except Exception as e:
                print(f"  [UYARI] '{harf}' referansı karşılaştırılamadı: {e}")
                continue

        if min_mesafe < float('inf'):
            mesafeler[harf] = round(min_mesafe, 2)
            print(f"  → '{harf}': DTW = {min_mesafe:.1f}")

    if not mesafeler:
        return "?", 0.0, {}

    # En düşük mesafeli harf = tanınan harf
    sirali = sorted(mesafeler.items(), key=lambda x: x[1])
    taninan_harf = sirali[0][0]
    en_dusuk = sirali[0][1]

    # Güven skoru: 1 - (en_düşük / ikinci_en_düşük)
    if len(sirali) >= 2:
        ikinci_en_dusuk = sirali[1][1]
        guven = 1.0 - (en_dusuk / (ikinci_en_dusuk + 1e-5))
        guven = max(0.0, min(1.0, guven))
    else:
        # Tek harf varsa — mesafeye göre güven
        guven = max(0.0, 1.0 - en_dusuk / TERRIBLE_DISTANCE)

    print(f"\n[Tanıma] Sonuç: '{taninan_harf}' (güven: {guven:.1%})")

    return taninan_harf, round(guven, 3), mesafeler


# ==========================================
# 8. 2D YÖRÜNGE HESAPLAMA
# ==========================================

def calculate_2d_trajectory(
    data: np.ndarray,
    # pressure: Optional[np.ndarray] = None,  # manyometre kaldırıldı
    sample_rate: int = SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """İvme verisini çift entegrasyonla 2D pozisyon yörüngesine çevirir.

    Not: Manyometre kaldırıldığı için tüm veri tek vuruş olarak işlenir.
    # ESKİ: Basınç verisi verilmişse vuruş segmentasyonu yapıyordu.

    Args:
        data: Sensör verisi. shape (N, 6).
              Sütun 0,1: ax, ay (ivme)
        sample_rate: Örnekleme hızı (Hz). Varsayılan: 100

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]:
            - px: X pozisyonları, shape (N,)
            - py: Y pozisyonları, shape (N,)
            - stroke_breaks: Vuruş başlangıç indeksleri
              (çizim sırasında kalem kaldırma noktalarını belirtir)
    """
    dt = 1.0 / sample_rate
    n = data.shape[0]

    # Basınç kanalını belirle — MANYOMETRE KALDIRILDI
    # basinc = None
    # if pressure is not None:
    #     basinc = pressure
    # elif data.shape[1] >= 7:
    #     basinc = data[:, 6]

    # Vuruş bölgelerini tespit et (manyometre kaldırıldı — tek vuruş)
    vuruslar = [(0, n - 1)]
    # ESKİ: Basınç kanalıyla vuruş tespiti yapılıyordu
    # if basinc is not None:
    #     vuruslar = _detect_strokes(...)
    # else:
    #     vuruslar = [(0, n - 1)]

    # Sonuç dizileri
    px = np.zeros(n, dtype=np.float64)
    py = np.zeros(n, dtype=np.float64)
    stroke_breaks: List[int] = []

    # Son vuruşun bitiş konumu (vuruşları birleştirmek için)
    son_x = 0.0
    son_y = 0.0

    for v_idx, (v_bas, v_bit) in enumerate(vuruslar):
        if v_idx > 0:
            stroke_breaks.append(v_bas)

        # Vuruş verisini kes
        ax_vurus = detrend(data[v_bas:v_bit + 1, 0].astype(np.float64))
        ay_vurus = detrend(data[v_bas:v_bit + 1, 1].astype(np.float64))
        vurus_len = v_bit - v_bas + 1

        if vurus_len < 2:
            px[v_bas:v_bit + 1] = son_x
            py[v_bas:v_bit + 1] = son_y
            continue

        # ZUPT — hareket büyüklüğü eşiğin altındaysa hızı sıfırla
        hareket = np.sqrt(ax_vurus**2 + ay_vurus**2)
        zupt_maske = hareket < ZUPT_THRESHOLD

        # İlk entegrasyon: ivme → hız
        vx = cumulative_trapezoid(ax_vurus, dx=dt, initial=0)
        vy = cumulative_trapezoid(ay_vurus, dx=dt, initial=0)

        # ZUPT uygula: hareketsiz bölgelerde hızı sıfırla
        vx[zupt_maske] = 0.0
        vy[zupt_maske] = 0.0

        # Hızı detrend et (drift kompanzasyonu)
        vx = detrend(vx)
        vy = detrend(vy)

        # İkinci entegrasyon: hız → pozisyon
        vurus_px = cumulative_trapezoid(vx, dx=dt, initial=0)
        vurus_py = cumulative_trapezoid(vy, dx=dt, initial=0)

        # Bu vuruşun başlangıcını (0,0)'a kilitle
        vurus_px = vurus_px - vurus_px[0]
        vurus_py = vurus_py - vurus_py[0]

        # Önceki vuruşun son konumuna göre ofsetle
        px[v_bas:v_bit + 1] = vurus_px + son_x
        py[v_bas:v_bit + 1] = vurus_py + son_y

        # Bu vuruşun son konumunu kaydet (bir sonraki için)
        son_x = px[v_bit]
        son_y = py[v_bit]

    # Vuruşlar arasını interpole et (boşlukları doldur)
    for i in range(len(vuruslar) - 1):
        bitis_idx = vuruslar[i][1]
        baslangic_idx = vuruslar[i + 1][0]
        if baslangic_idx > bitis_idx + 1:
            # Aradaki boşluğu doğrusal enterpole et
            px[bitis_idx + 1:baslangic_idx] = np.linspace(
                px[bitis_idx], px[baslangic_idx],
                baslangic_idx - bitis_idx - 1
            )
            py[bitis_idx + 1:baslangic_idx] = np.linspace(
                py[bitis_idx], py[baslangic_idx],
                baslangic_idx - bitis_idx - 1
            )

    return px, py, stroke_breaks


# ==========================================
# DOĞRUDAN ÇALIŞTIRMA — TEST MODU
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("  DTW ANALYSIS — TEST MODU")
    print("=" * 50)

    # Referans yükleme testi
    refs = load_references()

    # Sentetik veri ile test (manyometre kaldırıldı — 6 kanal)
    np.random.seed(42)
    t = np.linspace(0, 2, 200)

    # Referans: sinüs benzeri hareket (6 kanal, basınç yok)
    ref = np.column_stack([
        100 * np.sin(2 * np.pi * t),     # ax
        50 * np.cos(2 * np.pi * t),      # ay
        16384 + 10 * np.sin(t),          # az
        200 * np.sin(4 * np.pi * t),     # gx
        150 * np.cos(3 * np.pi * t),     # gy
        100 * np.sin(2 * np.pi * t),     # gz
        # np.where((t > 0.3) & (t < 1.7), 300, 0)  # pressure — MANYOMETRE KALDIRILDI
    ])

    # Kullanıcı: benzer ama gürültülü
    user = ref.copy() + np.random.randn(*ref.shape) * 20
    # user[:, 6] = ref[:, 6] + np.random.randn(200) * 5  # Basınç gürültüsü — KALDIRILDI

    # Skor hesaplama testi
    print("\n--- calculate_score testi ---")
    score, dist = calculate_score(ref, user)
    print(f"Skor: {score:.1f} / 100 | DTW mesafe: {dist:.1f}")

    # compare_with_reference testi
    print("\n--- compare_with_reference testi ---")
    dtw_dist, detay = compare_with_reference(user, ref)
    print(f"DTW: {dtw_dist:.1f}")
    print(f"Enerji: {detay['energy_ratio']:.1%}")
    print(f"Karmaşıklık: {detay['complexity_ratio']:.1%}")
    print(f"Vuruş detay: {detay['stroke_details']}")

    # 2D yörünge testi
    print("\n--- calculate_2d_trajectory testi ---")
    px, py, breaks = calculate_2d_trajectory(ref, sample_rate=100)
    print(f"Yörünge: {len(px)} nokta, {len(breaks)} vuruş kırılması")
    print(f"X aralığı: [{px.min():.2f}, {px.max():.2f}]")
    print(f"Y aralığı: [{py.min():.2f}, {py.max():.2f}]")