"""
preprocessing.py — IMU Sinyal İşleme Modülü

Ham sensör verilerini temizler, kalibre eder, vuruşlara ayırır
ve normalize eder. MPU6500 (±2g / ±250°/s) için
optimize edilmiştir.

Not: Manyometre (FSR 402 basınç sensörü) devre dışı bırakılmıştır.
Veri formatı: (N, 6) → [ax, ay, az, gx, gy, gz]
"""

import numpy as np
from scipy.signal import butter, sosfilt, medfilt
from scipy.interpolate import interp1d
from typing import Optional, List, Dict, Tuple

# ==========================================
# SABİTLER
# ==========================================

SAMPLE_RATE: int = 100              # Arduino örnekleme hızı (Hz)
BUTTERWORTH_CUTOFF: float = 20.0    # Düşük geçiren filtre kesim frekansı (Hz)
BUTTERWORTH_ORDER: int = 4          # Butterworth filtre derecesi
# MEDIAN_KERNEL: int = 5              # Basınç kanalı medyan filtre pencere boyutu — manyometre kaldırıldı

GRAVITY_RAW: float = 16384.0        # MPU6500 ±2g ölçeğinde 1g = 16384 LSB

# PRESSURE_THRESHOLD: int = 50        # Vuruş algılama basınç eşiği — manyometre kaldırıldı
MIN_STROKE_DURATION: float = 0.05   # Minimum vuruş süresi (saniye)
MIN_STROKE_GAP: float = 0.03        # Minimum vuruşlar arası boşluk (saniye)

# Örnek sayısına çevir (100 Hz örnekleme hızında)
MIN_STROKE_SAMPLES: int = int(MIN_STROKE_DURATION * SAMPLE_RATE)  # 5 örnek
MIN_GAP_SAMPLES: int = int(MIN_STROKE_GAP * SAMPLE_RATE)          # 3 örnek

DEFAULT_TARGET_LENGTH: int = 200    # Normalizasyon sonrası sabit uzunluk

# Eksen isimleri (referans için)
AXIS_NAMES: List[str] = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
# AXIS_NAMES_ESKI: List[str] = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pressure']  # ESKİ
IMU_AXES: List[str] = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']


# ==========================================
# 1. GÜRÜLTÜ FİLTRELEME
# ==========================================

def _design_lowpass(cutoff: float = BUTTERWORTH_CUTOFF,
                    fs: int = SAMPLE_RATE,
                    order: int = BUTTERWORTH_ORDER) -> np.ndarray:
    """Butterworth düşük geçiren filtre katsayılarını hesaplar.

    Args:
        cutoff: Kesim frekansı (Hz).
        fs: Örnekleme hızı (Hz).
        order: Filtre derecesi.

    Returns:
        np.ndarray: SOS (second-order sections) formatında filtre katsayıları.
    """
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    return sos


def filter_data(
    raw_data: np.ndarray,
    offsets: Optional[Dict[str, float]] = None,
    cutoff: float = BUTTERWORTH_CUTOFF,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Ham sensör verisine gürültü filtresi ve kalibrasyon uygular.

    İşlem sırası:
      1. Kalibrasyon offset'lerini çıkar (verilmişse)
      2. Yerçekimi kompanzasyonu: az ekseninden 16384 çıkar
      3. IMU kanallarına (ax-gz) Butterworth düşük geçiren filtre uygula
      # 4. Basınç kanalına medyan filtre uygula — manyometre kaldırıldı

    Args:
        raw_data: Ham sensör verisi, shape (N, 6).
                  Sütunlar: ax, ay, az, gx, gy, gz
        offsets: Kalibrasyon offset dictionary'si.
                 Anahtarlar: 'ax', 'ay', 'az', 'gx', 'gy', 'gz'
                 None ise offset çıkarma atlanır.
        cutoff: Butterworth filtre kesim frekansı (Hz). Varsayılan: 20
        sample_rate: Örnekleme hızı (Hz). Varsayılan: 100

    Returns:
        np.ndarray: Filtrelenmiş veri, shape (N, 6). Orijinal veri değiştirilmez.

    Raises:
        ValueError: Giriş verisi beklenen formatta değilse.
    """
    # --- Giriş doğrulama ---
    if raw_data.ndim != 2 or raw_data.shape[1] != 6:
        raise ValueError(
            f"Beklenen shape: (N, 6), gelen: {raw_data.shape}. "
            f"Sütunlar: ax, ay, az, gx, gy, gz olmalı."
        )

    if raw_data.shape[0] < 10:
        # Çok kısa veri — filtre uygulanamaz, olduğu gibi döndür
        print(f"[Filtre] UYARI: Veri çok kısa ({raw_data.shape[0]} örnek), "
              f"filtreleme atlanıyor.")
        return raw_data.copy()

    # Orijinal veriyi bozmamak için kopya oluştur
    filtered = raw_data.astype(np.float64).copy()

    # --- Kalibrasyon offset çıkarma ---
    if offsets is not None:
        for i, eksen in enumerate(AXIS_NAMES):
            if eksen in offsets:
                filtered[:, i] -= offsets[eksen]

    # --- Yerçekimi kompanzasyonu (az ekseni, indeks 2) ---
    # MPU6500 ±2g ölçeğinde düz duran sensörde az ≈ +16384
    filtered[:, 2] -= GRAVITY_RAW

    # --- Butterworth düşük geçiren filtre (6 IMU kanalı) ---
    sos = _design_lowpass(cutoff=cutoff, fs=sample_rate)
    for i in range(6):  # ax, ay, az, gx, gy, gz
        filtered[:, i] = sosfilt(sos, filtered[:, i])

    # --- Basınç kanalına medyan filtre --- MANYOMETRE KALDIRILDI
    # filtered[:, 6] = medfilt(filtered[:, 6], kernel_size=MEDIAN_KERNEL)

    return filtered


# ==========================================
# 2. VURUŞ SEGMENTASYONU
# ==========================================

def segment_strokes(
    filtered_data: np.ndarray,
    # threshold: int = PRESSURE_THRESHOLD,  # manyometre kaldırıldı
    sample_rate: int = SAMPLE_RATE,
    min_duration: float = MIN_STROKE_DURATION,
    min_gap: float = MIN_STROKE_GAP
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Filtrelenmiş veriyi vuruşlara ayırır.

    Not: Manyometre (basınç sensörü) kaldırıldığı için tüm veri
    tek vuruş olarak kabul edilir. Buton ile kayıt başlatılıp
    durdurulduğu için segmentasyon gerekmez.

    # ESKİ: Basınç kanalını kullanarak temas/kaldırma noktalarını tespit ediyordu.

    Args:
        filtered_data: Filtrelenmiş sensör verisi, shape (N, 6).
        sample_rate: Örnekleme hızı (Hz). Varsayılan: 100
        min_duration: Minimum vuruş süresi (saniye). Varsayılan: 0.05
        min_gap: Minimum vuruşlar arası boşluk (saniye). Varsayılan: 0.03

    Returns:
        Tuple[List[np.ndarray], List[Dict]]:
            - strokes: Her vuruş için ayrı NumPy array listesi, shape (M, 6)
            - stroke_info: Her vuruş için bilgi dictionary'si listesi:
                - start_idx (int): Başlangıç indeksi
                - end_idx (int): Bitiş indeksi
                - duration (float): Vuruş süresi (saniye)
                - sample_count (int): Örnek sayısı
    """
    n_samples = filtered_data.shape[0]

    if n_samples == 0:
        print("[Segmentasyon] Hiç vuruş algılanamadı.")
        return [], []

    # --- Manyometre kaldırıldı: tüm veriyi tek vuruş olarak kabul et ---
    # ESKİ KOD: Basınç eşiğine göre segmentasyon yapılıyordu
    # pressure = filtered_data[:, 6]
    # temas = pressure > threshold  # Boolean maske
    # ... (basınç tabanlı segmentasyon kodu) ...

    # Tek vuruş: tüm veri
    baslangic = 0
    bitis = n_samples - 1
    sure = n_samples / sample_rate

    strokes = [filtered_data.copy()]
    stroke_info = [{
        'start_idx': baslangic,
        'end_idx': bitis,
        'duration': round(sure, 4),
        # 'mean_pressure': 0.0,  # manyometre kaldırıldı
        'sample_count': n_samples
    }]

    print(f"[Segmentasyon] 1 vuruş (tek blok — buton kontrollü, "
          f"toplam {n_samples} örnek, {sure:.3f}s)")

    return strokes, stroke_info


# ==========================================
# 3. NORMALİZASYON
# ==========================================

def normalize_data(
    filtered_data: np.ndarray,
    target_length: int = DEFAULT_TARGET_LENGTH
) -> np.ndarray:
    """IMU verilerini z-score ile normalize eder ve sabit uzunluğa resample eder.

    İşlem sırası:
      1. Sadece IMU kanallarını (ax-gz, 6 kanal) al
      2. Her kanalı z-score normalize et: (x - mean) / std
      3. scipy.interpolate.interp1d ile target_length noktaya resample et

    Basınç kanalı bu işleme dahil edilmez — sadece 6 IMU kanalı
    normalize edilir ve döndürülür.

    Args:
        filtered_data: Filtrelenmiş sensör verisi, shape (N, 7) veya (N, 6).
                       İlk 6 sütun IMU kanalları olmalı.
        target_length: Hedef uzunluk (örnek sayısı). Varsayılan: 200

    Returns:
        np.ndarray: Normalize ve resample edilmiş veri, shape (target_length, 6).
                    Sütunlar: ax, ay, az, gx, gy, gz (z-score normalize)

    Raises:
        ValueError: Giriş verisi çok kısa veya geçersiz ise.
    """
    if filtered_data.shape[0] < 2:
        raise ValueError(
            f"Normalizasyon için en az 2 örnek gerekli, "
            f"gelen: {filtered_data.shape[0]}"
        )

    # Sadece IMU kanallarını al (ilk 6 sütun)
    imu_data = filtered_data[:, :6].astype(np.float64)
    n_samples = imu_data.shape[0]

    # --- Z-score normalizasyon (her kanal bağımsız) ---
    normalized = np.zeros_like(imu_data)
    for i in range(6):
        kanal = imu_data[:, i]
        ortalama = np.mean(kanal)
        std_sapma = np.std(kanal)

        if std_sapma < 1e-10:
            # Sabit sinyal — sıfır olarak bırak
            normalized[:, i] = 0.0
        else:
            normalized[:, i] = (kanal - ortalama) / std_sapma

    # --- Sabit uzunluğa resample (interpolasyon) ---
    if n_samples == target_length:
        # Zaten doğru uzunlukta, interpolasyon gereksiz
        return normalized

    # Orijinal zaman ekseni [0, 1] aralığına normalize edilir
    t_original = np.linspace(0, 1, n_samples)
    t_target = np.linspace(0, 1, target_length)

    resampled = np.zeros((target_length, 6), dtype=np.float64)
    for i in range(6):
        # Kübik interpolasyon — pürüzsüz geçişler için
        interpolator = interp1d(
            t_original, normalized[:, i],
            kind='cubic',
            fill_value='extrapolate'
        )
        resampled[:, i] = interpolator(t_target)

    return resampled


# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================

def process_pipeline(
    raw_data: np.ndarray,
    offsets: Optional[Dict[str, float]] = None,
    # threshold: int = PRESSURE_THRESHOLD,  # manyometre kaldırıldı
    target_length: int = DEFAULT_TARGET_LENGTH
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Tam ön-işleme hattı: filtrele → segmentasyon → normalizasyon.

    Kolaylık fonksiyonu — üç adımı tek çağrıda uygular.
    Her vuruş bağımsız olarak normalize edilir.

    Args:
        raw_data: Ham sensör verisi, shape (N, 6).
        offsets: Kalibrasyon offset dictionary'si. None ise atlanır.
        # threshold: Basınç eşik değeri — manyometre kaldırıldı
        target_length: Normalizasyon sonrası hedef uzunluk. Varsayılan: 200

    Returns:
        Tuple[List[np.ndarray], List[Dict]]:
            - normalized_strokes: Her vuruş normalize edilmiş, shape (target_length, 6)
            - stroke_info: Her vuruş için bilgi dictionary'si
    """
    print("=" * 50)
    print("  ÖN-İŞLEME HATTI BAŞLATILIYOR")
    print("=" * 50)
    print(f"  → Giriş: {raw_data.shape}")

    # Adım 1: Filtreleme + kalibrasyon
    print("\n[1/3] Filtreleme ve kalibrasyon...")
    filtered = filter_data(raw_data, offsets=offsets)

    # Adım 2: Vuruş segmentasyonu (tüm veri tek vuruş — buton kontrollü)
    print("\n[2/3] Vuruş segmentasyonu...")
    strokes, stroke_info = segment_strokes(filtered)
    # ESKİ: strokes, stroke_info = segment_strokes(filtered, threshold=threshold)

    if not strokes:
        print("\n[UYARI] Hiç vuruş bulunamadı, boş sonuç döndürülüyor.")
        return [], []

    # Adım 3: Her vuruşu normalize et
    print(f"\n[3/3] Normalizasyon ({target_length} noktaya resample)...")
    normalized_strokes: List[np.ndarray] = []

    for i, stroke in enumerate(strokes):
        try:
            norm = normalize_data(stroke, target_length=target_length)
            normalized_strokes.append(norm)
        except ValueError as e:
            print(f"  [UYARI] Vuruş #{i+1} normalize edilemedi: {e}")
            continue

    print(f"\n[Sonuç] {len(normalized_strokes)} vuruş başarıyla işlendi.")
    print(f"  → Her vuruş shape: ({target_length}, 6)")
    print("=" * 50)

    return normalized_strokes, stroke_info


# ==========================================
# DOĞRUDAN ÇALIŞTIRMA — TEST MODU
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("  PREPROCESSING — TEST MODU")
    print("=" * 50)

    # Sentetik test verisi oluştur (manyometre kaldırıldı — 6 kanal)
    np.random.seed(42)
    n = 500  # 5 saniyelik veri (100 Hz)
    t = np.linspace(0, 5, n)

    # IMU sinyalleri: sinüs + gürültü
    ax = 100 * np.sin(2 * np.pi * 3 * t) + np.random.randn(n) * 20
    ay = 50 * np.cos(2 * np.pi * 2 * t) + np.random.randn(n) * 15
    az = 16384 + 30 * np.sin(2 * np.pi * 1 * t) + np.random.randn(n) * 10
    gx = 200 * np.sin(2 * np.pi * 5 * t) + np.random.randn(n) * 30
    gy = 150 * np.cos(2 * np.pi * 4 * t) + np.random.randn(n) * 25
    gz = 100 * np.sin(2 * np.pi * 6 * t) + np.random.randn(n) * 20

    # Basınç: MANYOMETRE KALDIRILDI
    # pressure = np.zeros(n)
    # pressure[50:200] = 300    # Vuruş 1: indeks 50-199 (1.5 saniye)
    # pressure[280:400] = 250   # Vuruş 2: indeks 280-399 (1.2 saniye)
    # pressure += np.random.randn(n) * 5  # Gürültü ekle

    raw_data = np.column_stack([ax, ay, az, gx, gy, gz])
    # ESKİ: raw_data = np.column_stack([ax, ay, az, gx, gy, gz, pressure])
    print(f"\nSentetik veri oluşturuldu: {raw_data.shape}")

    # Tam pipeline testi
    offsets = {'ax': 2.0, 'ay': -1.5, 'az': 0.0,
               'gx': 5.0, 'gy': -3.0, 'gz': 1.0}
    # ESKİ: offsets içinde 'pressure': 0.0 de vardı

    normalized_strokes, stroke_info = process_pipeline(
        raw_data, offsets=offsets, target_length=200
    )
    # ESKİ: process_pipeline(raw_data, offsets=offsets, threshold=50, target_length=200)

    if normalized_strokes:
        print(f"\nİlk vuruş istatistikleri:")
        stroke = normalized_strokes[0]
        for i, eksen in enumerate(IMU_AXES):
            print(f"  {eksen}: mean={stroke[:, i].mean():.4f}, "
                  f"std={stroke[:, i].std():.4f}, "
                  f"min={stroke[:, i].min():.2f}, max={stroke[:, i].max():.2f}")
