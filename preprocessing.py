"""
preprocessing.py — Sinyal Ön İşleme Modülü

IMU sensör verisini filtreler, normalize eder.
Manyometre kaldırıldı — basınç/buton tabanlı segmentasyon yok.
Veri formatı: (N, 6) → [ax, ay, az, gx, gy, gz]
"""

import numpy as np
from scipy.signal import butter, sosfilt, medfilt
from scipy.interpolate import interp1d
from typing import Optional, Dict, List, Tuple

# ── Sabitler ──
DEFAULT_SAMPLE_RATE = 100       # Hz
DEFAULT_CUTOFF_FREQ = 20.0      # Hz (düşük geçiren filtre kesim frekansı)
DEFAULT_FILTER_ORDER = 4        # Butterworth filtre derecesi
DEFAULT_TARGET_LENGTH = 200     # Normalizasyon hedef uzunluk
GRAVITY_RAW = 16384.0           # MPU6500 ±2g ölçeğinde 1g değeri


def filter_data(
    raw_data: np.ndarray,
    offsets: Optional[Dict[str, float]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    cutoff: float = DEFAULT_CUTOFF_FREQ,
) -> np.ndarray:
    """Ham sensör verisini filtreler ve kalibrasyon offset'lerini çıkarır.

    Args:
        raw_data: Ham veri, shape (N, 6) [ax, ay, az, gx, gy, gz].
        offsets: Kalibrasyon offset dict'i. None ise atlanır.
        sample_rate: Örnekleme hızı (Hz).
        cutoff: Düşük geçiren filtre kesim frekansı (Hz).

    Returns:
        Filtrelenmiş veri, shape (N, 6).
    """
    data = raw_data.copy().astype(float)

    # Sadece 6 kanal al (fazla sütun varsa kırp)
    if data.shape[1] > 6:
        data = data[:, :6]

    # Kalibrasyon offset çıkarma
    if offsets is not None:
        eksen_isimleri = ["ax", "ay", "az", "gx", "gy", "gz"]
        for i, isim in enumerate(eksen_isimleri):
            if isim in offsets:
                data[:, i] -= offsets[isim]

    # Yerçekimi kompanzasyonu (az ekseninden)
    data[:, 2] -= GRAVITY_RAW

    # Butterworth düşük geçiren filtre
    if data.shape[0] > 15:
        try:
            nyquist = sample_rate / 2.0
            normalized_cutoff = min(cutoff / nyquist, 0.99)
            sos = butter(DEFAULT_FILTER_ORDER, normalized_cutoff,
                         btype="low", output="sos")
            for col in range(6):
                data[:, col] = sosfilt(sos, data[:, col])
        except Exception as e:
            print(f"[UYARI] Filtreleme hatası: {e}")

    return data


def normalize_data(
    filtered_data: np.ndarray,
    target_length: int = DEFAULT_TARGET_LENGTH,
) -> np.ndarray:
    """Veriyi sabit uzunluğa getirir ve z-score normalize eder.

    Args:
        filtered_data: Filtrelenmiş veri, shape (N, 6).
        target_length: Hedef uzunluk.

    Returns:
        Normalize edilmiş veri, shape (target_length, 6).
    """
    data = filtered_data[:, :6] if filtered_data.shape[1] > 6 else filtered_data
    n = data.shape[0]

    if n < 2:
        return np.zeros((target_length, 6))

    # Zaman ekseni interpolasyonu
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    result = np.zeros((target_length, 6))

    for col in range(6):
        interp_fn = interp1d(x_old, data[:, col], kind="linear",
                             fill_value="extrapolate")
        result[:, col] = interp_fn(x_new)

    # Z-score normalizasyon
    for col in range(6):
        std = np.std(result[:, col])
        if std > 1e-8:
            result[:, col] = (result[:, col] - np.mean(result[:, col])) / std

    return result


def process_pipeline(
    raw_data: np.ndarray,
    offsets: Optional[Dict[str, float]] = None,
    target_length: int = DEFAULT_TARGET_LENGTH,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Tam ön-işleme hattı: filtrele → normalizasyon.

    Manyometre olmadığı için tüm veri tek vuruş olarak işlenir.

    Args:
        raw_data: Ham sensör verisi, shape (N, 6).
        offsets: Kalibrasyon offset dict'i.
        target_length: Normalizasyon hedef uzunluk.

    Returns:
        Tuple: (normalized_strokes, stroke_info)
            normalized_strokes: [np.array shape (target_length, 6)]
            stroke_info: [dict] — tek vuruş bilgisi
    """
    # Filtrele
    filtered = filter_data(raw_data, offsets)

    # Normalize et (tek vuruş olarak)
    normalized = normalize_data(filtered, target_length)

    stroke_info = [{
        "start_idx": 0,
        "end_idx": filtered.shape[0] - 1,
        "duration": filtered.shape[0] / DEFAULT_SAMPLE_RATE,
        "sample_count": filtered.shape[0],
    }]

    return ([normalized], stroke_info)