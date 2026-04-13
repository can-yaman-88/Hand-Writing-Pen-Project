"""
dtw_analysis.py — DTW Tabanlı Harf Tanıma ve Sinyal Karşılaştırma Modülü

Tüm tanımlama DTW ile yapılır (ML kullanılmıyor).
Referans veritabanı: letters_db/<HARF>/ref_01.npy, ref_02.npy, ...
Veri formatı: (N, 6) → [ax, ay, az, gx, gy, gz]
Manyometre/basınç sensörü kaldırıldı, buton tabanlı kayıt kullanılıyor.
"""

import os
import glob
import numpy as np
from fastdtw import fastdtw
from scipy.signal import medfilt, detrend, butter, sosfilt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from typing import Optional, Dict, List, Tuple

# ── Sabitler ──
NORMALIZE_LENGTH = 200        # Normalizasyon hedef uzunluk
SAMPLE_RATE = 100             # Hz

# DTW skor eşikleri
PERFECT_DISTANCE = 800.0
TERRIBLE_DISTANCE = 2500.0

# Eksen ağırlıkları (DTW birleştirme)
EKSEN_AGIRLIKLARI = {
    "ax": 0.20, "ay": 0.20, "az": 0.10,
    "gx": 0.20, "gy": 0.20, "gz": 0.10,
}

# ──────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ──────────────────────────────────────────────

def _normalize_signal(data: np.ndarray, target_len: int = NORMALIZE_LENGTH) -> np.ndarray:
    """Sinyal verisini sabit uzunluğa getirir ve z-score normalize eder.

    Args:
        data: Ham sinyal, shape (N, K).
        target_len: Hedef uzunluk.

    Returns:
        Normalize edilmiş sinyal, shape (target_len, K).
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n, k = data.shape
    if n < 2:
        return np.zeros((target_len, k))

    # Zaman ekseninde interpolasyon
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, k))

    for col in range(k):
        interp_fn = interp1d(x_old, data[:, col], kind="linear", fill_value="extrapolate")
        result[:, col] = interp_fn(x_new)

    # Z-score normalizasyon
    for col in range(k):
        std = np.std(result[:, col])
        if std > 1e-8:
            result[:, col] = (result[:, col] - np.mean(result[:, col])) / std
        else:
            result[:, col] = 0.0

    return result


def _single_axis_dtw(ref: np.ndarray, user: np.ndarray) -> float:
    """Tek eksen için DTW mesafesi hesaplar.

    Args:
        ref: Referans ekseni, shape (N,).
        user: Kullanıcı ekseni, shape (M,).

    Returns:
        float: DTW mesafesi.
    """
    ref_2d = ref.reshape(-1, 1)
    user_2d = user.reshape(-1, 1)
    distance, _ = fastdtw(ref_2d, user_2d, radius=15,
                           dist=lambda x, y: np.linalg.norm(x - y))
    return float(distance)


def _multi_axis_dtw(ref: np.ndarray, user: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """6 eksen üzerinden ağırlıklı DTW mesafesi hesaplar.

    Args:
        ref: Referans verisi, shape (N, 6).
        user: Kullanıcı verisi, shape (M, 6).

    Returns:
        Tuple: (ağırlıklı toplam mesafe, eksen bazlı mesafeler dict).
    """
    eksen_isimleri = ["ax", "ay", "az", "gx", "gy", "gz"]
    eksen_mesafeleri = {}
    agirlikli_toplam = 0.0

    for i, isim in enumerate(eksen_isimleri):
        mesafe = _single_axis_dtw(ref[:, i], user[:, i])
        eksen_mesafeleri[isim] = mesafe
        agirlikli_toplam += mesafe * EKSEN_AGIRLIKLARI[isim]

    return agirlikli_toplam, eksen_mesafeleri


def _complexity_ratio(ref: np.ndarray, user: np.ndarray) -> float:
    """İki sinyalin karmaşıklık oranını hesaplar.

    Sıfır geçiş sayılarını karşılaştırır.

    Args:
        ref: Referans verisi, shape (N, K).
        user: Kullanıcı verisi, shape (M, K).

    Returns:
        float: 0.0-1.0 arası oran (1.0 = mükemmel eşleşme).
    """
    def count_zero_crossings(signal: np.ndarray) -> int:
        total = 0
        for col in range(signal.shape[1]):
            s = signal[:, col]
            s = s - np.mean(s)
            crossings = np.where(np.diff(np.sign(s)))[0]
            total += len(crossings)
        return max(total, 1)

    ref_zc = count_zero_crossings(ref)
    user_zc = count_zero_crossings(user)

    ratio = min(ref_zc, user_zc) / max(ref_zc, user_zc)
    return float(ratio)


def _energy_ratio(ref: np.ndarray, user: np.ndarray) -> float:
    """İki sinyalin enerji oranını hesaplar.

    Args:
        ref: Referans verisi.
        user: Kullanıcı verisi.

    Returns:
        float: 0.0-1.0 arası oran.
    """
    ref_energy = np.sum(ref ** 2)
    user_energy = np.sum(user ** 2)

    if ref_energy < 1e-8 and user_energy < 1e-8:
        return 1.0
    if ref_energy < 1e-8 or user_energy < 1e-8:
        return 0.0

    ratio = min(ref_energy, user_energy) / max(ref_energy, user_energy)
    return float(ratio)


# ──────────────────────────────────────────────
# Referans Veritabanı Yönetimi
# ──────────────────────────────────────────────

def load_references(db_path: str = "letters_db") -> Dict[str, List[np.ndarray]]:
    """Tüm referans verilerini yükler.

    Klasör yapısı:
      letters_db/A/ref_01.npy, ref_02.npy, ...
      letters_db/B/ref_01.npy, ...
    
    Eski format da desteklenir (geriye uyumluluk):
      letters_db/letter_A.npy

    Args:
        db_path: Veritabanı kök klasörü.

    Returns:
        dict: {harf: [referans_array_listesi]}
    """
    references = {}

    if not os.path.exists(db_path):
        os.makedirs(db_path)
        return references

    # Yeni format: letters_db/A/ref_XX.npy
    for item in os.listdir(db_path):
        item_path = os.path.join(db_path, item)
        if os.path.isdir(item_path) and len(item) <= 2:
            letter = item.upper()
            refs = []
            for npy_file in sorted(glob.glob(os.path.join(item_path, "ref_*.npy"))):
                try:
                    data = np.load(npy_file)
                    if data.shape[0] > 5 and data.shape[1] >= 6:
                        refs.append(data[:, :6])  # Sadece 6 IMU kanalı
                except Exception as e:
                    print(f"[UYARI] {npy_file} yüklenemedi: {e}")
            if refs:
                references[letter] = refs

    # Eski format: letters_db/letter_A.npy (geriye uyumluluk)
    for npy_file in glob.glob(os.path.join(db_path, "letter_*.npy")):
        try:
            fname = os.path.basename(npy_file)
            letter = fname.replace("letter_", "").replace(".npy", "").upper()
            if letter not in references:
                data = np.load(npy_file)
                if data.shape[0] > 5:
                    arr = data[:, :6] if data.shape[1] >= 6 else data
                    references[letter] = [arr]
        except Exception as e:
            print(f"[UYARI] {npy_file} yüklenemedi: {e}")

    return references


def save_reference(letter: str, data: np.ndarray, db_path: str = "letters_db") -> str:
    """Yeni referans kaydeder.

    Args:
        letter: Harf (ör: "A").
        data: Sensör verisi, shape (N, 6).
        db_path: Veritabanı kök klasörü.

    Returns:
        str: Kaydedilen dosya yolu.
    """
    letter = letter.upper()
    letter_dir = os.path.join(db_path, letter)
    os.makedirs(letter_dir, exist_ok=True)

    # Mevcut referans sayısını bul
    existing = glob.glob(os.path.join(letter_dir, "ref_*.npy"))
    next_num = len(existing) + 1
    file_path = os.path.join(letter_dir, f"ref_{next_num:02d}.npy")

    np.save(file_path, data[:, :6])
    print(f"[OK] Referans kaydedildi: {file_path}")
    return file_path


# ──────────────────────────────────────────────
# Harf Tanımlama
# ──────────────────────────────────────────────

def identify_letter(
    user_data: np.ndarray,
    references: Dict[str, List[np.ndarray]],
) -> Tuple[str, float, Dict[str, float]]:
    """Kullanıcının yazısını tüm referanslarla karşılaştırarak harf tanımlar.

    Args:
        user_data: Kullanıcı verisi, shape (N, 6).
        references: load_references() çıktısı.

    Returns:
        Tuple: (tanınan_harf, güven_skoru, tüm_mesafeler)
               tanınan_harf: En yakın harf (str)
               güven_skoru: 0.0-1.0 arası
               tüm_mesafeler: {harf: min_mesafe}
    """
    if not references:
        return ("?", 0.0, {})

    user_norm = _normalize_signal(user_data[:, :6])
    tum_mesafeler = {}

    for harf, ref_list in references.items():
        min_mesafe = float("inf")
        for ref_data in ref_list:
            ref_norm = _normalize_signal(ref_data[:, :6])
            mesafe, _ = _multi_axis_dtw(ref_norm, user_norm)
            min_mesafe = min(min_mesafe, mesafe)
        tum_mesafeler[harf] = min_mesafe

    if not tum_mesafeler:
        return ("?", 0.0, {})

    # En yakın harfi bul
    sirali = sorted(tum_mesafeler.items(), key=lambda x: x[1])
    en_yakin_harf = sirali[0][0]
    en_yakin_mesafe = sirali[0][1]

    # Güven skoru hesapla
    if len(sirali) >= 2:
        ikinci_mesafe = sirali[1][1]
        if ikinci_mesafe > 1e-8:
            guven = 1.0 - (en_yakin_mesafe / ikinci_mesafe)
            guven = max(0.0, min(1.0, guven))
        else:
            guven = 0.0
    else:
        # Tek harf var — mesafeye göre güven
        guven = max(0.0, 1.0 - (en_yakin_mesafe / TERRIBLE_DISTANCE))

    return (en_yakin_harf, guven, tum_mesafeler)


# ──────────────────────────────────────────────
# Skor Hesaplama
# ──────────────────────────────────────────────

def preprocess_data(raw_data: np.ndarray) -> np.ndarray:
    """Ham veriyi normalize eder (geriye uyumluluk fonksiyonu).

    Args:
        raw_data: Ham sensör verisi.

    Returns:
        Normalize edilmiş veri, shape (NORMALIZE_LENGTH, 6).
    """
    data = raw_data[:, :6] if raw_data.shape[1] > 6 else raw_data
    return _normalize_signal(data)


def calculate_score(
    ref_data: np.ndarray,
    user_data: np.ndarray,
) -> Tuple[float, float]:
    """Referans ve kullanıcı verisi arasında skor hesaplar.

    NOT: Bu fonksiyon geriye uyumluluk için 2 değer döndürür.
    GUI ve diğer modüller bu formatı bekler.

    Args:
        ref_data: Referans verisi (ham veya normalize).
        user_data: Kullanıcı verisi (ham veya normalize).

    Returns:
        Tuple: (final_score, raw_distance)
               final_score: 0-100 arası skor
               raw_distance: Ham DTW mesafesi
    """
    # Normalize et (eğer henüz yapılmadıysa)
    if ref_data.shape[0] != NORMALIZE_LENGTH:
        ref_norm = _normalize_signal(ref_data[:, :6])
    else:
        ref_norm = ref_data[:, :6] if ref_data.shape[1] > 6 else ref_data

    if user_data.shape[0] != NORMALIZE_LENGTH:
        user_norm = _normalize_signal(user_data[:, :6])
    else:
        user_norm = user_data[:, :6] if user_data.shape[1] > 6 else user_data

    # Çok eksenli DTW mesafesi
    distance, eksen_detay = _multi_axis_dtw(ref_norm, user_norm)

    # Enerji oranı
    e_ratio = _energy_ratio(ref_norm, user_norm)

    # Karmaşıklık oranı
    c_ratio = _complexity_ratio(ref_norm, user_norm)

    # Temel skor (mesafeden)
    base_score = float(np.interp(distance,
                                  [PERFECT_DISTANCE, TERRIBLE_DISTANCE],
                                  [100.0, 0.0]))
    base_score = max(0.0, min(100.0, base_score))

    # Tamamen yanlış yazım kontrolü
    if distance >= TERRIBLE_DISTANCE or e_ratio < 0.15 or c_ratio < 0.20:
        return (0.0, distance)

    # Yumuşak cezalar
    gentle_complexity_penalty = np.sqrt(c_ratio)
    final_score = base_score * e_ratio * gentle_complexity_penalty
    final_score = max(0.0, min(100.0, final_score))

    print(f"\n[Debug] DTW: {distance:.1f} | Energy: {e_ratio*100:.0f}% | "
          f"Complexity: {c_ratio*100:.0f}% | Score: {final_score:.1f}%")

    return (final_score, distance)


# ──────────────────────────────────────────────
# 2D Yörünge Hesaplama
# ──────────────────────────────────────────────

def calculate_2d_trajectory(
    data: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """İvme verisinden çift entegrasyon ile tahmini 2D yörünge çıkarır.

    NOT: Bu fonksiyon geriye uyumluluk için 2 değer döndürür:
    (x_positions, y_positions)

    Args:
        data: Sensör verisi, shape (N, 6) veya (N, K).
        sample_rate: Örnekleme hızı (Hz).

    Returns:
        Tuple: (x_positions, y_positions) — her biri shape (N,)
    """
    if data.shape[0] < 3:
        return (np.zeros(2), np.zeros(2))

    dt = 1.0 / sample_rate

    # İvme verisi — trend çıkarma
    accel_x = detrend(data[:, 0].astype(float))
    accel_y = detrend(data[:, 1].astype(float))

    # Yüksek geçiren filtre (drift azaltma)
    try:
        sos = butter(2, 0.5, btype="high", fs=sample_rate, output="sos")
        accel_x = sosfilt(sos, accel_x)
        accel_y = sosfilt(sos, accel_y)
    except Exception:
        pass  # Filtre başarısız olursa filtresiz devam et

    # İlk entegrasyon: ivme → hız
    vel_x = cumulative_trapezoid(accel_x, dx=dt, initial=0)
    vel_y = cumulative_trapezoid(accel_y, dx=dt, initial=0)

    # ZUPT: düşük hareket anlarında hızı sıfırla
    accel_mag = np.sqrt(accel_x**2 + accel_y**2)
    zupt_threshold = np.percentile(accel_mag, 15)
    zupt_mask = accel_mag < zupt_threshold
    vel_x[zupt_mask] = 0.0
    vel_y[zupt_mask] = 0.0

    # Hız trend çıkarma
    vel_x = detrend(vel_x)
    vel_y = detrend(vel_y)

    # İkinci entegrasyon: hız → konum
    pos_x = cumulative_trapezoid(vel_x, dx=dt, initial=0)
    pos_y = cumulative_trapezoid(vel_y, dx=dt, initial=0)

    # Konum normalizasyon (-1, 1 aralığına)
    max_range = max(np.ptp(pos_x), np.ptp(pos_y), 1e-8)
    pos_x = (pos_x - np.mean(pos_x)) / max_range
    pos_y = (pos_y - np.mean(pos_y)) / max_range

    return (pos_x, pos_y)