"""
trajectory.py — IMU Verisinden 2D Kalem İzi Çıkarma ve Görselleştirme Modülü

IMU sensör verisini çift entegrasyonla (ivme → hız → konum) yaklaşık
bir 2D kalem izine dönüştürür. Drift azaltma teknikleri uygulanır.

ÖNEMLİ: Bu görselleştirme YAKLAŞIK bir tahmindir.
Çift entegrasyondaki kümülatif drift nedeniyle mükemmel değildir.
Kullanıcıya "yazdığın harf kabaca böyle görünüyor" bilgisi vermek
için kullanılır. Ana harf analizi buna bağımlı DEĞİLDİR — o iş
DTW ile yapılır (dtw_analysis.py).

Veri formatı: (N, 6+) → [ax, ay, az, gx, gy, gz]
Not: Manyometre (FSR 402) kaldırıldığı için basınç sütunu kullanılmamaktadır.

Bağımlılıklar:
    - numpy, scipy, matplotlib
    - dtw_analysis.py (referans yörünge hesaplama için, opsiyonel)
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için güvenli backend
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.figure import Figure

from scipy.signal import (
    detrend,
    medfilt,
    butter,
    filtfilt,
)
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import UnivariateSpline

from typing import (
    Optional,
    Dict,
    List,
    Tuple,
    Any,
)


# ─────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────

# Örnekleme hızı (Hz) — dtw_analysis.py ile uyumlu [1]
SAMPLE_RATE: int = 100

# Zero Velocity Update eşiği — dtw_analysis.py ile uyumlu [1]
ZUPT_THRESHOLD: float = 500.0

# Basınç eşiği: kalem kağıda değiyor mu?
# PRESSURE_THRESHOLD: float = 0.3  # manyometre kaldırıldı

# Yüksek geçiren filtre kesim frekansı (Hz)
HPF_CUTOFF: float = 0.5

# Butterworth filtre derecesi
HPF_ORDER: int = 2

# Medyan filtre çekirdek boyutu (gürültü azaltma)
MEDFILT_KERNEL: int = 5

# Spline yumuşatma faktörü (0 = tam enterpolasyon, büyük = daha pürüzsüz)
SPLINE_SMOOTHING_FACTOR: float = 0.5

# Vuruşlar arası X ekseni offset (normalize koordinatlarda)
STROKE_GAP_OFFSET: float = 0.0

# Görselleştirme renkleri
COLOR_REF: str = "#AAAAAA"       # Referans: gri
COLOR_USER: str = "#2196F3"      # Kullanıcı: mavi
COLOR_STROKE_LINE: str = "#1565C0"
COLOR_LIFT: str = "#FF9800"      # Kalem havada: turuncu kesikli
COLOR_START: str = "#4CAF50"     # Başlangıç noktası: yeşil
COLOR_END: str = "#F44336"       # Bitiş noktası: kırmızı

# Çizgi kalınlığı aralığı (basınca göre)
LINE_WIDTH_MIN: float = 0.5
LINE_WIDTH_MAX: float = 4.0


# ─────────────────────────────────────────────────────────────────────
# Sinyal Ön İşleme
# ─────────────────────────────────────────────────────────────────────

def _medyan_filtre(sinyal: np.ndarray, cekirdek: int = MEDFILT_KERNEL) -> np.ndarray:
    """Medyan filtre ile gürültü azaltma.

    Args:
        sinyal: 1D sinyal dizisi.
        cekirdek: Çekirdek boyutu (tek sayı olmalı).

    Returns:
        Filtrelenmiş sinyal.
    """
    if sinyal.size < cekirdek:
        return sinyal.copy()
    k = min(cekirdek, sinyal.size)
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return sinyal.copy()
    return medfilt(sinyal, kernel_size=k)


def _yuksek_geciren_filtre(
    sinyal: np.ndarray,
    kesim_frekans: float = HPF_CUTOFF,
    ornekleme_hz: int = SAMPLE_RATE,
    derece: int = HPF_ORDER,
) -> np.ndarray:
    """Butterworth yüksek geçiren filtre — düşük frekanslı drift'i kaldırır.

    Args:
        sinyal: 1D sinyal dizisi.
        kesim_frekans: Kesim frekansı (Hz).
        ornekleme_hz: Örnekleme hızı (Hz).
        derece: Filtre derecesi.

    Returns:
        Filtrelenmiş sinyal.
    """
    nyquist = ornekleme_hz / 2.0

    # Kesim frekansı Nyquist'e çok yakınsa veya sinyal çok kısaysa filtre uygulama
    if kesim_frekans >= nyquist * 0.95:
        return sinyal.copy()

    # filtfilt için minimum sinyal uzunluğu: 3 * (derece * 2 + 1) padlen
    min_uzunluk = 3 * (derece * 2 + 1) + 1
    if sinyal.size < min_uzunluk:
        return sinyal.copy()

    wn = kesim_frekans / nyquist
    b, a = butter(derece, wn, btype="high")
    return filtfilt(b, a, sinyal)


def _trend_cikar(sinyal: np.ndarray) -> np.ndarray:
    """Doğrusal trend çıkarma (scipy.signal.detrend).

    Args:
        sinyal: 1D sinyal dizisi.

    Returns:
        Trendi çıkarılmış sinyal.
    """
    if sinyal.size < 2:
        return sinyal.copy()
    return detrend(sinyal, type="linear")


def _ivme_on_isleme(
    ax: np.ndarray,
    ay: np.ndarray,
    ornekleme_hz: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """İvme kanallarına tüm ön işleme adımlarını uygular.

    Sıra:
        1. Medyan filtre (gürültü)
        2. Trend çıkarma (doğrusal drift)
        3. Yüksek geçiren filtre (düşük frekanslı drift)

    Args:
        ax: X ivme dizisi, shape (N,).
        ay: Y ivme dizisi, shape (N,).
        ornekleme_hz: Örnekleme hızı.

    Returns:
        (ax_temiz, ay_temiz): Ön işlenmiş ivme dizileri.
    """
    # Adım 1 — Medyan filtre
    ax_f = _medyan_filtre(ax)
    ay_f = _medyan_filtre(ay)

    # Adım 2 — Trend çıkarma
    ax_f = _trend_cikar(ax_f)
    ay_f = _trend_cikar(ay_f)

    # Adım 3 — Yüksek geçiren filtre
    ax_f = _yuksek_geciren_filtre(ax_f, ornekleme_hz=ornekleme_hz)
    ay_f = _yuksek_geciren_filtre(ay_f, ornekleme_hz=ornekleme_hz)

    return ax_f, ay_f


# ─────────────────────────────────────────────────────────────────────
# Vuruş (Stroke) Tespiti
# ─────────────────────────────────────────────────────────────────────

def _vuruslari_tespit_et(
    pressure: np.ndarray,
    # esik: float = PRESSURE_THRESHOLD,  # manyometre kaldırıldı
    min_uzunluk: int = 5,
) -> List[Tuple[int, int]]:
    """Tüm veriyi tek bir vuruş olarak döndürür.

    Not: Manyometre kaldırıldığı için basınç tabanlı tespit devre dışı.
    # ESKİ: Basınç > eşik olan ardışık bölgeler vuruş olarak tanımlanıyordu.

    Args:
        pressure: Basınç dizisi, shape (N,).
        # esik: Basınç eşik değeri (kullanılmıyor).
        min_uzunluk: Minimum vuruş uzunluğu (örnek sayısı).

    Returns:
        Vuruş aralıkları listesi: [(start_idx, end_idx)] (tek vuruş)
    """
    # Manyometre kaldırıldı — tüm veri tek vuruş:
    return [(0, len(pressure) - 1)]

    # --- ESKİ KOD: Basınç tabanlı vuruş tespiti ---
    # aktif = pressure > esik
    # vuruslar: List[Tuple[int, int]] = []
    # baslangic: Optional[int] = None
    # for i in range(len(aktif)):
    #     if aktif[i] and baslangic is None:
    #         baslangic = i
    #     elif not aktif[i] and baslangic is not None:
    #         if (i - 1 - baslangic) >= min_uzunluk:
    #             vuruslar.append((baslangic, i - 1))
    #         baslangic = None
    # if baslangic is not None:
    #     if (len(aktif) - 1 - baslangic) >= min_uzunluk:
    #         vuruslar.append((baslangic, len(aktif) - 1))
    # if not vuruslar:
    #     vuruslar = [(0, len(pressure) - 1)]
    # return vuruslar


# ─────────────────────────────────────────────────────────────────────
# Çift Entegrasyon — Tek Vuruş
# ─────────────────────────────────────────────────────────────────────

def _zupt_maske_olustur(
    ax: np.ndarray,
    ay: np.ndarray,
    esik: float = ZUPT_THRESHOLD,
    pencere: int = 5,
) -> np.ndarray:
    """Zero Velocity Update maskesi oluşturur.

    İvme büyüklüğü eşiğin altında olan bölgelerde hız sıfırlanmalıdır.
    Hareketli pencere ortalaması kullanılarak gürültüye karşı
    dayanıklılık sağlanır. Bu eşik dtw_analysis.py ile uyumludur [1].

    Args:
        ax: X ivme dizisi.
        ay: Y ivme dizisi.
        esik: ZUPT eşiği.
        pencere: Hareketli pencere boyutu.

    Returns:
        Boolean maske: True = durgun (hız sıfırlanmalı).
    """
    buyukluk = ax ** 2 + ay ** 2

    # Hareketli ortalama
    if buyukluk.size >= pencere:
        kernel = np.ones(pencere) / pencere
        buyukluk_yumusak = np.convolve(buyukluk, kernel, mode="same")
    else:
        buyukluk_yumusak = buyukluk

    return buyukluk_yumusak < esik


def _son_nokta_hiz_duzeltmesi(
    velocity: np.ndarray,
) -> np.ndarray:
    """Vuruş sonunda hız → 0 olacak şekilde doğrusal düzeltme uygular.

    Varsayım: vuruş başında hız zaten 0 (sıfırlanıyor).
    Vuruş sonunda kalan residual hızı doğrusal olarak dağıtarak kaldırır.

    Args:
        velocity: 1D hız dizisi, shape (N,).

    Returns:
        Düzeltilmiş hız dizisi.
    """
    n = velocity.size
    if n < 2:
        return velocity.copy()

    # Son noktadaki hız
    v_son = velocity[-1]

    # Doğrusal düzeltme: her noktada (i/N) * v_son çıkar
    duzeltme = np.linspace(0.0, v_son, n)
    return velocity - duzeltme


def _tek_vurus_entegrasyon(
    ax: np.ndarray,
    ay: np.ndarray,
    ornekleme_hz: int = SAMPLE_RATE,
    zupt_esik: float = ZUPT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tek bir vuruş için çift entegrasyon: ivme → hız → konum.

    İşlem sırası:
        1. İvme ön işleme (filtre + trend çıkarma + HPF)
        2. İvme → hız (cumulative_trapezoid)
        3. ZUPT: durgun bölgelerde hız = 0
        4. Son nokta hız düzeltmesi
        5. Hız → konum (cumulative_trapezoid)
        6. Konum trend çıkarma

    Bu fonksiyon dtw_analysis.py'deki calculate_2d_trajectory ile
    aynı mantığı izler: her vuruş başında velocity ve position
    sıfırlanır [1].

    Args:
        ax: X ivme dizisi, shape (M,).
        ay: Y ivme dizisi, shape (M,).
        ornekleme_hz: Örnekleme hızı (Hz).
        zupt_esik: Zero Velocity Update eşiği.

    Returns:
        (px, py): X ve Y konum dizileri, shape (M,).
    """
    n = ax.size
    if n < 2:
        return np.zeros(n), np.zeros(n)

    dt = 1.0 / ornekleme_hz

    # ── Ön işleme ──
    ax_temiz, ay_temiz = _ivme_on_isleme(ax, ay, ornekleme_hz)

    # ── ZUPT maskesi ──
    zupt_mask = _zupt_maske_olustur(ax_temiz, ay_temiz, esik=zupt_esik)

    # ── İvme → Hız (birinci entegrasyon) ──
    vx = np.concatenate([[0.0], cumulative_trapezoid(ax_temiz, dx=dt)])
    vy = np.concatenate([[0.0], cumulative_trapezoid(ay_temiz, dx=dt)])

    # ── ZUPT uygula: durgun bölgelerde hız sıfırla ──
    vx[zupt_mask] = 0.0
    vy[zupt_mask] = 0.0

    # ── Son nokta hız düzeltmesi ──
    vx = _son_nokta_hiz_duzeltmesi(vx)
    vy = _son_nokta_hiz_duzeltmesi(vy)

    # ── Hız → Konum (ikinci entegrasyon) ──
    px = np.concatenate([[0.0], cumulative_trapezoid(vx, dx=dt)])
    py = np.concatenate([[0.0], cumulative_trapezoid(vy, dx=dt)])

    # ── Konum trend çıkarma (kalan drift) ──
    px = _trend_cikar(px)
    py = _trend_cikar(py)

    return px, py


# ─────────────────────────────────────────────────────────────────────
# Çok Vuruşlu Yörünge Hesaplama
# ─────────────────────────────────────────────────────────────────────

def calculate_trajectory(
    data: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
    zupt_esik: float = ZUPT_THRESHOLD,
    # pressure_esik: float = PRESSURE_THRESHOLD,  # manyometre kaldırıldı
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """IMU verisini çift entegrasyonla 2D kalem izine dönüştürür.

    Not: Manyometre kaldırıldığı için tüm veri tek vuruş olarak kabul
    edilir ve tek parça olarak entegre edilir.

    Args:
        data: Sensör verisi, shape (N, 6+) veya (N, 7).
              Sütunlar: [ax, ay, az, gx, gy, gz, (pressure)]
        pressure: Basınç verisi, shape (N,). None ise ve data 7
                  sütunluysa son sütun kullanılır.
        sample_rate: Örnekleme hızı (Hz).
        zupt_esik: ZUPT eşiği.
        # pressure_esik: Vuruş tespiti için basınç eşiği (kullanılmıyor).

    Returns:
        Tuple:
            - px: X pozisyonları, shape (N,)
            - py: Y pozisyonları, shape (N,)
            - pressure_out: Basınç dizisi, shape (N,)
            - vuruslar: Vuruş aralıkları [(start, end), ...]
    """
    n = data.shape[0]

    # ── Basınç kanalını belirle ──
    if pressure is not None:
        basinc = pressure.copy()
    elif data.shape[1] >= 7:
        basinc = data[:, 6].copy()
    else:
        # Basınç yok — tüm veriyi tek vuruş olarak kabul et
        basinc = np.ones(n)

    # ── İvme kanalları ──
    ax_ham = data[:, 0].copy()
    ay_ham = data[:, 1].copy()

    # ── Vuruş tespiti (manyometre kaldırıldı — tek vuruş) ──
    vuruslar = _vuruslari_tespit_et(basinc)

    # ── Her vuruş için ayrı entegrasyon ──
    px_tam = np.zeros(n)
    py_tam = np.zeros(n)

    # Önceki vuruşun son noktası (birleştirme için offset)
    onceki_son_x = 0.0
    onceki_son_y = 0.0

    for v_idx, (v_bas, v_son) in enumerate(vuruslar):
        # Vuruş verisini kes
        ax_vurus = ax_ham[v_bas: v_son + 1]
        ay_vurus = ay_ham[v_bas: v_son + 1]

        if ax_vurus.size < 2:
            continue

        # Çift entegrasyon
        seg_px, seg_py = _tek_vurus_entegrasyon(
            ax_vurus, ay_vurus,
            ornekleme_hz=sample_rate,
            zupt_esik=zupt_esik,
        )

        # Önceki vuruşun son noktasına göre offset ekle
        if v_idx > 0:
            seg_px += onceki_son_x + STROKE_GAP_OFFSET
            seg_py += onceki_son_y
        # İlk vuruş — offset yok

        # Tam diziye yerleştir
        px_tam[v_bas: v_son + 1] = seg_px
        py_tam[v_bas: v_son + 1] = seg_py

        # Sonraki vuruş için son noktayı kaydet
        onceki_son_x = seg_px[-1]
        onceki_son_y = seg_py[-1]

    # ── Vuruşlar arası (kalem havada) bölgeleri enterpolasyon ile doldur ──
    px_tam, py_tam = _havada_enterpolasyon(px_tam, py_tam, vuruslar, n)

    return px_tam, py_tam, basinc, vuruslar


def _havada_enterpolasyon(
    px: np.ndarray,
    py: np.ndarray,
    vuruslar: List[Tuple[int, int]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vuruşlar arası boşlukları doğrusal enterpolasyonla doldurur.

    Kalem havadayken pozisyon, önceki vuruşun sonundan sonraki
    vuruşun başına doğru düzgün geçiş yapar.

    Args:
        px, py: Konum dizileri.
        vuruslar: Vuruş aralıkları.
        n: Toplam dizi uzunluğu.

    Returns:
        Güncellenmiş (px, py).
    """
    for i in range(len(vuruslar) - 1):
        _, onceki_son = vuruslar[i]
        sonraki_bas, _ = vuruslar[i + 1]

        if sonraki_bas <= onceki_son + 1:
            continue

        # Doğrusal enterpolasyon
        aralik = sonraki_bas - onceki_son
        px[onceki_son: sonraki_bas + 1] = np.linspace(
            px[onceki_son], px[sonraki_bas], aralik + 1
        )
        py[onceki_son: sonraki_bas + 1] = np.linspace(
            py[onceki_son], py[sonraki_bas], aralik + 1
        )

    return px, py


# ─────────────────────────────────────────────────────────────────────
# İz Yumuşatma
# ─────────────────────────────────────────────────────────────────────

def _iz_yumusat(
    x: np.ndarray,
    y: np.ndarray,
    smoothing: float = SPLINE_SMOOTHING_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spline ile iz yumuşatma uygular.

    UnivariateSpline kullanarak konum verisini pürüzsüzleştirir.
    Çok kısa segmentlerde yumuşatma atlanır.

    Args:
        x, y: Konum dizileri, shape (N,).
        smoothing: Yumuşatma faktörü.

    Returns:
        Yumuşatılmış (x_smooth, y_smooth).
    """
    n = x.size
    if n < 5:
        return x.copy(), y.copy()

    t = np.arange(n)

    try:
        # s parametresi: toplam hata karesi üst sınırı
        s_param = n * smoothing
        spline_x = UnivariateSpline(t, x, s=s_param, k=min(3, n - 1))
        spline_y = UnivariateSpline(t, y, s=s_param, k=min(3, n - 1))
        return spline_x(t), spline_y(t)
    except Exception:
        # Spline başarısızsa orijinali döndür
        return x.copy(), y.copy()


# ─────────────────────────────────────────────────────────────────────
# Normalizasyon
# ─────────────────────────────────────────────────────────────────────

def _normalize_koordinat(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Koordinatları [-1, 1] aralığına normalize eder.

    En-boy oranı korunur (en büyük eksen baz alınır).

    Args:
        x, y: Konum dizileri.

    Returns:
        Normalize edilmiş (x_norm, y_norm).
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    maks_range = max(x_range, y_range, 1e-9)

    # Merkeze al ve normalize et
    x_norm = (x - (x_min + x_max) / 2.0) / (maks_range / 2.0)
    y_norm = (y - (y_min + y_max) / 2.0) / (maks_range / 2.0)

    return x_norm, y_norm


# ─────────────────────────────────────────────────────────────────────
# Basınca Göre Çizgi Kalınlığı Hesaplama
# ─────────────────────────────────────────────────────────────────────

def _basinc_kalinlik(
    pressure: np.ndarray,
    min_kalinlik: float = LINE_WIDTH_MIN,
    maks_kalinlik: float = LINE_WIDTH_MAX,
) -> np.ndarray:
    """Sabit çizgi kalınlığı döndürür.

    Not: Manyometre kaldırıldığı için basınç bilgisinden kalınlık
    hesaplanmaz. Sabit ortalama kalınlık kullanılır.

    # ESKİ: Basınç verisini çizgi kalınlığına dönüştürür.

    Args:
        pressure: Basınç dizisi, shape (N,).
        min_kalinlik: Minimum çizgi kalınlığı.
        maks_kalinlik: Maksimum çizgi kalınlığı.

    Returns:
        Çizgi kalınlıkları, shape (N,).
    """
    # Manyometre kaldırıldı — sabit kalınlık döndür
    return np.full_like(pressure, (min_kalinlik + maks_kalinlik) / 2.0)

    # --- ESKİ KOD: Basınca göre kalınlık ---
    # p = pressure.copy()
    # p_min = p.min()
    # p_max = p.max()
    # if p_max - p_min < 1e-9:
    #     return np.full_like(p, (min_kalinlik + maks_kalinlik) / 2.0)
    # p_norm = (p - p_min) / (p_max - p_min)
    # return min_kalinlik + p_norm * (maks_kalinlik - min_kalinlik)


# ─────────────────────────────────────────────────────────────────────
# Değişken Kalınlıklı Çizgi Çizme (LineCollection)
# ─────────────────────────────────────────────────────────────────────

def _degisken_kalinlik_ciz(
    ax_plot: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    linewidths: np.ndarray,
    color: str = COLOR_STROKE_LINE,
    alpha: float = 1.0,
    linestyle: str = "solid",
) -> None:
    """Değişken çizgi kalınlığıyla eğri çizer (LineCollection).

    Args:
        ax_plot: Matplotlib Axes nesnesi.
        x, y: Konum dizileri.
        linewidths: Her segment için çizgi kalınlığı.
        color: Çizgi rengi.
        alpha: Saydamlık.
        linestyle: Çizgi stili.
    """
    if x.size < 2:
        return

    # Segment oluştur: [[p0, p1], [p1, p2], ...]
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Her segmentin kalınlığı: iki uç noktanın ortalaması
    lw = (linewidths[:-1] + linewidths[1:]) / 2.0

    lc = mcoll.LineCollection(
        segments,
        linewidths=lw,
        colors=color,
        alpha=alpha,
        linestyle=linestyle,
    )
    ax_plot.add_collection(lc)


# ─────────────────────────────────────────────────────────────────────
# Görselleştirme Fonksiyonları
# ─────────────────────────────────────────────────────────────────────

def plot_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    vuruslar: Optional[List[Tuple[int, int]]] = None,
    title: str = "",
    smoothing: float = SPLINE_SMOOTHING_FACTOR,
    figsize: Tuple[float, float] = (6, 6),
) -> Figure:
    """Kalem izini basınca göre değişen kalınlıkla çizer.

    Vuruş geçişleri (kalem havada) kesikli çizgiyle gösterilir.
    Başlangıç ve bitiş noktaları işaretlenir.

    Args:
        x, y: Konum dizileri, shape (N,).
        pressure: Basınç dizisi, shape (N,). None ise sabit kalınlık.
        vuruslar: Vuruş aralıkları. None ise tek vuruş varsayılır.
        title: Grafik başlığı.
        smoothing: Yumuşatma faktörü.
        figsize: Figure boyutu.

    Returns:
        matplotlib Figure nesnesi.
    """
    n = x.size

    # Varsayılanlar
    if pressure is None:
        pressure = np.ones(n)
    if vuruslar is None:
        vuruslar = [(0, n - 1)]

    # Yumuşatma
    x_s, y_s = _iz_yumusat(x, y, smoothing=smoothing)

    # Normalize
    x_n, y_n = _normalize_koordinat(x_s, y_s)

    # Kalınlık
    kalinlik = _basinc_kalinlik(pressure)

    # ── Figure oluştur ──
    fig, ax_plot = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    ax_plot.set_facecolor("#FAFAFA")

    # ── Vuruşları çiz ──
    for v_bas, v_son in vuruslar:
        if v_son - v_bas < 1:
            continue
        seg_x = x_n[v_bas: v_son + 1]
        seg_y = y_n[v_bas: v_son + 1]
        seg_lw = kalinlik[v_bas: v_son + 1]
        _degisken_kalinlik_ciz(
            ax_plot, seg_x, seg_y, seg_lw,
            color=COLOR_STROKE_LINE, alpha=0.85,
        )

    # ── Vuruşlar arası kesikli çizgi (kalem havada) ──
    for i in range(len(vuruslar) - 1):
        _, onceki_son = vuruslar[i]
        sonraki_bas, _ = vuruslar[i + 1]
        if sonraki_bas > onceki_son + 1:
            ax_plot.plot(
                [x_n[onceki_son], x_n[sonraki_bas]],
                [y_n[onceki_son], y_n[sonraki_bas]],
                color=COLOR_LIFT,
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
                label="Kalem havada" if i == 0 else None,
            )

    # ── Başlangıç / bitiş noktaları ──
    ax_plot.plot(x_n[0], y_n[0], "o",
                 color=COLOR_START, markersize=10, zorder=5,
                 label="Başlangıç")
    ax_plot.plot(x_n[-1], y_n[-1], "s",
                 color=COLOR_END, markersize=10, zorder=5,
                 label="Bitiş")

    # ── Eksen ayarları ──
    ax_plot.set_xlim(-1.3, 1.3)
    ax_plot.set_ylim(-1.3, 1.3)
    ax_plot.set_aspect("equal")
    ax_plot.invert_yaxis()  # Yazı koordinat sistemi: y aşağı pozitif
    ax_plot.grid(True, alpha=0.15)
    ax_plot.legend(loc="upper right", fontsize=8, framealpha=0.7)

    baslik = title if title else "Tahmini Kalem İzi"
    ax_plot.set_title(baslik, fontsize=14, fontweight="bold", pad=12)
    ax_plot.set_xlabel("X", fontsize=10)
    ax_plot.set_ylabel("Y", fontsize=10)

    # Uyarı notu
    fig.text(
        0.5, 0.01,
        "⚠ Bu iz yaklaşık bir tahmindir (çift entegrasyon drift'i nedeniyle).",
        ha="center", fontsize=7, color="#999999", style="italic",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def plot_comparison(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    user_x: np.ndarray,
    user_y: np.ndarray,
    score: float = 0.0,
    ref_title: str = "Referans",
    user_title: str = "Senin Yazın",
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Referans ve kullanıcı izini yan yana karşılaştırır.

    Args:
        ref_x, ref_y: Referans konum dizileri.
        user_x, user_y: Kullanıcı konum dizileri.
        score: Toplam kalite skoru (0-10).
        ref_title: Sol grafik başlığı.
        user_title: Sağ grafik başlığı.
        figsize: Figure boyutu.

    Returns:
        matplotlib Figure nesnesi.
    """
    fig, (ax_ref, ax_usr) = plt.subplots(
        1, 2, figsize=figsize, facecolor="white"
    )

    # Normalize
    rx, ry = _normalize_koordinat(ref_x, ref_y)
    ux, uy = _normalize_koordinat(user_x, user_y)

    # ── Referans ──
    ax_ref.plot(rx, ry, color=COLOR_REF, linewidth=2.5, solid_capstyle="round")
    ax_ref.plot(rx[0], ry[0], "o", color=COLOR_START, markersize=8)
    ax_ref.plot(rx[-1], ry[-1], "s", color=COLOR_END, markersize=8)
    ax_ref.set_title(ref_title, fontsize=13, fontweight="bold")
    ax_ref.set_xlim(-1.3, 1.3)
    ax_ref.set_ylim(-1.3, 1.3)
    ax_ref.set_aspect("equal")
    ax_ref.invert_yaxis()
    ax_ref.grid(True, alpha=0.15)

    # ── Kullanıcı ──
    ax_usr.plot(ux, uy, color=COLOR_USER, linewidth=2.5, solid_capstyle="round")
    ax_usr.plot(ux[0], uy[0], "o", color=COLOR_START, markersize=8)
    ax_usr.plot(ux[-1], uy[-1], "s", color=COLOR_END, markersize=8)
    ax_usr.set_title(user_title, fontsize=13, fontweight="bold")
    ax_usr.set_xlim(-1.3, 1.3)
    ax_usr.set_ylim(-1.3, 1.3)
    ax_usr.set_aspect("equal")
    ax_usr.invert_yaxis()
    ax_usr.grid(True, alpha=0.15)

    # ── Skor gösterimi ──
    skor_renk = _skor_rengi(score)
    fig.suptitle(
        f"Skor: {score:.1f} / 10",
        fontsize=16,
        fontweight="bold",
        color=skor_renk,
        y=1.02,
    )

    fig.tight_layout()
    return fig


def plot_overlay(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    user_x: np.ndarray,
    user_y: np.ndarray,
    title: str = "Üst Üste Karşılaştırma",
    figsize: Tuple[float, float] = (7, 7),
) -> Figure:
    """Referans ve kullanıcı izini üst üste bindirir.

    Referans gri, kullanıcı mavi olarak çizilir.
    Örtüşme alanları görsel olarak değerlendirilebilir.

    Args:
        ref_x, ref_y: Referans konum dizileri.
        user_x, user_y: Kullanıcı konum dizileri.
        title: Grafik başlığı.
        figsize: Figure boyutu.

    Returns:
        matplotlib Figure nesnesi.
    """
    fig, ax_plot = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    ax_plot.set_facecolor("#FAFAFA")

    # Normalize
    rx, ry = _normalize_koordinat(ref_x, ref_y)
    ux, uy = _normalize_koordinat(user_x, user_y)

    # ── Referans (arka plan, kalın, gri) ──
    ax_plot.plot(
        rx, ry,
        color=COLOR_REF, linewidth=5.0, solid_capstyle="round",
        alpha=0.5, label="Referans", zorder=1,
    )

    # ── Kullanıcı (ön plan, mavi) ──
    ax_plot.plot(
        ux, uy,
        color=COLOR_USER, linewidth=2.5, solid_capstyle="round",
        alpha=0.85, label="Senin Yazın", zorder=2,
    )

    # ── Başlangıç / bitiş ──
    ax_plot.plot(rx[0], ry[0], "o", color=COLOR_START, markersize=12,
                 zorder=3, alpha=0.4)
    ax_plot.plot(ux[0], uy[0], "o", color=COLOR_START, markersize=8,
                 zorder=4)
    ax_plot.plot(ux[-1], uy[-1], "s", color=COLOR_END, markersize=8,
                 zorder=4)

    ax_plot.set_xlim(-1.3, 1.3)
    ax_plot.set_ylim(-1.3, 1.3)
    ax_plot.set_aspect("equal")
    ax_plot.invert_yaxis()
    ax_plot.grid(True, alpha=0.15)
    ax_plot.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax_plot.set_title(title, fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def _skor_rengi(skor: float) -> str:
    """Skora göre renk döndürür (görsel geri bildirim).

    Args:
        skor: 0-10 arası skor.

    Returns:
        Hex renk kodu.
    """
    if skor >= 8.0:
        return "#4CAF50"   # Yeşil
    elif skor >= 6.0:
        return "#FF9800"   # Turuncu
    elif skor >= 4.0:
        return "#FF5722"   # Koyu turuncu
    else:
        return "#F44336"   # Kırmızı


# ─────────────────────────────────────────────────────────────────────
# Referans Harf Şablonları
# ─────────────────────────────────────────────────────────────────────

# Basit harfler için sabit koordinat dizileri (normalize [-1, 1])
# Bu şablonlar, gerçek referans verileri olmadığında fallback olarak
# kullanılır.
_SABLONLAR: Dict[str, Dict[str, Any]] = {
    "I": {
        "vuruslar": [
            # Üst çizgi (yatay)
            {"x": np.array([-0.3, 0.3]), "y": np.array([-0.9, -0.9])},
            # Dikey gövde
            {"x": np.array([0.0, 0.0]), "y": np.array([-0.9, 0.9])},
            # Alt çizgi (yatay)
            {"x": np.array([-0.3, 0.3]), "y": np.array([0.9, 0.9])},
        ],
    },
    "L": {
        "vuruslar": [
            # Dikey gövde
            {"x": np.array([-0.3, -0.3]), "y": np.array([-0.9, 0.9])},
            # Alt yatay
            {"x": np.array([-0.3, 0.6]), "y": np.array([0.9, 0.9])},
        ],
    },
    "O": {
        "vuruslar": [
            # Daire (parametrik)
            {
                "x": 0.7 * np.cos(np.linspace(0, 2 * np.pi, 60)),
                "y": 0.85 * np.sin(np.linspace(0, 2 * np.pi, 60)),
            },
        ],
    },
    "C": {
        "vuruslar": [
            # Yarım daire (açık sağ taraf)
            {
                "x": 0.7 * np.cos(np.linspace(0.4 * np.pi, 1.6 * np.pi, 40)),
                "y": 0.85 * np.sin(np.linspace(0.4 * np.pi, 1.6 * np.pi, 40)),
            },
        ],
    },
    "V": {
        "vuruslar": [
            # Sol çizgi (aşağı)
            {"x": np.array([-0.6, 0.0]), "y": np.array([-0.9, 0.9])},
            # Sağ çizgi (yukarı)
            {"x": np.array([0.0, 0.6]), "y": np.array([0.9, -0.9])},
        ],
    },
    "T": {
        "vuruslar": [
            # Üst yatay çizgi
            {"x": np.array([-0.6, 0.6]), "y": np.array([-0.9, -0.9])},
            # Dikey gövde
            {"x": np.array([0.0, 0.0]), "y": np.array([-0.9, 0.9])},
        ],
    },
    "A": {
        "vuruslar": [
            # Sol bacak
            {"x": np.array([-0.6, 0.0]), "y": np.array([0.9, -0.9])},
            # Sağ bacak
            {"x": np.array([0.0, 0.6]), "y": np.array([-0.9, 0.9])},
            # Orta çizgi
            {"x": np.array([-0.3, 0.3]), "y": np.array([0.1, 0.1])},
        ],
    },
    "X": {
        "vuruslar": [
            # Sol üst → sağ alt
            {"x": np.array([-0.6, 0.6]), "y": np.array([-0.9, 0.9])},
            # Sağ üst → sol alt
            {"x": np.array([0.6, -0.6]), "y": np.array([-0.9, 0.9])},
        ],
    },
    "Z": {
        "vuruslar": [
            # Üst yatay
            {"x": np.array([-0.6, 0.6]), "y": np.array([-0.9, -0.9])},
            # Çapraz
            {"x": np.array([0.6, -0.6]), "y": np.array([-0.9, 0.9])},
            # Alt yatay
            {"x": np.array([-0.6, 0.6]), "y": np.array([0.9, 0.9])},
        ],
    },
}


def get_reference_shape(
    letter: str,
    ref_data: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Referans harf şeklinin (x, y) koordinatlarını döndürür.

    Öncelik sırası:
        1. ref_data verilmişse → onun trajectory'sini hesapla
        2. Sabit şablonlarda varsa → şablonu kullan
        3. Hiçbiri yoksa → boş dizi döndür

    Referans verisinden trajectory hesaplama, dtw_analysis.py'deki
    calculate_2d_trajectory ile aynı prensiptedir [1].

    Args:
        letter: Harf karakteri (büyük harf, ör. "A").
        ref_data: Referans IMU verisi, shape (N, 6+). Opsiyonel.
        sample_rate: Örnekleme hızı.

    Returns:
        (ref_x, ref_y): Referans koordinat dizileri.
    """
    harf = letter.upper().strip()

    # ── 1) Referans verisinden hesapla ──
    if ref_data is not None and ref_data.size > 0:
        try:
            px, py, _, _ = calculate_trajectory(
                ref_data, sample_rate=sample_rate
            )
            return px, py
        except Exception:
            pass  # Hesaplama başarısızsa şablona düş

    # ── 2) Sabit şablon ──
    if harf in _SABLONLAR:
        sablon = _SABLONLAR[harf]
        tum_x: List[np.ndarray] = []
        tum_y: List[np.ndarray] = []

        for vurus in sablon["vuruslar"]:
            # Her vuruşu 30 noktaya enterpolasyon ile genişlet
            vx = vurus["x"]
            vy = vurus["y"]

            if vx.size < 2:
                tum_x.append(vx)
                tum_y.append(vy)
                continue

            t_orijinal = np.linspace(0, 1, vx.size)
            t_yeni = np.linspace(0, 1, max(30, vx.size))
            vx_interp = np.interp(t_yeni, t_orijinal, vx)
            vy_interp = np.interp(t_yeni, t_orijinal, vy)

            tum_x.append(vx_interp)
            tum_y.append(vy_interp)

        if tum_x:
            return np.concatenate(tum_x), np.concatenate(tum_y)

    # ── 3) Bulunamadı ──
    return np.array([]), np.array([])


def list_available_templates() -> List[str]:
    """Mevcut sabit şablon harflerini listeler.

    Returns:
        Harf listesi (sıralı).
    """
    return sorted(_SABLONLAR.keys())


# ─────────────────────────────────────────────────────────────────────
# Referans dosyasından otomatik yörünge hesaplama
# ─────────────────────────────────────────────────────────────────────

def load_and_compute_reference_trajectory(
    letter: str,
    db_path: str = "letters_db",
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Referans veritabanından harf yörüngesini yükler ve hesaplar.

    letters_db/<HARF>/ref_01.npy dosyasını arar. Bulamazsa
    sabit şablona düşer. Bu veritabanı yapısı dtw_analysis.py
    ile aynıdır [1].

    Args:
        letter: Harf karakteri.
        db_path: Veritabanı dizin yolu.
        sample_rate: Örnekleme hızı.

    Returns:
        (ref_x, ref_y) koordinat dizileri.
    """
    harf = letter.upper().strip()
    harf_dizin = os.path.join(db_path, harf)

    if os.path.isdir(harf_dizin):
        dosyalar = sorted(glob.glob(os.path.join(harf_dizin, "ref_*.npy")))
        if dosyalar:
            try:
                ref_data = np.load(dosyalar[0])
                px, py, _, _ = calculate_trajectory(
                    ref_data, sample_rate=sample_rate
                )
                return px, py
            except Exception:
                pass

    # Veritabanında bulunamadı — şablona düş
    return get_reference_shape(harf)


# ─────────────────────────────────────────────────────────────────────
# Tam pipeline: veri → görsel
# ─────────────────────────────────────────────────────────────────────

def process_and_visualize(
    user_data: np.ndarray,
    letter: str = "",
    ref_data: Optional[np.ndarray] = None,
    score: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Tam pipeline: kullanıcı verisinden iz hesapla ve görselleştir.

    Args:
        user_data: Kullanıcı IMU verisi, shape (N, 6+).
        letter: Hedef harf (opsiyonel).
        ref_data: Referans IMU verisi (opsiyonel).
        score: Kalite skoru (scoring.py'den).
        sample_rate: Örnekleme hızı.
        output_dir: Grafiklerin kaydedileceği dizin. None ise kaydetmez.

    Returns:
        Dict:
        {
            "user_trajectory": (px, py),
            "ref_trajectory": (ref_x, ref_y) veya None,
            "vuruslar": List[Tuple[int, int]],
            "figures": {
                "trajectory": Figure,
                "comparison": Figure veya None,
                "overlay": Figure veya None,
            },
        }
    """
    # ── Kullanıcı yörüngesi ──
    px, py, basinc, vuruslar = calculate_trajectory(
        user_data, sample_rate=sample_rate
    )

    sonuc: Dict[str, Any] = {
        "user_trajectory": (px, py),
        "ref_trajectory": None,
        "vuruslar": vuruslar,
        "figures": {},
    }

    # ── Kullanıcı izi grafiği ──
    baslik = f'Harf: "{letter.upper()}"' if letter else "Tahmini Kalem İzi"
    fig_traj = plot_trajectory(px, py, basinc, vuruslar, title=baslik)
    sonuc["figures"]["trajectory"] = fig_traj

    # ── Referans varsa karşılaştırma ──
    ref_x: Optional[np.ndarray] = None
    ref_y: Optional[np.ndarray] = None

    if ref_data is not None and ref_data.size > 0:
        ref_x, ref_y = get_reference_shape(letter, ref_data, sample_rate)
    elif letter:
        ref_x, ref_y = get_reference_shape(letter)

    if ref_x is not None and ref_x.size > 0:
        sonuc["ref_trajectory"] = (ref_x, ref_y)

        fig_comp = plot_comparison(ref_x, ref_y, px, py, score=score)
        sonuc["figures"]["comparison"] = fig_comp

        fig_over = plot_overlay(ref_x, ref_y, px, py)
        sonuc["figures"]["overlay"] = fig_over
    else:
        sonuc["figures"]["comparison"] = None
        sonuc["figures"]["overlay"] = None

    # ── Dosyaya kaydet (opsiyonel) ──
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        harf_tag = letter.upper() if letter else "unknown"

        for anahtar, fig in sonuc["figures"].items():
            if fig is not None:
                dosya_adi = f"{harf_tag}_{anahtar}.png"
                yol = os.path.join(output_dir, dosya_adi)
                fig.savefig(yol, dpi=150, bbox_inches="tight")

    return sonuc


# ─────────────────────────────────────────────────────────────────────
# Modül testi
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TRAJECTORY.PY — Modül Testi")
    print("=" * 60)

    np.random.seed(42)
    N = 300  # örnek sayısı
    t = np.linspace(0, 3 * np.pi, N)

    # Sentetik "L" harfi:
    #   Vuruş 1 (dikey): aşağı ivme → t ∈ [0, π]
    #   Kalem havada: t ∈ (π, 1.2π)
    #   Vuruş 2 (yatay): sağa ivme → t ∈ [1.2π, 2.5π]
    ax_sim = np.zeros(N)
    ay_sim = np.zeros(N)
    pressure_sim = np.zeros(N)

    # Vuruş 1: dikey aşağı
    v1_end = int(N * 0.35)
    ay_sim[:v1_end] = 2.0 + np.random.normal(0, 0.1, v1_end)
    pressure_sim[:v1_end] = 0.8

    # Kalem havada
    gap_end = int(N * 0.42)

    # Vuruş 2: yatay sağa
    ax_sim[gap_end:] = 2.0 + np.random.normal(0, 0.1, N - gap_end)
    # pressure_sim[gap_end:] = 0.7  # MANYOMETRE KALDIRILDI

    # Veri matrisi oluştur (N, 6)
    data = np.column_stack([
        ax_sim,                          # ax
        ay_sim,                          # ay
        np.zeros(N),                     # az
        np.random.normal(0, 0.05, N),    # gx
        np.random.normal(0, 0.05, N),    # gy
        np.zeros(N),                     # gz
        # pressure_sim,                    # pressure — MANYOMETRE KALDIRILDI
    ])

    print(f"\n  Sentetik veri: shape = {data.shape}")
    print(f"  Örnekleme hızı: {SAMPLE_RATE} Hz")

    # ── Yörünge hesapla ──
    px, py, basinc, vuruslar = calculate_trajectory(data)
    print(f"\n  Tespit edilen vuruş sayısı: {len(vuruslar)}")
    for i, (vb, vs) in enumerate(vuruslar):
        print(f"    Vuruş {i + 1}: [{vb}, {vs}] ({vs - vb + 1} örnek)")

    print(f"\n  Konum aralığı X: [{px.min():.3f}, {px.max():.3f}]")
    print(f"  Konum aralığı Y: [{py.min():.3f}, {py.max():.3f}]")

    # ── Mevcut şablonlar ──
    sablonlar = list_available_templates()
    print(f"\n  Mevcut şablonlar: {', '.join(sablonlar)}")

    # ── Referans şekli ──
    ref_x, ref_y = get_reference_shape("L")
    print(f"  'L' şablonu: {ref_x.size} nokta")

    # ── Tam pipeline ──
    sonuc = process_and_visualize(
        user_data=data,
        letter="L",
        score=7.5,
        output_dir="test_output",
    )

    print(f"\n  Üretilen grafikler:")
    for anahtar, fig in sonuc["figures"].items():
        durum = "✅" if fig is not None else "⬜ (referans yok)"
        print(f"    {anahtar}: {durum}")

    # Grafikleri kapat (bellek)
    for fig in sonuc["figures"].values():
        if fig is not None:
            plt.close(fig)

    print("\n" + "=" * 60)
    print("  Test tamamlandı! Grafikler 'test_output/' dizinine kaydedildi.")
    print("=" * 60)