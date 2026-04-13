"""
sensor_data.py — Arduino Micro/Nano + MPU6500 Seri Port Veri Toplama Modülü

Buton tabanlı kayıt: Butona basıldığında kayıt başlar, bırakıldığında durur.
Manyometre (FSR) kaldırılmıştır.
Veri formatı: (N, 7) → [ax, ay, az, gx, gy, gz, button]
"""

import serial
import serial.tools.list_ports
import time
import numpy as np
from typing import Optional, Callable, Tuple, Dict

# ── Sabitler ──
BAUD_RATE = 115200
BOOT_BEKLEME = 2.0          # Arduino boot süresi (saniye)
KAYIT_MAKS_SURE = 10.0      # Maksimum kayıt süresi (saniye)
KAYIT_MIN_SURE = 0.3        # Minimum geçerli kayıt süresi (saniye)
BEKLEME_TIMEOUT = 60.0      # Buton bekleme zaman aşımı (saniye)
ORNEKLEME_HIZI = 100        # Hz
BEKLENEN_KANAL = 7          # ax, ay, az, gx, gy, gz, button


def find_arduino_port() -> str:
    """Arduino'yu otomatik algılar veya kullanıcıdan manuel giriş ister.

    Returns:
        str: Bulunan veya girilen seri port adı.
    """
    portlar = list(serial.tools.list_ports.comports())

    # Arduino Micro / Nano VID/PID kontrol
    arduino_vidpid = [
        "2341:8037",  # Arduino Micro
        "2341:0043",  # Arduino UNO
        "2341:0001",  # Arduino Nano (orijinal)
        "1A86:7523",  # CH340 (klon Arduino)
        "0403:6001",  # FTDI (klon Arduino)
    ]

    for port in portlar:
        hwid_upper = port.hwid.upper()
        for vidpid in arduino_vidpid:
            if vidpid in hwid_upper:
                print(f"[OK] Arduino bulundu: {port.device} — {port.description}")
                return port.device

    # Otomatik bulunamadı
    print("\n[UYARI] Arduino otomatik algılanamadı!")
    if portlar:
        print("Mevcut seri portlar:")
        for i, port in enumerate(portlar):
            print(f"  [{i}] {port.device} — {port.description}")
    else:
        print("Hiçbir seri port bulunamadı. Arduino bağlı mı?")

    manuel_port = input("\nSeri port adını girin (örn: COM3): ").strip()
    return manuel_port


def connect_serial(port: Optional[str] = None) -> serial.Serial:
    """Seri bağlantı açar ve Arduino'nun hazır olmasını bekler.

    Args:
        port: Seri port adı. None ise otomatik algılama yapılır.

    Returns:
        serial.Serial: Hazır seri bağlantı nesnesi.

    Raises:
        ConnectionError: Bağlantı kurulamazsa.
    """
    if port is None:
        port = find_arduino_port()

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"[OK] {port} portuna bağlanıldı.")
    except Exception as e:
        raise ConnectionError(
            f"Seri port açılamadı: {port}\n"
            f"Hata: {e}\n"
            f"Arduino IDE Serial Monitor açıksa kapatın!"
        )

    # Arduino boot/kalibrasyon bekle
    print(f"[...] Arduino hazırlanıyor ({BOOT_BEKLEME}s)...")
    time.sleep(BOOT_BEKLEME)
    ser.flushInput()

    # İlk birkaç satırı oku (boot mesajları)
    for _ in range(20):
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  Arduino: {line}")
        except Exception:
            pass

    ser.flushInput()
    return ser


def calibrate(
    duration: float = 3.0,
    port: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, float]:
    """Kalibrasyon yapar — kalem sabitken sensör offset'lerini hesaplar.

    Args:
        duration: Kalibrasyon süresi (saniye).
        port: Seri port adı.
        progress_callback: İlerleme geri çağırma fonksiyonu.

    Returns:
        dict: Her eksen için offset değerleri.
    """
    ser = connect_serial(port)
    veriler = []

    print(f"\n[KALİBRASYON] Kalemi sabit tutun ({duration} saniye)...")
    baslangic = time.time()

    while (time.time() - baslangic) < duration:
        if progress_callback:
            kalan = duration - (time.time() - baslangic)
            progress_callback(max(0.0, kalan))

        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            values = [float(x) for x in line.split(",")]
            if len(values) == BEKLENEN_KANAL:
                veriler.append(values[:6])  # Sadece IMU kanalları
        except (ValueError, UnicodeDecodeError):
            pass

    ser.close()

    if len(veriler) < 10:
        print("[UYARI] Yeterli kalibrasyon verisi toplanamadı!")
        return {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}

    arr = np.array(veriler)
    offsets = {
        "ax": float(np.mean(arr[:, 0])),
        "ay": float(np.mean(arr[:, 1])),
        "az": float(np.mean(arr[:, 2])),
        "gx": float(np.mean(arr[:, 3])),
        "gy": float(np.mean(arr[:, 4])),
        "gz": float(np.mean(arr[:, 5])),
    }

    print("[OK] Kalibrasyon tamamlandı:")
    for k, v in offsets.items():
        print(f"  {k}: {v:.1f}")

    return offsets


def get_sensor_data(
    is_reference: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
    port: Optional[str] = None,
    ser: Optional[serial.Serial] = None,
) -> np.ndarray:
    """Buton tabanlı veri toplama — butona basılınca kayıt başlar, bırakılınca biter.

    Arduino'dan gelen son sütun buton durumudur (1=basılı, 0=bırakılmış).
    Buton basılı olduğu süre boyunca veri kaydedilir.

    Args:
        is_reference: True ise referans kaydı yapılıyor (bilgi amaçlı).
        progress_callback: İlerleme callback'i — kalan süreyi alır.
        port: Seri port adı.
        ser: Zaten açılmış seri bağlantı. None ise yeni bağlantı açılır.

    Returns:
        np.ndarray: Sensör verisi, shape (N, 6) — [ax, ay, az, gx, gy, gz].
                    Buton basılı olduğu süredeki veriler.
                    Veri toplanamadıysa shape (0, 6) döner.
    """
    kendi_baglantisi = False
    if ser is None:
        ser = connect_serial(port)
        kendi_baglantisi = True

    ser.flushInput()
    data = []
    kayit_aktif = False
    kayit_baslangic = 0.0
    bekleme_baslangic = time.time()

    mod = "referans" if is_reference else "deneme"
    print(f"\n[KAYIT] Butona basarak {mod} kaydını başlatın...")

    try:
        while True:
            # Zaman aşımı kontrolü (buton bekleme)
            if not kayit_aktif and (time.time() - bekleme_baslangic) > BEKLEME_TIMEOUT:
                print("[UYARI] Zaman aşımı — buton basılmadı.")
                break

            # Maksimum kayıt süresi kontrolü
            if kayit_aktif and (time.time() - kayit_baslangic) > KAYIT_MAKS_SURE:
                print(f"[BİLGİ] Maksimum süre ({KAYIT_MAKS_SURE}s) doldu.")
                break

            # Progress callback
            if progress_callback and kayit_aktif:
                gecen = time.time() - kayit_baslangic
                progress_callback(gecen)

            # Seri veri oku
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                values = [float(x) for x in line.split(",")]
                if len(values) != BEKLENEN_KANAL:
                    continue

                buton = int(values[6])  # Son sütun: buton durumu
                imu = values[:6]        # İlk 6 sütun: IMU verileri

                # Buton basıldı → kayıt başla
                if buton == 1 and not kayit_aktif:
                    kayit_aktif = True
                    kayit_baslangic = time.time()
                    data = []
                    print("[REC ●] Kayıt başladı — yazın!")

                # Kayıt aktifken veri topla
                if kayit_aktif and buton == 1:
                    data.append(imu)

                # Buton bırakıldı → kayıt dur
                if buton == 0 and kayit_aktif:
                    sure = time.time() - kayit_baslangic
                    if sure >= KAYIT_MIN_SURE:
                        print(f"[STOP ■] Kayıt bitti — {len(data)} örnek, {sure:.1f}s")
                        break
                    else:
                        # Çok kısa basım — devam et
                        print(f"[...] Çok kısa ({sure:.2f}s), basmaya devam edin...")
                        kayit_aktif = False
                        data = []

            except (ValueError, UnicodeDecodeError):
                continue

    except KeyboardInterrupt:
        print("\n[İPTAL] Kullanıcı tarafından iptal edildi.")

    finally:
        if kendi_baglantisi:
            ser.close()

    if len(data) == 0:
        print("[UYARI] Veri toplanamadı!")
        return np.zeros((0, 6))

    result = np.array(data)
    print(f"[OK] {result.shape[0]} örnek toplandı, shape: {result.shape}")
    return result