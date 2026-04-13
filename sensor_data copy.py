"""
sensor_data.py — Arduino Micro + MPU6500 + FSR 402 Seri Port Veri Toplama Modülü

Bu modül USB seri port üzerinden Arduino'dan gelen IMU (ivmeölçer + jiroskop)
ve basınç sensörü verilerini toplar. Otomatik port algılama, kalibrasyon ve
basınç eşik tabanlı akıllı kayıt sistemi içerir.
"""

import serial
import serial.tools.list_ports
import time
import numpy as np
from typing import Optional, Callable, Tuple, Dict, List

# ==========================================
# DONANIM YAPILKANDIRMASI
# ==========================================

BAUD_RATE: int = 115200           # Arduino seri haberleşme hızı
SERIAL_TIMEOUT: float = 1.0      # Seri port okuma zaman aşımı (saniye)

# Arduino Micro USB tanımlayıcıları
ARDUINO_VID: int = 0x2341        # Arduino USB Vendor ID
ARDUINO_PIDS: list = [0x8037, 0x0037]  # Arduino Micro Product ID'leri (bootloader + normal)

# ==========================================
# KAYIT PARAMETRELERİ
# ==========================================

PRESSURE_THRESHOLD: int = 50     # Basınç eşik değeri (0-1023 aralığında)
MIN_STROKE_DURATION: float = 0.8 # Minimum çizgi süresi (saniye) — bundan kısa çizgiler yok sayılır
MAX_RECORD_TIME: float = 10.0    # Maksimum kayıt süresi (saniye) — güvenlik limiti
WAIT_TIMEOUT: float = 30.0       # Kayıt başlaması için maksimum bekleme süresi (saniye)
NUM_COLUMNS: int = 7             # Beklenen sütun sayısı: ax, ay, az, gx, gy, gz, pressure


def find_arduino_port() -> str:
    """Arduino Micro'nun bağlı olduğu seri portu otomatik algılar.

    Arduino Micro'nun USB VID/PID değerlerini kullanarak sisteme bağlı
    tüm seri portları tarar. Bulunamazsa kullanıcıdan manuel giriş ister.

    Returns:
        str: Bulunan seri port adı (örn: '/dev/ttyACM0' veya 'COM3')
    """
    print("[Donanım] Arduino Micro aranıyor...")
    portlar = serial.tools.list_ports.comports()

    for port in portlar:
        # VID ve PID kontrolü
        if port.vid == ARDUINO_VID and port.pid in ARDUINO_PIDS:
            print(f"[Donanım] Arduino Micro bulundu: {port.device}")
            print(f"  → Açıklama: {port.description}")
            return port.device

    # Otomatik bulunamadı — mevcut portları listele ve kullanıcıya sor
    print("\n[UYARI] Arduino Micro otomatik algılanamadı!")
    if portlar:
        print("Mevcut seri portlar:")
        for i, port in enumerate(portlar):
            print(f"  [{i}] {port.device} — {port.description}")
    else:
        print("Hiçbir seri port bulunamadı. Arduino'nun bağlı olduğundan emin olun.")

    manuel_port = input("\nSeri port adını girin (örn: /dev/ttyACM0 veya COM3): ").strip()
    return manuel_port


def connect_serial(port: Optional[str] = None) -> serial.Serial:
    """Belirtilen porta seri bağlantı açar ve Arduino'nun hazır olmasını bekler.

    Bağlantı açıldıktan sonra giriş buffer'ını temizler ve Arduino'nun
    boot/kalibrasyon mesajlarının geçmesini bekler.

    Args:
        port: Bağlanılacak seri port adı. None ise otomatik algılama yapılır.

    Returns:
        serial.Serial: Açık ve hazır seri bağlantı nesnesi.

    Raises:
        ConnectionError: Bağlantı kurulamazsa.
    """
    if port is None:
        port = find_arduino_port()

    print(f"\n[Donanım] {port} portuna bağlanılıyor (baud: {BAUD_RATE})...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    except serial.SerialException as e:
        raise ConnectionError(
            f"HATA: {port} portuna bağlanılamadı!\n"
            f"  → Sebep: {e}\n"
            f"  → Arduino IDE Serial Monitor açıksa kapatın.\n"
            f"  → Arduino'nun USB ile bağlı olduğunu kontrol edin."
        )

    # Arduino Micro reset sonrası boot süresi için bekle
    time.sleep(2.0)
    ser.reset_input_buffer()

    # Arduino'nun boot mesajlarını oku ve atla (KALIBRASYON, SISTEM HAZIR vb.)
    print("[Donanım] Arduino'nun hazır olması bekleniyor...")
    hazir = False
    bekleme_baslangic = time.time()
    while time.time() - bekleme_baslangic < 10.0:
        try:
            satir = ser.readline().decode('utf-8', errors='ignore').strip()
            if satir:
                print(f"  Arduino → {satir}")
                # "SISTEM HAZIR" veya "HAZIR" mesajını gördüğümüzde devam et
                if "HAZIR" in satir.upper():
                    hazir = True
                    break
        except Exception:
            pass

    if not hazir:
        print("[UYARI] Arduino 'HAZIR' mesajı göndermedi, yine de devam ediliyor.")

    # Buffer'ı son kez temizle
    ser.reset_input_buffer()
    print("[Donanım] Bağlantı hazır ✓")
    return ser


def parse_line(satir: str) -> Optional[List[float]]:
    """Seri porttan gelen tek bir satırı parse eder.

    Arduino'dan gelen CSV formatındaki satırı (ax,ay,az,gx,gy,gz,pressure)
    float listesine çevirir. Hatalı satırları sessizce atlar.

    Args:
        satir: Arduino'dan gelen ham metin satırı.

    Returns:
        Optional[List[float]]: 7 elemanlı float listesi, hatalıysa None.
    """
    try:
        parcalar = satir.strip().split(',')
        if len(parcalar) != NUM_COLUMNS:
            return None
        degerler = [float(x) for x in parcalar]
        return degerler
    except (ValueError, UnicodeDecodeError):
        return None


def calibrate(
    duration: float = 3.0,
    ser: Optional[serial.Serial] = None,
    port: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Dict[str, float]:
    """Sensör kalibrasyonu yapar — kalemi sabit tutarak offset değerlerini hesaplar.

    Belirtilen süre boyunca sensörden veri toplar ve her eksenin ortalamasını
    hesaplayarak offset (sapma) değerlerini döndürür. Bu offset'ler sonraki
    tüm okumalardan çıkarılarak sıfır noktası düzeltmesi yapılır.

    Bağımsız olarak çağrılabilir (port otomatik algılanır) veya mevcut bir
    seri bağlantı nesnesiyle kullanılabilir.

    Args:
        duration: Kalibrasyon süresi (saniye). Varsayılan: 3.0
        ser: Açık seri bağlantı nesnesi. None ise yeni bağlantı açılır.
        port: Seri port adı (ser=None ise). None ise otomatik algılama yapılır.
        progress_callback: GUI entegrasyonu için ilerleme callback'i.
                          Kalan süreyi (float) parametre olarak alır.

    Returns:
        Dict[str, float]: Her eksen için offset değerleri.
            Anahtarlar: 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pressure'
    """
    # Seri bağlantı yoksa yeni aç
    baglanti_bizde = False
    if ser is None:
        ser = connect_serial(port)
        baglanti_bizde = True  # Bu fonksiyon açtı, bu fonksiyon kapatacak

    print(f"\n[Kalibrasyon] Kalemi/sensörü SABİT TUTUN! ({duration} saniye)")
    print("[Kalibrasyon] Veri toplanıyor...")

    eksen_isimleri = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pressure']
    toplam = {eksen: 0.0 for eksen in eksen_isimleri}
    okuma_sayisi = 0

    ser.reset_input_buffer()
    baslangic = time.time()

    try:
        while (time.time() - baslangic) < duration:
            # İlerleme callback'i çağır
            if progress_callback:
                kalan = duration - (time.time() - baslangic)
                progress_callback(max(0.0, kalan))

            try:
                ham_satir = ser.readline().decode('utf-8', errors='ignore').strip()
                degerler = parse_line(ham_satir)

                if degerler is not None:
                    for i, eksen in enumerate(eksen_isimleri):
                        toplam[eksen] += degerler[i]
                    okuma_sayisi += 1
            except Exception:
                continue
    finally:
        # Sadece bu fonksiyon açtıysa kapat
        if baglanti_bizde:
            ser.close()
            print("[Donanım] Seri port kapatıldı.")

    # Ortalama hesapla
    if okuma_sayisi == 0:
        print("[UYARI] Kalibrasyon sırasında hiç veri alınamadı!")
        return {eksen: 0.0 for eksen in eksen_isimleri}

    offsets = {eksen: toplam[eksen] / okuma_sayisi for eksen in eksen_isimleri}

    print(f"[Kalibrasyon] Tamamlandı! ({okuma_sayisi} okuma)")
    print(f"  → Offset'ler: ax={offsets['ax']:.1f}, ay={offsets['ay']:.1f}, "
          f"az={offsets['az']:.1f}, gx={offsets['gx']:.1f}, "
          f"gy={offsets['gy']:.1f}, gz={offsets['gz']:.1f}")
    print(f"  → Basınç sıfır noktası: {offsets['pressure']:.1f}")

    return offsets


def get_sensor_data(
    is_reference: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
    port: Optional[str] = None,
    threshold: int = PRESSURE_THRESHOLD,
    kalibrasyon_yap: bool = False,
    kalibrasyon_suresi: float = 3.0
) -> Tuple[np.ndarray, Dict]:
    """Arduino'dan IMU ve basınç sensörü verilerini akıllı kayıt sistemiyle toplar.

    Basınç eşik değerine göre kayıt otomatik başlar ve biter:
      - Basınç > threshold → kayıt başlar
      - Basınç < threshold ve çizgi süresi > 0.8s → kayıt biter
      - Maksimum kayıt süresi: 10 saniye (güvenlik)

    Args:
        is_reference: True ise referans (master) kaydı yapılıyor.
        progress_callback: GUI entegrasyonu için callback fonksiyonu.
                          Kayıt sırasında kalan süreyi / durumu iletir.
        port: Seri port adı. None ise otomatik algılama yapılır.
        threshold: Basınç eşik değeri (0-1023). Varsayılan: 50
        kalibrasyon_yap: True ise kayıt öncesi kalibrasyon yapılır.
        kalibrasyon_suresi: Kalibrasyon süresi (saniye).

    Returns:
        Tuple[np.ndarray, Dict]: İki elemanlı tuple:
            - np.ndarray: Shape (N, 7) — sütunlar: ax, ay, az, gx, gy, gz, pressure
            - Dict: Meta veriler:
                - sample_rate (float): Gerçek örnekleme hızı (Hz)
                - duration (float): Toplam kayıt süresi (saniye)
                - num_samples (int): Toplanan örnek sayısı
                - num_strokes (int): Algılanan çizgi sayısı
                - start_time (float): Kayıt başlangıç zaman damgası (epoch)
    """
    bos_sonuc = (
        np.zeros((0, NUM_COLUMNS)),
        {'sample_rate': 0.0, 'duration': 0.0, 'num_samples': 0,
         'num_strokes': 0, 'start_time': 0.0}
    )

    # --- Seri bağlantı kur ---
    try:
        ser = connect_serial(port)
    except ConnectionError as e:
        print(f"\n{e}")
        return bos_sonuc

    # --- Opsiyonel kalibrasyon ---
    offsets = None
    if kalibrasyon_yap:
        offsets = calibrate(duration=kalibrasyon_suresi, ser=ser,
                            progress_callback=progress_callback)

    # --- Kayıt durumu değişkenleri ---
    veri: List[List[float]] = []    # Toplanan veriler
    kayit_aktif: bool = False       # Şu anda kayıt yapılıyor mu?
    kayit_baslangic: float = 0.0    # Kayıt başlama zamanı
    cizgi_sayisi: int = 0           # Algılanan çizgi (stroke) sayısı
    toplam_baslangic: float = 0.0   # Genel başlangıç zamanı (ilk çizgiden itibaren)
    bekleme_baslangic: float = time.time()  # Timeout sayacı

    # Bekleme modu mesajı
    mod_str = "REFERANS" if is_reference else "KULLANICI"
    print(f"\n[Kayıt] {mod_str} kaydı bekleniyor...")
    print(f"  → Basınç eşiği: {threshold}")
    print(f"  → Kalem bastırıldığında kayıt otomatik başlayacak.")
    print(f"  → Kaldırıldığında (>{MIN_STROKE_DURATION}s sonra) kayıt duracak.")

    if progress_callback:
        progress_callback(-1.0)  # -1 = bekleme modunda

    ser.reset_input_buffer()

    try:
        while True:
            # --- Bekleme timeout kontrolü (kayıt başlamadıysa) ---
            if not kayit_aktif and (time.time() - bekleme_baslangic) >= WAIT_TIMEOUT:
                print(f"\n[UYARI] {WAIT_TIMEOUT:.0f} saniye boyunca basınç algılanamadı, iptal ediliyor.")
                break

            # --- Satır oku ---
            try:
                ham_satir = ser.readline().decode('utf-8', errors='ignore').strip()
            except serial.SerialException as e:
                print(f"\n[HATA] Seri port okuma hatası: {e}")
                break
            except Exception:
                continue

            degerler = parse_line(ham_satir)
            if degerler is None:
                continue

            # Kalibrasyon offset'lerini uygula
            if offsets is not None:
                eksen_isimleri = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pressure']
                degerler = [d - offsets[e] for d, e in zip(degerler, eksen_isimleri)]

            basinc = degerler[6]  # Son sütun: basınç

            # --- Durum makinesi: basınç tabanlı kayıt ---
            if not kayit_aktif:
                # Bekleme modunda — basınç eşiği aşıldı mı?
                if basinc > threshold:
                    kayit_aktif = True
                    kayit_baslangic = time.time()
                    cizgi_sayisi += 1

                    if toplam_baslangic == 0.0:
                        toplam_baslangic = kayit_baslangic

                    print(f"\n  ✏️  Çizgi #{cizgi_sayisi} başladı (basınç: {basinc:.0f})")

                    # İlk basınç verisini de kaydet
                    veri.append(degerler)

                    if progress_callback:
                        progress_callback(MAX_RECORD_TIME)
            else:
                # Kayıt modunda — veriyi topla
                veri.append(degerler)
                gecen_sure = time.time() - kayit_baslangic

                # İlerleme callback'i
                if progress_callback:
                    kalan = MAX_RECORD_TIME - gecen_sure
                    progress_callback(max(0.0, kalan))

                # Maksimum süre aşıldı mı? (güvenlik)
                if gecen_sure >= MAX_RECORD_TIME:
                    print(f"\n  ⏱️  Maksimum süre ({MAX_RECORD_TIME}s) aşıldı, kayıt durduruluyor.")
                    break

                # Basınç eşiğin altına düştü mü?
                if basinc < threshold:
                    if gecen_sure >= MIN_STROKE_DURATION:
                        # Yeterli süre geçti — kayıt tamamlandı
                        print(f"  ✅ Çizgi #{cizgi_sayisi} bitti (süre: {gecen_sure:.2f}s, "
                              f"{len(veri)} örnek)")
                        break
                    # Süre çok kısa — muhtemelen anlık temas, devam et
                    # (kayıt aktif kalır, kısa süreli basınç düşüşlerini tolere eder)

    except KeyboardInterrupt:
        print("\n[Kayıt] Kullanıcı tarafından iptal edildi.")
    finally:
        ser.close()
        print("[Donanım] Seri port kapatıldı.")

    # --- Sonuçları hazırla ---
    if len(veri) == 0:
        print("[UYARI] Hiç veri toplanamadı!")
        return bos_sonuc

    veri_array = np.array(veri, dtype=np.float64)
    toplam_sure = time.time() - toplam_baslangic if toplam_baslangic > 0 else 0.0
    ornekleme_hizi = len(veri) / toplam_sure if toplam_sure > 0 else 0.0

    meta = {
        'sample_rate': round(ornekleme_hizi, 1),
        'duration': round(toplam_sure, 3),
        'num_samples': len(veri),
        'num_strokes': cizgi_sayisi,
        'start_time': toplam_baslangic
    }

    print(f"\n[Sonuç] Kayıt tamamlandı:")
    print(f"  → Örnekler: {meta['num_samples']}")
    print(f"  → Süre: {meta['duration']:.2f}s")
    print(f"  → Örnekleme hızı: {meta['sample_rate']:.1f} Hz")
    print(f"  → Çizgi sayısı: {meta['num_strokes']}")
    print(f"  → Veri shape: {veri_array.shape}")

    return veri_array, meta


# ==========================================
# DOĞRUDAN ÇALIŞTIRMA — TEST MODU
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("  SENSOR DATA — TEST MODU")
    print("=" * 50)

    # Port algılama testi
    port = find_arduino_port()
    print(f"\nBulunan port: {port}")

    # Veri toplama testi
    print("\nVeri toplamaya hazır mısınız? Kalem bastırıldığında kayıt başlayacak.")
    input("Devam etmek için ENTER'a basın...")

    data, meta = get_sensor_data(port=port)
    print(f"\nToplanan veri: {data.shape}")
    print(f"Meta bilgiler: {meta}")

    if data.shape[0] > 0:
        print(f"\nİlk 5 satır:")
        print(data[:5])