// ============================================================
// Arduino Micro + MPU6500 (GY-6500) + FSR 402 Firmware
// 6-Eksen IMU + Basınç Sensörü - 100Hz Örnekleme
// ============================================================

#include <Wire.h>  // I2C haberleşme kütüphanesi

// ---- MPU6500 I2C Adresi ----
#define MPU6500_ADDR 0x69  // MPU6500 varsayılan I2C adresi

// ---- MPU6500 Register Adresleri ----
#define REG_WHO_AM_I    0x75  // Sensör kimlik register'ı (0x70 döndürmeli)
#define REG_PWR_MGMT_1  0x6B  // Güç yönetimi register'ı
#define REG_PWR_MGMT_2  0x6C  // Güç yönetimi register'ı 2
#define REG_SMPLRT_DIV  0x19  // Örnekleme hızı bölücü
#define REG_CONFIG      0x1A  // Genel yapılandırma (DLPF ayarı)
#define REG_GYRO_CONFIG 0x1B  // Jiroskop yapılandırma register'ı
#define REG_ACCEL_CONFIG 0x1C // İvmeölçer yapılandırma register'ı
#define REG_ACCEL_CONFIG2 0x1D // İvmeölçer yapılandırma register'ı 2
#define REG_ACCEL_XOUT_H 0x3B // İvmeölçer X ekseni yüksek bayt

// ---- Pin Tanımlamaları ----
#define FSR_PIN A0          // FSR 402 basınç sensörü analog pini
#define LED_PIN LED_BUILTIN // Dahili LED pini (Arduino Micro: Pin 13)

// ---- Kalibrasyon Ayarları ----
#define KALIBRASYON_SAYISI 500  // Kalibrasyon için alınacak okuma sayısı

// ---- Zamanlama ----
#define ORNEKLEME_PERIYODU 10  // 10ms = 100Hz örnekleme periyodu

// ---- Kalibrasyon Offset Değişkenleri ----
// İvmeölçer offset'leri (uzun tam sayı, taşma önleme)
long ax_offset = 0;
long ay_offset = 0;
long az_offset = 0;

// Jiroskop offset'leri
long gx_offset = 0;
long gy_offset = 0;
long gz_offset = 0;

// ---- Zamanlama Değişkeni ----
unsigned long onceki_zaman = 0;  // Son okuma zamanını tutar

// ============================================================
// MPU6500'e tek bir register yazma fonksiyonu
// ============================================================
void registerYaz(uint8_t reg, uint8_t deger) {
  Wire.beginTransmission(MPU6500_ADDR);  // I2C iletişimi başlat
  Wire.write(reg);                        // Hedef register adresini gönder
  Wire.write(deger);                      // Yazılacak değeri gönder
  Wire.endTransmission(true);            // İletişimi sonlandır
}

// ============================================================
// MPU6500'den tek bir register okuma fonksiyonu
// ============================================================
uint8_t registerOku(uint8_t reg) {
  Wire.beginTransmission(MPU6500_ADDR);  // I2C iletişimi başlat
  Wire.write(reg);                        // Okunacak register adresini gönder
  Wire.endTransmission(false);           // Tekrarlı başlatma için bağlantıyı koru
  Wire.requestFrom(MPU6500_ADDR, (uint8_t)1); // 1 bayt veri iste
  return Wire.read();                     // Okunan baytı döndür
}

// ============================================================
// MPU6500'den 14 bayt ham veri okuma fonksiyonu
// (ax, ay, az, sıcaklık, gx, gy, gz = 7x2 bayt)
// ============================================================
void sensorVeriOku(int16_t* ax, int16_t* ay, int16_t* az,
                   int16_t* gx, int16_t* gy, int16_t* gz) {
  Wire.beginTransmission(MPU6500_ADDR);    // I2C iletişimi başlat
  Wire.write(REG_ACCEL_XOUT_H);           // Okuma başlangıç adresi (0x3B)
  Wire.endTransmission(false);             // Tekrarlı başlatma için bağlantıyı koru
  Wire.requestFrom(MPU6500_ADDR, (uint8_t)14); // 14 bayt veri iste

  // İvmeölçer verileri (her eksen 2 bayt: yüksek + düşük)
  *ax = (Wire.read() << 8) | Wire.read();  // X ekseni ivme
  *ay = (Wire.read() << 8) | Wire.read();  // Y ekseni ivme
  *az = (Wire.read() << 8) | Wire.read();  // Z ekseni ivme

  // Sıcaklık verisini atla (2 bayt - kullanmıyoruz)
  Wire.read();
  Wire.read();

  // Jiroskop verileri (her eksen 2 bayt: yüksek + düşük)
  *gx = (Wire.read() << 8) | Wire.read();  // X ekseni açısal hız
  *gy = (Wire.read() << 8) | Wire.read();  // Y ekseni açısal hız
  *gz = (Wire.read() << 8) | Wire.read();  // Z ekseni açısal hız
}

// ============================================================
// MPU6500 sensör bağlantı kontrolü
// WHO_AM_I register'ını okuyarak sensörün bağlı olup olmadığını doğrular
// ============================================================
bool sensorKontrol() {
  uint8_t kimlik = registerOku(REG_WHO_AM_I);  // Kimlik register'ını oku
  // MPU6500 için WHO_AM_I değeri 0x70 olmalı
  // Bazı klonlarda 0x68 veya 0x73 dönebilir, hepsini kabul et
  return (kimlik == 0x70 || kimlik == 0x68 || kimlik == 0x73);
}

// ============================================================
// MPU6500 başlatma ve yapılandırma fonksiyonu
// ============================================================
void sensorBaslat() {
  registerYaz(REG_PWR_MGMT_1, 0x80);  // Sensörü sıfırla (reset)
  delay(100);                           // Sıfırlamanın tamamlanmasını bekle

  registerYaz(REG_PWR_MGMT_1, 0x01);  // Uyku modundan çık, saat kaynağı: PLL
  delay(10);                            // Kararlı hale gelmesini bekle

  registerYaz(REG_PWR_MGMT_2, 0x00);  // Tüm eksenleri etkinleştir

  registerYaz(REG_SMPLRT_DIV, 0x09);  // Örnekleme hızı bölücü: 1kHz/(1+9) = 100Hz

  registerYaz(REG_CONFIG, 0x03);      // DLPF ayarı: bant genişliği ~41Hz
  // Düşük geçiren filtre gürültüyü azaltır

  registerYaz(REG_GYRO_CONFIG, 0x00); // Jiroskop aralığı: ±250°/s
  // Daha hassas ölçüm için en düşük aralık

  registerYaz(REG_ACCEL_CONFIG, 0x00);  // İvmeölçer aralığı: ±2g
  // Daha hassas ölçüm için en düşük aralık

  registerYaz(REG_ACCEL_CONFIG2, 0x03); // İvmeölçer DLPF: bant genişliği ~41Hz
}

// ============================================================
// Kalibrasyon fonksiyonu
// Sensör sabitken 500 okuma alır ve ortalamayı offset olarak kaydeder
// ============================================================
void kalibrasyon() {
  // Kalibrasyon offset'lerini sıfırla
  ax_offset = 0;
  ay_offset = 0;
  az_offset = 0;
  gx_offset = 0;
  gy_offset = 0;
  gz_offset = 0;

  int16_t ax, ay, az, gx, gy, gz;  // Geçici okuma değişkenleri

  // Kalibrasyon sırasında dahili LED'i yak
  digitalWrite(LED_PIN, HIGH);

  // Seri port üzerinden bilgi mesajı gönder
  Serial.println("KALIBRASYON BASLIYOR...");

  // Belirtilen sayıda okuma al ve topla
  for (int i = 0; i < KALIBRASYON_SAYISI; i++) {
    sensorVeriOku(&ax, &ay, &az, &gx, &gy, &gz);  // Sensörden veri oku

    // Okunan değerleri toplam değişkenlere ekle
    ax_offset += ax;
    ay_offset += ay;
    az_offset += az;
    gx_offset += gx;
    gy_offset += gy;
    gz_offset += gz;

    delay(2);  // Okumalar arası kısa bekleme (kararlılık için)
  }

  // Toplam değerleri okuma sayısına bölerek ortalamayı hesapla
  ax_offset /= KALIBRASYON_SAYISI;
  ay_offset /= KALIBRASYON_SAYISI;
  az_offset /= KALIBRASYON_SAYISI;
  gx_offset /= KALIBRASYON_SAYISI;
  gy_offset /= KALIBRASYON_SAYISI;
  gz_offset /= KALIBRASYON_SAYISI;

  // Kalibrasyon bitince LED'i söndür
  digitalWrite(LED_PIN, LOW);

  // Kalibrasyon tamamlandı bilgisi
  Serial.println("KALIBRASYON TAMAMLANDI");
}

// ============================================================
// Hata durumunda LED'i hızlı yanıp söndürme fonksiyonu
// Sensör bulunamazsa sonsuz döngüde kalır
// ============================================================
void hataLedYanipSon() {
  while (true) {
    digitalWrite(LED_PIN, HIGH);  // LED'i yak
    delay(100);                    // 100ms bekle
    digitalWrite(LED_PIN, LOW);   // LED'i söndür
    delay(100);                    // 100ms bekle
    // Bu döngü hiç bitmez - sensör bağlanana kadar yeniden başlatma gerekir
  }
}

// ============================================================
// SETUP - Başlangıç ayarları (bir kez çalışır)
// ============================================================
void setup() {
  // Seri haberleşmeyi başlat (115200 baud)
  Serial.begin(115200);

  // Seri portun hazır olmasını bekle (USB bağlantısı için gerekli)
  // Arduino Micro USB üzerinden haberleştiği için bu bekleme önemli
  while (!Serial) {
    ;  // USB bağlantısı kurulana kadar bekle
  }

  // LED pinini çıkış olarak ayarla
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);  // LED'i başlangıçta söndür

  // FSR pini analog giriş olarak zaten varsayılan, ayar gerekmez

  // I2C haberleşmeyi başlat
  // Arduino Micro'da SDA=Pin2, SCL=Pin3 (donanımsal I2C)
  Wire.begin();
  Wire.setClock(400000);  // I2C hızını 400kHz'e ayarla (Fast Mode)

  delay(100);  // I2C veri yolunun kararlı hale gelmesini bekle

  // Sensör bağlantı kontrolü yap
  if (!sensorKontrol()) {
    // Sensör bulunamadı - hata mesajı gönder
    Serial.println("HATA: MPU6500 BULUNAMADI!");
    // LED hızlı yanıp sönsün (sonsuz döngü)
    hataLedYanipSon();
  }

  // Sensör bulundu - bilgi mesajı gönder
  Serial.println("MPU6500 BULUNDU");

  // MPU6500'ü yapılandır
  sensorBaslat();

  // Sensörün kararlı hale gelmesini bekle
  delay(100);

  // Kalibrasyon işlemini gerçekleştir
  kalibrasyon();

  // Zamanlama referans noktasını ayarla
  onceki_zaman = millis();

  // Sisteme hazır bilgisi gönder
  Serial.println("SISTEM HAZIR - 100Hz");
}

// ============================================================
// LOOP - Ana döngü (sürekli tekrarlanır)
// ============================================================
void loop() {
  // Şu anki zamanı al
  unsigned long simdiki_zaman = millis();

  // 10ms (100Hz) geçip geçmediğini kontrol et
  if (simdiki_zaman - onceki_zaman >= ORNEKLEME_PERIYODU) {
    // Zamanlama referansını güncelle
    onceki_zaman = simdiki_zaman;

    // Ham sensör verilerini tutacak değişkenler
    int16_t ax, ay, az, gx, gy, gz;

    // MPU6500'den 6 eksen verisini oku
    sensorVeriOku(&ax, &ay, &az, &gx, &gy, &gz);

    // Kalibrasyon offset'lerini çıkararak düzeltilmiş değerleri hesapla
    int16_t ax_duzeltilmis = ax - (int16_t)ax_offset;
    int16_t ay_duzeltilmis = ay - (int16_t)ay_offset;
    int16_t az_duzeltilmis = az - (int16_t)az_offset;
    int16_t gx_duzeltilmis = gx - (int16_t)gx_offset;
    int16_t gy_duzeltilmis = gy - (int16_t)gy_offset;
    int16_t gz_duzeltilmis = gz - (int16_t)gz_offset;

    // Verileri seri port üzerinden gönder
    // Format: ax,ay,az,gx,gy,gz
    Serial.print(ax_duzeltilmis);  // İvme X
    Serial.print(',');              // Ayırıcı virgül
    Serial.print(ay_duzeltilmis);  // İvme Y
    Serial.print(',');
    Serial.print(az_duzeltilmis);  // İvme Z
    Serial.print(',');
    Serial.print(gx_duzeltilmis);  // Jiroskop X
    Serial.print(',');
    Serial.print(gy_duzeltilmis);  // Jiroskop Y
    Serial.print(',');
    Serial.println(gz_duzeltilmis);  // Jiroskop Z (satır sonu ile)
  }
}
