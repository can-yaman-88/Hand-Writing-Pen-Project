// ============================================================
// Arduino Nano + MPU6500 (GY-6500) + Buton Firmware
// 6-Eksen IMU + Buton - 100Hz Örnekleme
// ============================================================

#include <Wire.h>  // I2C haberleşme kütüphanesi

// ---- MPU6500 I2C Adresi ----
#define MPU6500_ADDR 0x68  // MPU6500 varsayılan I2C adresi

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
#define BUTTON_PIN 2        // Basınç sensörü yerine kullanılan fiziksel buton pini (Dijital)
#define LED_PIN LED_BUILTIN // Dahili LED pini (Arduino Nano: Pin 13)

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
// ============================================================
void sensorVeriOku(int16_t* ax, int16_t* ay, int16_t* az,
                   int16_t* gx, int16_t* gy, int16_t* gz) {
  Wire.beginTransmission(MPU6500_ADDR);    // I2C iletişimi başlat
  Wire.write(REG_ACCEL_XOUT_H);           // Okuma başlangıç adresi (0x3B)
  Wire.endTransmission(false);             // Tekrarlı başlatma için bağlantıyı koru
  Wire.requestFrom(MPU6500_ADDR, (uint8_t)14); // 14 bayt veri iste

  // İvmeölçer verileri
  *ax = (Wire.read() << 8) | Wire.read();  
  *ay = (Wire.read() << 8) | Wire.read();  
  *az = (Wire.read() << 8) | Wire.read();  

  // Sıcaklık verisini atla
  Wire.read();
  Wire.read();

  // Jiroskop verileri
  *gx = (Wire.read() << 8) | Wire.read();  
  *gy = (Wire.read() << 8) | Wire.read();  
  *gz = (Wire.read() << 8) | Wire.read();  
}

// ============================================================
// MPU6500 sensör bağlantı kontrolü
// ============================================================
bool sensorKontrol() {
  uint8_t kimlik = registerOku(REG_WHO_AM_I);
  return (kimlik == 0x70 || kimlik == 0x68 || kimlik == 0x73);
}

// ============================================================
// MPU6500 başlatma ve yapılandırma fonksiyonu
// ============================================================
void sensorBaslat() {
  registerYaz(REG_PWR_MGMT_1, 0x80);  // Sensörü sıfırla
  delay(100);

  registerYaz(REG_PWR_MGMT_1, 0x01);  // Uyku modundan çık
  delay(10);

  registerYaz(REG_PWR_MGMT_2, 0x00);  // Tüm eksenleri etkinleştir

  registerYaz(REG_SMPLRT_DIV, 0x09);  // Örnekleme hızı 100Hz

  registerYaz(REG_CONFIG, 0x03);      // DLPF ayarı: 41Hz bant genişliği
  registerYaz(REG_GYRO_CONFIG, 0x00); // Jiroskop: ±250°/s
  registerYaz(REG_ACCEL_CONFIG, 0x00);  // İvmeölçer: ±2g
  registerYaz(REG_ACCEL_CONFIG2, 0x03); // İvmeölçer DLPF: 41Hz
}

// ============================================================
// Kalibrasyon fonksiyonu
// ============================================================
void kalibrasyon() {
  ax_offset = 0; ay_offset = 0; az_offset = 0;
  gx_offset = 0; gy_offset = 0; gz_offset = 0;

  int16_t ax, ay, az, gx, gy, gz;

  digitalWrite(LED_PIN, HIGH);
  Serial.println("KALIBRASYON BASLIYOR...");

  for (int i = 0; i < KALIBRASYON_SAYISI; i++) {
    sensorVeriOku(&ax, &ay, &az, &gx, &gy, &gz);

    ax_offset += ax; ay_offset += ay; az_offset += az;
    gx_offset += gx; gy_offset += gy; gz_offset += gz;

    delay(2);
  }

  ax_offset /= KALIBRASYON_SAYISI; ay_offset /= KALIBRASYON_SAYISI; az_offset /= KALIBRASYON_SAYISI;
  gx_offset /= KALIBRASYON_SAYISI; gy_offset /= KALIBRASYON_SAYISI; gz_offset /= KALIBRASYON_SAYISI;

  digitalWrite(LED_PIN, LOW);
  Serial.println("KALIBRASYON TAMAMLANDI");
}

// ============================================================
// Hata durumunda LED yanıp sönme
// ============================================================
void hataLedYanipSon() {
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Buton pinini giriş olarak ayarla ve dahili pull-up direncini aktif et
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(400000);

  delay(100);

  if (!sensorKontrol()) {
    Serial.println("HATA: MPU6500 BULUNAMADI!");
    hataLedYanipSon();
  }

  Serial.println("MPU6500 BULUNDU");
  sensorBaslat();
  delay(100);
  kalibrasyon();

  onceki_zaman = millis();
  Serial.println("SISTEM HAZIR - 100Hz");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  unsigned long simdiki_zaman = millis();

  if (simdiki_zaman - onceki_zaman >= ORNEKLEME_PERIYODU) {
    onceki_zaman = simdiki_zaman;

    int16_t ax, ay, az, gx, gy, gz;
    sensorVeriOku(&ax, &ay, &az, &gx, &gy, &gz);

    int16_t ax_duzeltilmis = ax - (int16_t)ax_offset;
    int16_t ay_duzeltilmis = ay - (int16_t)ay_offset;
    int16_t az_duzeltilmis = az - (int16_t)az_offset;
    int16_t gx_duzeltilmis = gx - (int16_t)gx_offset;
    int16_t gy_duzeltilmis = gy - (int16_t)gy_offset;
    int16_t gz_duzeltilmis = gz - (int16_t)gz_offset;

    // Buton durumunu oku: INPUT_PULLUP olduğu için basılıysa LOW(0), değilse HIGH(1) döner.
    // Veriyi tersliyoruz ki basılı durumu 1 olsun.
    int buton_durumu = (digitalRead(BUTTON_PIN) == LOW) ? 1 : 0;

    // Verileri seri port üzerinden gönder
    // Format: ax,ay,az,gx,gy,gz,button
    Serial.print(ax_duzeltilmis); 
    Serial.print(',');            
    Serial.print(ay_duzeltilmis); 
    Serial.print(',');
    Serial.print(az_duzeltilmis); 
    Serial.print(',');
    Serial.print(gx_duzeltilmis); 
    Serial.print(',');
    Serial.print(gy_duzeltilmis); 
    Serial.print(',');
    Serial.print(gz_duzeltilmis); 
    Serial.print(',');
    Serial.println(buton_durumu); // 7. sütun olarak buton durumu
  }
}