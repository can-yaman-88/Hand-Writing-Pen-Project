#include <Wire.h>

#define MPU_ADDR 0x68  // MPU6500 / MPU9250 default adres

int16_t AcX, AcY, AcZ;
int16_t GyX, GyY, GyZ;

void setup() {
  Wire.begin();
  Serial.begin(115200);

  // MPU'yu uyandır (sleep moddan çıkar)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // sleep = 0
  Wire.endTransmission(true);

  // sensor_data.py bağlantı kurulduğunda "HAZIR" mesajı bekliyor
  Serial.println("SISTEM HAZIR");
}

void loop() {
  // Veri okumaya başla (ACCEL_XOUT_H register)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  // Accelerometer
  AcX = Wire.read() << 8 | Wire.read();
  AcY = Wire.read() << 8 | Wire.read();
  AcZ = Wire.read() << 8 | Wire.read();

  Wire.read(); Wire.read(); // Temperature skip

  // Gyroscope
  GyX = Wire.read() << 8 | Wire.read();
  GyY = Wire.read() << 8 | Wire.read();
  GyZ = Wire.read() << 8 | Wire.read();

  // Verileri virgülle ayırarak gönder (sensor_data.py ile uyumlu CSV formatı)
  Serial.print(AcX);
  Serial.print(",");
  Serial.print(AcY);
  Serial.print(",");
  Serial.print(AcZ);
  Serial.print(",");
  Serial.print(GyX);
  Serial.print(",");
  Serial.print(GyY);
  Serial.print(",");
  Serial.println(GyZ);

  // sensor_data.py yaklaşık 100Hz örnekleme hızı bekler, bu yüzden 500ms yerine 10ms bekliyoruz.
  delay(10);
}