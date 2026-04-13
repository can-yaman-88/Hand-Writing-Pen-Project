"""
save_data.py — Komut satırından referans harf kaydetme aracı.

Yeni klasör yapısı: letters_db/<HARF>/ref_01.npy, ref_02.npy, ...
Buton tabanlı kayıt sistemi kullanır.
"""

import numpy as np
import os
from sensor_data import get_sensor_data
from dtw_analysis import save_reference

if not os.path.exists("letters_db"):
    os.makedirs("letters_db")

print("=" * 50)
print("  REFERANS KAYIT ARACI (Buton Tabanlı)")
print("=" * 50)
print()
print("Kullanım: Harf girin → butona basarak yazın → referans kaydedilir")
print("Çıkmak için 'exit' yazın.")
print()

while True:
    target_letter = input("Hangi harfi kaydetmek istiyorsunuz? (veya 'exit'): ").strip().upper()

    if target_letter == "EXIT":
        print("Çıkılıyor...")
        break

    if not target_letter.isalpha() or len(target_letter) != 1:
        print("[HATA] Lütfen tek bir harf girin!")
        continue

    print(f"\n'{target_letter}' harfini yazmak için butona basın...")

    data = get_sensor_data(is_reference=True)

    if data.shape[0] == 0:
        print("[HATA] Veri toplanamadı! Tekrar deneyin.\n")
        continue

    file_path = save_reference(target_letter, data)
    print(f"[OK] '{target_letter}' referansı kaydedildi: {file_path}")
    print(f"     {data.shape[0]} örnek, {data.shape[0]/100:.1f} saniye\n")