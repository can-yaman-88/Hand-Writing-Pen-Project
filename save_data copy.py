import numpy as np
import os
from sensor_data import get_sensor_data

if not os.path.exists("letters_db"):
    os.makedirs("letters_db")

print("=== HARDWARE REFERENCE GENERATOR ===")
while True:
    target_letter = input("\nWhich letter to record? (or 'exit'): ").strip().upper()
    if target_letter == 'EXIT': break
    if not target_letter.isalpha() or len(target_letter) != 1: continue

    print(f"\n[!] Recording physical data for '{target_letter}' for 5 seconds... DRAW NOW!")
    ideal_data = get_sensor_data()
    file_path = f"letters_db/letter_{target_letter}.npy"
    np.save(file_path, ideal_data)
    print(f"Success! Saved to '{file_path}'.")