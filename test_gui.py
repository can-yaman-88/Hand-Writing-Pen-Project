import sys
import numpy as np
sys.path.append('.')
import gui
from gui import HandwritingApp

app = HandwritingApp()
# app'a sentetik veri vererek _pipeline_calistir i test edelim
N = 100
t = np.linspace(0, 1, N)
veri = np.column_stack([np.sin(t), np.cos(t), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)])

import traceback
try:
    print("Testing pipeline...")
    sonuc = app._pipeline_calistir(veri)
    print("Success")
except Exception as e:
    print("Failed")
    traceback.print_exc()
