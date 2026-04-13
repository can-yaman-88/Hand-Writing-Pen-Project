"""
gui.py — El Yazısı Öğrenme Uygulaması Ana Arayüz Modülü

CustomTkinter tabanlı modern GUI. IMU sensör verisini DTW ile
analiz ederek kullanıcının el yazısı kalitesini değerlendirir.
Makine öğrenmesi KULLANILMAZ — tüm analiz DTW ve matematiksel
metriklerle yapılır [1].

Modül Bağlantıları:
    - sensor_data.py   → Sensör veri toplama
    - preprocessing.py → Filtreleme ve segmentasyon
    - dtw_analysis.py  → DTW analizi ve harf tanıma [1]
    - scoring.py       → Çok boyutlu skorlama
    - trajectory.py    → 2D iz görselleştirme
    - progress.py      → İlerleme yönetimi

Bağımlılıklar:
    - customtkinter, matplotlib, numpy
    - threading, time, os, sys, json
"""

import os
import sys
import time
import json
import threading
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────
# Proje modülleri (güvenli import — eksik modüller simüle edilir)
# ─────────────────────────────────────────────────────────────────────

try:
    from sensor_data import get_sensor_data, calibrate as sensor_calibrate
except ImportError:
    def get_sensor_data(*a, **kw):
        """Simüle sensör verisi (test için)."""
        time.sleep(2)
        N = 200
        t = np.linspace(0, 2 * np.pi, N)
        return np.column_stack([
            np.sin(t) + np.random.normal(0, 0.05, N),
            np.cos(t) + np.random.normal(0, 0.05, N),
            np.random.normal(0, 0.02, N),
            np.sin(t) * 0.5,
            np.cos(t) * 0.5,
            np.zeros(N),
            np.where(t < 5.5, 0.8, 0.0),
        ])

    def sensor_calibrate(*a, **kw):
        time.sleep(1)
        return True

try:
    from preprocessing import filter_data, segment_strokes, normalize_data
except ImportError:
    def filter_data(data, **kw):
        return data

    def segment_strokes(data, **kw):
        pressure = data[:, 6] if data.shape[1] > 6 else np.ones(data.shape[0])
        aktif = pressure > 0.3
        vuruslar = []
        bas = None
        for i, d in enumerate(aktif):
            if d and bas is None:
                bas = i
            elif not d and bas is not None:
                vuruslar.append((bas, i - 1))
                bas = None
        if bas is not None:
            vuruslar.append((bas, len(aktif) - 1))
        if not vuruslar:
            vuruslar = [(0, len(data) - 1)]
        return vuruslar

    def normalize_data(data, **kw):
        return data

try:
    from dtw_analysis import identify_letter, compare_with_reference, calculate_2d_trajectory
except ImportError:
    def identify_letter(data, **kw):
        return {"letter": "A", "confidence": 0.85, "distances": {}}

    def compare_with_reference(user, ref, **kw):
        return 15.0

    def calculate_2d_trajectory(data, **kw):
        N = data.shape[0]
        t = np.linspace(0, 2 * np.pi, N)
        px = np.cumsum(data[:, 0]) * 0.001
        py = np.cumsum(data[:, 1]) * 0.001
        pressure = data[:, 6] if data.shape[1] > 6 else np.ones(N)
        vuruslar = [(0, N - 1)]
        return px, py, pressure, vuruslar

try:
    from scoring import calculate_quality_score
except ImportError:
    def calculate_quality_score(user_data, ref_data, user_strokes,
                                ref_strokes, dtw_distance,
                                identification_confidence, **kw):
        skor = max(0, min(10, 10 - dtw_distance / 15))
        return {
            "total_score": round(skor, 1),
            "details": {
                "shape_similarity": round(skor + 0.5, 1),
                "stroke_accuracy": round(skor - 0.3, 1),
                "smoothness": round(skor + 0.2, 1),
                "speed_consistency": round(skor - 0.1, 1),
                "confidence": round(min(10, identification_confidence * 10), 1),
            },
            "feedback": [
                "Genel şekil iyi görünüyor.",
                "Vuruş geçişlerini daha akıcı yapabilirsin.",
            ],
            "pass": skor >= 7.0,
            "grade": "İyi" if skor >= 7 else "Orta",
        }

try:
    from trajectory import (
        calculate_trajectory,
        plot_trajectory,
        plot_comparison,
        plot_overlay,
        get_reference_shape,
    )
except ImportError:
    def calculate_trajectory(data, **kw):
        N = data.shape[0]
        px = np.cumsum(data[:, 0]) * 0.001
        py = np.cumsum(data[:, 1]) * 0.001
        pressure = data[:, 6] if data.shape[1] > 6 else np.ones(N)
        return px, py, pressure, [(0, N - 1)]

    def plot_trajectory(x, y, pressure=None, vuruslar=None, title="", **kw):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(x, y, "b-", linewidth=2)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def plot_comparison(rx, ry, ux, uy, score=0, **kw):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
        a1.plot(rx, ry, "gray", lw=2)
        a1.set_title("Referans")
        a1.set_aspect("equal")
        a1.invert_yaxis()
        a2.plot(ux, uy, "b-", lw=2)
        a2.set_title(f"Senin Yazın ({score:.1f})")
        a2.set_aspect("equal")
        a2.invert_yaxis()
        fig.tight_layout()
        return fig

    def plot_overlay(rx, ry, ux, uy, **kw):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rx, ry, "gray", lw=4, alpha=0.5, label="Referans")
        ax.plot(ux, uy, "b-", lw=2, label="Sen")
        ax.legend()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def get_reference_shape(letter, **kw):
        t = np.linspace(0, 2 * np.pi, 60)
        return 0.7 * np.cos(t), 0.85 * np.sin(t)

try:
    from progress import ProgressManager, SEVIYELER, HARF_SEVIYE, TUM_HARFLER
except ImportError:
    SEVIYELER = {
        1: ["O", "I", "C", "L"],
        2: ["U", "T", "V", "J"],
        3: ["A", "E", "M", "N", "K"],
        4: ["B", "D", "H", "P", "R"],
        5: ["F", "G", "S", "Y", "İ"],
        6: ["Ç", "Ğ", "Ö", "Ş", "Ü", "Z"],
    }
    TUM_HARFLER = []
    HARF_SEVIYE = {}
    for _s, _hl in sorted(SEVIYELER.items()):
        TUM_HARFLER.extend(_hl)
        for _h in _hl:
            HARF_SEVIYE[_h] = _s

    class ProgressManager:
        def __init__(self, **kw):
            self.username = kw.get("username", "Kullanıcı")
            self.unlocked_letters = set(SEVIYELER[1])
            self.current_level = 1
            self.letter_scores = {}
            self.total_attempts = 0
            self.streak = 0
            self.mode = "letter"
            self.letter_streaks = {}

        def harf_acik_mi(self, h):
            return h in self.unlocked_letters

        def erisilebilir_harfler(self):
            r = []
            for s in sorted(SEVIYELER.keys()):
                if s <= self.current_level:
                    r.extend(SEVIYELER[s])
            return r

        def deneme_kaydet(self, harf, skor, **kw):
            h = harf.upper()
            if h not in self.letter_scores:
                self.letter_scores[h] = []
            self.letter_scores[h].append(skor)
            self.total_attempts += 1
            basarili = skor >= 7.0
            if basarili:
                self.streak += 1
                self.letter_streaks[h] = self.letter_streaks.get(h, 0) + 1
                if self.letter_streaks[h] >= 3:
                    self.unlocked_letters.add(h)
            else:
                self.streak = 0
                self.letter_streaks[h] = 0
            return {
                "harf": h, "skor": skor, "basarili": basarili,
                "streak": self.streak, "yeni_acilan": [],
                "seviye_atladi": False, "mesaj": f"{h}: {skor:.1f}",
            }

        def get_next_task(self):
            return {"type": "letter", "target": "O", "level": 1,
                    "reason": "Pratik yap!"}

        def toplam_ilerleme(self):
            return round(len(self.unlocked_letters) / len(TUM_HARFLER) * 100, 1)

        def kelime_modu_acik_mi(self):
            return len(self.unlocked_letters) >= 8

        def kullanilabilir_kelimeler(self, **kw):
            return ["AL", "EL", "OL"]

        def istatistikler(self):
            return {
                "toplam_ilerleme_yuzde": self.toplam_ilerleme(),
                "mevcut_seviye": self.current_level,
                "toplam_deneme": self.total_attempts,
                "genel_ortalama_skor": 0.0,
                "ard_arda_basari": self.streak,
                "en_guclu_5": [], "en_zayif_5": [],
            }

        def harf_ortalama_skor(self, h):
            s = self.letter_scores.get(h.upper(), [])
            return round(sum(s) / len(s), 1) if s else 0.0

        def kaydet(self, **kw):
            pass

        def yukle(self, **kw):
            pass

        def ozet_rapor(self):
            return "İlerleme raporu"

        def mod_degistir(self, m):
            self.mode = m
            return True


# ─────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────

APP_TITLE = "El Yazısı Studio v3.0"
APP_GEOMETRY = "1400x850"
MIN_SIZE = (1100, 700)

# Renkler
COLORS = {
    "bg_dark": "#1a1a2e",
    "bg_panel": "#16213e",
    "bg_mid": "#0f3460",
    "accent": "#e94560",
    "accent_hover": "#ff6b6b",
    "success": "#4CAF50",
    "success_light": "#81C784",
    "warning": "#FF9800",
    "error": "#F44336",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0BEC5",
    "text_muted": "#78909C",
    "btn_active": "#2196F3",
    "btn_locked": "#37474F",
    "btn_current": "#FF9800",
    "score_bad": "#F44336",
    "score_mid": "#FF9800",
    "score_good": "#8BC34A",
    "score_great": "#4CAF50",
    "chart_bg": "#1e2a3a",
}

# Kriter isimleri (Türkçe)
KRITER_ISIMLERI = {
    "shape_similarity": "Şekil Benzerliği",
    "stroke_accuracy": "Vuruş Doğruluğu",
    "smoothness": "Akıcılık",
    "speed_consistency": "Hız Tutarlılığı",
    "confidence": "Tanıma Güveni",
}

# Referans veritabanı dizini
LETTERS_DB = "letters_db"

# Varsayılan COM port
DEFAULT_COM_PORT = "COM3"
DEFAULT_BAUD_RATE = 115200
DEFAULT_PRESSURE_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────

def skor_rengi(skor: float) -> str:
    """Skora göre renk kodu döndürür."""
    if skor >= 9.0:
        return COLORS["score_great"]
    elif skor >= 7.0:
        return COLORS["score_good"]
    elif skor >= 4.0:
        return COLORS["score_mid"]
    else:
        return COLORS["score_bad"]


def referans_yukle(harf: str) -> Optional[np.ndarray]:
    """letters_db dizininden referans verisini yükler.

    DTW analizi bu referans verileriyle karşılaştırma yapar [1].
    """
    harf_dizin = os.path.join(LETTERS_DB, harf.upper())
    if not os.path.isdir(harf_dizin):
        return None
    dosyalar = sorted(glob.glob(os.path.join(harf_dizin, "ref_*.npy")))
    if dosyalar:
        try:
            return np.load(dosyalar[0])
        except Exception:
            pass
    return None


def referans_kaydet(harf: str, veri: np.ndarray) -> str:
    """Referans verisini letters_db dizinine kaydeder."""
    harf_dizin = os.path.join(LETTERS_DB, harf.upper())
    os.makedirs(harf_dizin, exist_ok=True)
    mevcut = glob.glob(os.path.join(harf_dizin, "ref_*.npy"))
    idx = len(mevcut) + 1
    dosya_adi = f"ref_{idx:02d}.npy"
    yol = os.path.join(harf_dizin, dosya_adi)
    np.save(yol, veri)
    return yol


# ═════════════════════════════════════════════════════════════════════
# SOL PANEL — Harf Izgarası ve İlerleme
# ═════════════════════════════════════════════════════════════════════

class SolPanel(ctk.CTkFrame):
    """Sol panel: kullanıcı bilgisi, ilerleme, harf ızgarası, mod butonları.

    Attributes:
        app: Ana uygulama referansı.
        harf_butonlari: Harf → buton eşlemesi.
    """

    def __init__(self, master: Any, app: "HandwritingApp", **kwargs) -> None:
        super().__init__(master, width=260, corner_radius=0,
                         fg_color=COLORS["bg_panel"], **kwargs)
        self.app = app
        self.harf_butonlari: Dict[str, ctk.CTkButton] = {}
        self.pack_propagate(False)
        self._arayuz_olustur()

    def _arayuz_olustur(self) -> None:
        """Panel bileşenlerini oluşturur."""

        # ── Kullanıcı adı ──
        self.kullanici_label = ctk.CTkLabel(
            self, text=f"👤 {self.app.progress.username}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        self.kullanici_label.pack(pady=(15, 5), padx=10)

        # ── Seviye etiketi ──
        self.seviye_label = ctk.CTkLabel(
            self,
            text=f"📊 Seviye {self.app.progress.current_level}",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["accent"],
        )
        self.seviye_label.pack(pady=(0, 5))

        # ── İlerleme çubuğu ──
        self.ilerleme_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ilerleme_frame.pack(fill="x", padx=15, pady=(0, 5))

        self.ilerleme_bar = ctk.CTkProgressBar(
            self.ilerleme_frame,
            height=12,
            corner_radius=6,
            progress_color=COLORS["accent"],
        )
        self.ilerleme_bar.pack(fill="x", pady=(0, 2))
        self.ilerleme_bar.set(self.app.progress.toplam_ilerleme() / 100.0)

        self.ilerleme_label = ctk.CTkLabel(
            self.ilerleme_frame,
            text=f"%{self.app.progress.toplam_ilerleme():.0f} tamamlandı",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.ilerleme_label.pack()

        # ── Ayırıcı ──
        ctk.CTkFrame(self, height=2, fg_color=COLORS["bg_mid"]).pack(
            fill="x", padx=10, pady=8
        )

        # ── Harf ızgarası başlık ──
        ctk.CTkLabel(
            self, text="HARFLER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_secondary"],
        ).pack(pady=(0, 5))

        # ── Harf ızgarası ──
        self.izgara_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            height=380,
        )
        self.izgara_frame.pack(fill="x", padx=8, pady=(0, 5))

        self._harf_izgarasi_olustur()

        # ── Ayırıcı ──
        ctk.CTkFrame(self, height=2, fg_color=COLORS["bg_mid"]).pack(
            fill="x", padx=10, pady=8
        )

        # ── Alt butonlar ──
        self.alt_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.alt_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.kelime_btn = ctk.CTkButton(
            self.alt_frame,
            text="📝 Kelime Modu",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["bg_mid"],
            hover_color=COLORS["accent"],
            command=self._kelime_modu_ac,
            state="normal" if self.app.progress.kelime_modu_acik_mi() else "disabled",
            height=34,
        )
        self.kelime_btn.pack(fill="x", pady=2)

        self.referans_btn = ctk.CTkButton(
            self.alt_frame,
            text="💾 Referans Kaydet",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["bg_mid"],
            hover_color=COLORS["btn_active"],
            command=self.app.referans_kaydet_dialog,
            height=34,
        )
        self.referans_btn.pack(fill="x", pady=2)

        self.ayarlar_btn = ctk.CTkButton(
            self.alt_frame,
            text="⚙️ Ayarlar",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["bg_mid"],
            hover_color=COLORS["btn_active"],
            command=self.app.ayarlar_ac,
            height=34,
        )
        self.ayarlar_btn.pack(fill="x", pady=2)

        self.rapor_btn = ctk.CTkButton(
            self.alt_frame,
            text="📊 Rapor",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["bg_mid"],
            hover_color=COLORS["btn_active"],
            command=self.app.rapor_goster,
            height=34,
        )
        self.rapor_btn.pack(fill="x", pady=2)

    def _harf_izgarasi_olustur(self) -> None:
        """Seviye gruplarıyla harf buton ızgarasını oluşturur."""
        SUTUN = 6

        for seviye in sorted(SEVIYELER.keys()):
            harfler = SEVIYELER[seviye]
            erisebilir = seviye <= self.app.progress.current_level

            # Seviye başlığı
            renk = COLORS["text_secondary"] if erisebilir else COLORS["text_muted"]
            ctk.CTkLabel(
                self.izgara_frame,
                text=f"— Seviye {seviye} —",
                font=ctk.CTkFont(size=10),
                text_color=renk,
            ).pack(pady=(6, 2))

            satir_frame = ctk.CTkFrame(self.izgara_frame, fg_color="transparent")
            satir_frame.pack(fill="x", pady=1)

            for col_idx in range(SUTUN):
                satir_frame.columnconfigure(col_idx, weight=1)

            for idx, harf in enumerate(harfler):
                satir = idx // SUTUN
                sutun = idx % SUTUN

                if idx % SUTUN == 0 and idx > 0:
                    satir_frame = ctk.CTkFrame(
                        self.izgara_frame, fg_color="transparent"
                    )
                    satir_frame.pack(fill="x", pady=1)
                    for c in range(SUTUN):
                        satir_frame.columnconfigure(c, weight=1)

                acik = self.app.progress.harf_acik_mi(harf)
                hedef = (self.app.hedef_harf == harf)

                if hedef:
                    fg = COLORS["btn_current"]
                    hover = COLORS["accent_hover"]
                    state = "normal"
                elif acik:
                    fg = COLORS["btn_active"]
                    hover = COLORS["accent"]
                    state = "normal"
                elif erisebilir:
                    fg = COLORS["btn_locked"]
                    hover = COLORS["btn_locked"]
                    state = "normal"
                else:
                    fg = "#263238"
                    hover = "#263238"
                    state = "disabled"

                btn = ctk.CTkButton(
                    satir_frame,
                    text=harf if acik or erisebilir else "🔒",
                    width=34,
                    height=34,
                    corner_radius=6,
                    font=ctk.CTkFont(size=13, weight="bold"),
                    fg_color=fg,
                    hover_color=hover,
                    state=state,
                    command=lambda h=harf: self.app.harf_sec(h),
                )
                btn.grid(row=0, column=sutun, padx=2, pady=2, sticky="nsew")
                self.harf_butonlari[harf] = btn

    def guncelle(self) -> None:
        """Panel bilgilerini güncel duruma göre yeniler."""
        self.seviye_label.configure(
            text=f"📊 Seviye {self.app.progress.current_level}"
        )
        ilerleme = self.app.progress.toplam_ilerleme()
        self.ilerleme_bar.set(ilerleme / 100.0)
        self.ilerleme_label.configure(text=f"%{ilerleme:.0f} tamamlandı")

        # Kelime modu butonu
        if self.app.progress.kelime_modu_acik_mi():
            self.kelime_btn.configure(state="normal")
        else:
            self.kelime_btn.configure(state="disabled")

        # Harf butonlarını güncelle
        for harf, btn in self.harf_butonlari.items():
            acik = self.app.progress.harf_acik_mi(harf)
            hedef = (self.app.hedef_harf == harf)
            seviye = HARF_SEVIYE.get(harf, 99)
            erisebilir = seviye <= self.app.progress.current_level

            if hedef:
                btn.configure(
                    fg_color=COLORS["btn_current"],
                    hover_color=COLORS["accent_hover"],
                    text=harf, state="normal",
                )
            elif acik:
                ort = self.app.progress.harf_ortalama_skor(harf)
                renk = skor_rengi(ort) if ort > 0 else COLORS["btn_active"]
                btn.configure(
                    fg_color=renk,
                    hover_color=COLORS["accent"],
                    text=harf, state="normal",
                )
            elif erisebilir:
                btn.configure(
                    fg_color=COLORS["btn_locked"],
                    text=harf, state="normal",
                )
            else:
                btn.configure(
                    fg_color="#263238",
                    text="🔒", state="disabled",
                )

    def _kelime_modu_ac(self) -> None:
        """Kelime moduna geçiş yapar."""
        self.app.kelime_moduna_gec()


# ═════════════════════════════════════════════════════════════════════
# ORTA PANEL — Hedef Gösterimi, Grafikler, Kontroller
# ═════════════════════════════════════════════════════════════════════

class OrtaPanel(ctk.CTkFrame):
    """Orta panel: hedef gösterimi, matplotlib grafik alanı, kontrol butonları."""

    def __init__(self, master: Any, app: "HandwritingApp", **kwargs) -> None:
        super().__init__(master, corner_radius=0,
                         fg_color=COLORS["bg_dark"], **kwargs)
        self.app = app
        self.mevcut_canvas: Optional[FigureCanvasTkAgg] = None
        self.mevcut_figure: Optional[Figure] = None
        self._arayuz_olustur()

    def _arayuz_olustur(self) -> None:
        """Panel bileşenlerini oluşturur."""

        # ── Hedef gösterimi (üst kısım) ──
        self.hedef_frame = ctk.CTkFrame(
            self, height=100, fg_color=COLORS["bg_panel"], corner_radius=12,
        )
        self.hedef_frame.pack(fill="x", padx=15, pady=(10, 5))
        self.hedef_frame.pack_propagate(False)

        self.hedef_harf_label = ctk.CTkLabel(
            self.hedef_frame,
            text="O",
            font=ctk.CTkFont(family="Arial", size=52, weight="bold"),
            text_color=COLORS["accent"],
        )
        self.hedef_harf_label.pack(side="left", padx=(25, 15), pady=10)

        self.hedef_mesaj_frame = ctk.CTkFrame(
            self.hedef_frame, fg_color="transparent"
        )
        self.hedef_mesaj_frame.pack(side="left", fill="both", expand=True, pady=10)

        self.hedef_baslik = ctk.CTkLabel(
            self.hedef_mesaj_frame,
            text="Bu harfi yaz",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_primary"],
            anchor="w",
        )
        self.hedef_baslik.pack(anchor="w")

        self.hedef_aciklama = ctk.CTkLabel(
            self.hedef_mesaj_frame,
            text="Seviye 1 • Basit şekiller",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_muted"],
            anchor="w",
        )
        self.hedef_aciklama.pack(anchor="w")

        self.hedef_ipucu = ctk.CTkLabel(
            self.hedef_mesaj_frame,
            text="💡 Yuvarlak ve akıcı hareketlerle çiz",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
            anchor="w",
        )
        self.hedef_ipucu.pack(anchor="w", pady=(3, 0))

        # ── Grafik alanı (orta) ──
        self.grafik_frame = ctk.CTkFrame(
            self, fg_color=COLORS["chart_bg"], corner_radius=12,
        )
        self.grafik_frame.pack(fill="both", expand=True, padx=15, pady=5)

        # Başlangıç mesajı
        self.bekleme_label = ctk.CTkLabel(
            self.grafik_frame,
            text="✍️ Yazmaya başlamak için\naşağıdaki butona tıkla",
            font=ctk.CTkFont(size=18),
            text_color=COLORS["text_muted"],
        )
        self.bekleme_label.pack(expand=True)

        # ── Skor çubuğu ──
        self.skor_cubuk_frame = ctk.CTkFrame(
            self, height=30, fg_color=COLORS["bg_panel"], corner_radius=8,
        )
        self.skor_cubuk_frame.pack(fill="x", padx=15, pady=(0, 5))
        self.skor_cubuk_frame.pack_propagate(False)

        self.skor_cubuk_label = ctk.CTkLabel(
            self.skor_cubuk_frame,
            text="Henüz skor yok",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        self.skor_cubuk_label.pack(side="left", padx=10)

        self.skor_cubuk_bar = ctk.CTkProgressBar(
            self.skor_cubuk_frame,
            height=14,
            corner_radius=7,
            progress_color=COLORS["text_muted"],
        )
        self.skor_cubuk_bar.pack(side="right", fill="x", expand=True, padx=10, pady=8)
        self.skor_cubuk_bar.set(0)

        # ── Kontrol butonları (alt kısım) ──
        self.kontrol_frame = ctk.CTkFrame(
            self, height=70, fg_color="transparent",
        )
        self.kontrol_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.kontrol_frame.pack_propagate(False)

        self.yaz_btn = ctk.CTkButton(
            self.kontrol_frame,
            text="✍️  YAZMAYA BAŞLA",
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=50,
            corner_radius=12,
            command=self.app.yazmaya_basla,
        )
        self.yaz_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.kalibrasyon_btn = ctk.CTkButton(
            self.kontrol_frame,
            text="🎯 Kalibrasyon",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["bg_mid"],
            hover_color=COLORS["btn_active"],
            height=50,
            width=130,
            corner_radius=12,
            command=self.app.kalibrasyon_baslat,
        )
        self.kalibrasyon_btn.pack(side="right", padx=(5, 0))

        # ── Durum mesajı ──
        self.durum_label = ctk.CTkLabel(
            self,
            text="Hazır",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        self.durum_label.pack(pady=(0, 5))

    def hedef_guncelle(self, harf: str, mod: str = "letter") -> None:
        """Hedef gösterimini günceller."""
        if mod == "letter":
            self.hedef_harf_label.configure(text=harf)
            seviye = HARF_SEVIYE.get(harf, 1)
            seviye_adi = {
                1: "Basit şekiller", 2: "Düz çizgiler",
                3: "Açılı çizgiler", 4: "Kavisli + düz",
                5: "Karmaşık", 6: "Özel karakterler",
            }
            self.hedef_baslik.configure(text="Bu harfi yaz")
            self.hedef_aciklama.configure(
                text=f"Seviye {seviye} • {seviye_adi.get(seviye, '')}"
            )
        else:
            self.hedef_harf_label.configure(text=harf)
            self.hedef_baslik.configure(text="Bu kelimeyi yaz")
            self.hedef_aciklama.configure(text="Kelime Modu")

    def grafik_goster(self, figure: Figure) -> None:
        """Matplotlib figürünü grafik alanında gösterir."""
        # Önceki grafiği temizle
        self._grafik_temizle()

        # Bekleme etiketini gizle
        self.bekleme_label.pack_forget()

        self.mevcut_figure = figure
        self.mevcut_canvas = FigureCanvasTkAgg(figure, master=self.grafik_frame)
        self.mevcut_canvas.draw()
        widget = self.mevcut_canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=5, pady=5)

    def _grafik_temizle(self) -> None:
        """Mevcut grafiği temizler."""
        if self.mevcut_canvas is not None:
            self.mevcut_canvas.get_tk_widget().destroy()
            self.mevcut_canvas = None
        if self.mevcut_figure is not None:
            plt.close(self.mevcut_figure)
            self.mevcut_figure = None

    def skor_cubugu_guncelle(self, skor: float) -> None:
        """Skor çubuğunu günceller."""
        renk = skor_rengi(skor)
        self.skor_cubuk_bar.configure(progress_color=renk)
        self.skor_cubuk_bar.set(skor / 10.0)
        self.skor_cubuk_label.configure(
            text=f"Skor: {skor:.1f} / 10",
            text_color=renk,
        )

    def durum_mesaji(self, mesaj: str, renk: Optional[str] = None) -> None:
        """Durum mesajını günceller."""
        self.durum_label.configure(
            text=mesaj,
            text_color=renk or COLORS["text_muted"],
        )

    def geri_sayim_goster(self, sayi: str) -> None:
        """Geri sayım sayısını grafik alanında büyük gösterir."""
        self._grafik_temizle()
        self.bekleme_label.configure(
            text=sayi,
            font=ctk.CTkFont(size=72, weight="bold"),
            text_color=COLORS["accent"],
        )
        self.bekleme_label.pack(expand=True)

    def bekleme_goster(self, mesaj: str = "✍️ Yazıyor...") -> None:
        """Bekleme mesajı gösterir."""
        self._grafik_temizle()
        self.bekleme_label.configure(
            text=mesaj,
            font=ctk.CTkFont(size=24),
            text_color=COLORS["warning"],
        )
        self.bekleme_label.pack(expand=True)


# ═════════════════════════════════════════════════════════════════════
# SAĞ PANEL — Skor Göstergesi, Detaylar, Geri Bildirim
# ═════════════════════════════════════════════════════════════════════

class SagPanel(ctk.CTkFrame):
    """Sağ panel: büyük skor göstergesi, detay barlar, geri bildirim, mini grafik."""

    def __init__(self, master: Any, app: "HandwritingApp", **kwargs) -> None:
        super().__init__(master, width=310, corner_radius=0,
                         fg_color=COLORS["bg_panel"], **kwargs)
        self.app = app
        self.pack_propagate(False)
        self.kriter_barlar: Dict[str, Tuple[ctk.CTkProgressBar, ctk.CTkLabel]] = {}
        self.mini_fig: Optional[Figure] = None
        self.mini_canvas: Optional[FigureCanvasTkAgg] = None
        self._arayuz_olustur()

    def _arayuz_olustur(self) -> None:
        """Panel bileşenlerini oluşturur."""

        # ── Büyük skor göstergesi ──
        self.skor_frame = ctk.CTkFrame(
            self, height=130, fg_color=COLORS["bg_mid"], corner_radius=15,
        )
        self.skor_frame.pack(fill="x", padx=12, pady=(12, 5))
        self.skor_frame.pack_propagate(False)

        self.skor_deger = ctk.CTkLabel(
            self.skor_frame,
            text="—",
            font=ctk.CTkFont(family="Arial", size=52, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        self.skor_deger.pack(pady=(10, 0))

        self.skor_etiket = ctk.CTkLabel(
            self.skor_frame,
            text="Henüz deneme yapılmadı",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_muted"],
        )
        self.skor_etiket.pack()

        # ── Detay skorları ──
        ctk.CTkLabel(
            self, text="KRİTER DETAYLARI",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"],
        ).pack(pady=(10, 5))

        self.detay_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.detay_frame.pack(fill="x", padx=12)

        for anahtar, isim in KRITER_ISIMLERI.items():
            satir = ctk.CTkFrame(self.detay_frame, fg_color="transparent")
            satir.pack(fill="x", pady=2)

            lbl = ctk.CTkLabel(
                satir, text=isim, width=130, anchor="w",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_secondary"],
            )
            lbl.pack(side="left")

            bar = ctk.CTkProgressBar(
                satir, height=10, corner_radius=5, width=100,
                progress_color=COLORS["text_muted"],
            )
            bar.pack(side="left", padx=(5, 5), fill="x", expand=True)
            bar.set(0)

            deger_lbl = ctk.CTkLabel(
                satir, text="—", width=35,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS["text_muted"],
            )
            deger_lbl.pack(side="right")

            self.kriter_barlar[anahtar] = (bar, deger_lbl)

        # ── Geri bildirim ──
        ctk.CTkLabel(
            self, text="GERİ BİLDİRİM",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"],
        ).pack(pady=(12, 3))

        self.geri_bildirim_box = ctk.CTkTextbox(
            self,
            height=120,
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["bg_dark"],
            text_color=COLORS["text_primary"],
            corner_radius=8,
            wrap="word",
        )
        self.geri_bildirim_box.pack(fill="x", padx=12, pady=(0, 5))
        self.geri_bildirim_box.insert("1.0", "Henüz geri bildirim yok.\nBir deneme yap!")
        self.geri_bildirim_box.configure(state="disabled")

        # ── Son 5 deneme mini grafiği ──
        ctk.CTkLabel(
            self, text="SON DENEMELER",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"],
        ).pack(pady=(5, 3))

        self.mini_grafik_frame = ctk.CTkFrame(
            self, height=100, fg_color=COLORS["chart_bg"], corner_radius=8,
        )
        self.mini_grafik_frame.pack(fill="x", padx=12, pady=(0, 5))
        self.mini_grafik_frame.pack_propagate(False)

        self.mini_bekleme = ctk.CTkLabel(
            self.mini_grafik_frame,
            text="📈 Grafik için deneme yap",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"],
        )
        self.mini_bekleme.pack(expand=True)

        # ── Aksiyon butonları ──
        self.buton_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.buton_frame.pack(fill="x", padx=12, pady=(5, 10), side="bottom")

        self.tekrar_btn = ctk.CTkButton(
            self.buton_frame,
            text="🔄 Tekrar Dene",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["bg_mid"],
            hover_color=COLORS["btn_active"],
            height=38,
            command=self.app.tekrar_dene,
            state="disabled",
        )
        self.tekrar_btn.pack(fill="x", pady=2)

        self.sonraki_btn = ctk.CTkButton(
            self.buton_frame,
            text="➡️ Sonraki Harf",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=38,
            command=self.app.sonraki_harf,
            state="disabled",
        )
        self.sonraki_btn.pack(fill="x", pady=2)

    def skor_goster(self, sonuc: Dict[str, Any]) -> None:
        """Skor sonuçlarını panelde gösterir."""
        toplam = sonuc["total_score"]
        renk = skor_rengi(toplam)
        derece = sonuc.get("grade", "")

        # Büyük skor
        self.skor_deger.configure(
            text=f"{toplam:.1f}",
            text_color=renk,
        )

        gecti_str = "✅ GEÇTİ" if sonuc.get("pass", False) else "❌ TEKRAR DENE"
        self.skor_etiket.configure(
            text=f"{derece} — {gecti_str}",
            text_color=renk,
        )

        # Kriter detayları
        detay = sonuc.get("details", {})
        for anahtar, (bar, lbl) in self.kriter_barlar.items():
            deger = detay.get(anahtar, 0.0)
            r = skor_rengi(deger)
            bar.configure(progress_color=r)
            bar.set(deger / 10.0)
            lbl.configure(text=f"{deger:.1f}", text_color=r)

        # Geri bildirim
        geri_bildirimler = sonuc.get("feedback", [])
        self.geri_bildirim_box.configure(state="normal")
        self.geri_bildirim_box.delete("1.0", "end")
        for fb in geri_bildirimler:
            self.geri_bildirim_box.insert("end", f"• {fb}\n\n")
        self.geri_bildirim_box.configure(state="disabled")

        # Butonları aktifle
        self.tekrar_btn.configure(state="normal")
        self.sonraki_btn.configure(state="normal")

    def mini_grafik_guncelle(self, harf: str) -> None:
        """Son 5 denemenin mini skor grafiğini çizer."""
        # Önceki grafiği temizle
        if self.mini_canvas is not None:
            self.mini_canvas.get_tk_widget().destroy()
            self.mini_canvas = None
        if self.mini_fig is not None:
            plt.close(self.mini_fig)
            self.mini_fig = None

        skorlar = self.app.progress.letter_scores.get(harf.upper(), [])
        if not skorlar:
            self.mini_bekleme.pack(expand=True)
            return

        self.mini_bekleme.pack_forget()
        son_5 = skorlar[-5:]

        fig, ax = plt.subplots(figsize=(3.2, 1.0), facecolor=COLORS["chart_bg"])
        ax.set_facecolor(COLORS["chart_bg"])

        x = range(1, len(son_5) + 1)
        renkler = [skor_rengi(s) for s in son_5]
        ax.bar(x, son_5, color=renkler, width=0.6, edgecolor="none")
        ax.axhline(y=7.0, color=COLORS["success"], linewidth=0.8,
                    linestyle="--", alpha=0.5)
        ax.set_ylim(0, 10.5)
        ax.set_xticks(list(x))
        ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["text_muted"])
        ax.spines["bottom"].set_color(COLORS["text_muted"])
        fig.tight_layout(pad=0.3)

        self.mini_fig = fig
        self.mini_canvas = FigureCanvasTkAgg(fig, master=self.mini_grafik_frame)
        self.mini_canvas.draw()
        self.mini_canvas.get_tk_widget().pack(fill="both", expand=True)


# ═════════════════════════════════════════════════════════════════════
# AYARLAR PENCERESİ
# ═════════════════════════════════════════════════════════════════════

class AyarlarPenceresi(ctk.CTkToplevel):
    """Ayarlar penceresi: COM port, basınç eşiği, sıfırlama."""

    def __init__(self, master: "HandwritingApp", **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.app = master
        self.title("⚙️ Ayarlar")
        self.geometry("400x500")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self._arayuz_olustur()

    def _arayuz_olustur(self) -> None:
        """Ayarlar arayüzünü oluşturur."""
        ana = ctk.CTkFrame(self, fg_color="transparent")
        ana.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        ctk.CTkLabel(
            ana, text="Uygulama Ayarları",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(0, 15))

        # ── COM Port ──
        ctk.CTkLabel(ana, text="Sensör COM Port:", anchor="w").pack(
            fill="x", pady=(10, 2)
        )
        self.com_var = ctk.StringVar(value=self.app.com_port)
        port_frame = ctk.CTkFrame(ana, fg_color="transparent")
        port_frame.pack(fill="x")
        self.com_entry = ctk.CTkEntry(
            port_frame, textvariable=self.com_var, width=200,
        )
        self.com_entry.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            port_frame, text="(ör: COM3, /dev/ttyUSB0)",
            text_color=COLORS["text_muted"], font=ctk.CTkFont(size=11),
        ).pack(side="left")

        # ── Baud Rate ──
        ctk.CTkLabel(ana, text="Baud Rate:", anchor="w").pack(
            fill="x", pady=(10, 2)
        )
        self.baud_var = ctk.StringVar(value=str(self.app.baud_rate))
        self.baud_entry = ctk.CTkEntry(
            ana, textvariable=self.baud_var, width=200,
        )
        self.baud_entry.pack(anchor="w")

        # ── Basınç eşiği ──
        ctk.CTkLabel(ana, text="Basınç Eşiği:", anchor="w").pack(
            fill="x", pady=(15, 2)
        )
        self.basinc_var = ctk.DoubleVar(value=self.app.pressure_threshold)
        self.basinc_label = ctk.CTkLabel(
            ana, text=f"{self.app.pressure_threshold:.2f}",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.basinc_label.pack()
        self.basinc_slider = ctk.CTkSlider(
            ana, from_=0.05, to=1.0,
            variable=self.basinc_var,
            command=self._basinc_degisti,
        )
        self.basinc_slider.pack(fill="x", pady=5)

        # ── Kullanıcı adı ──
        ctk.CTkLabel(ana, text="Kullanıcı Adı:", anchor="w").pack(
            fill="x", pady=(15, 2)
        )
        self.isim_var = ctk.StringVar(value=self.app.progress.username)
        self.isim_entry = ctk.CTkEntry(
            ana, textvariable=self.isim_var, width=250,
        )
        self.isim_entry.pack(anchor="w")

        # ── Butonlar ──
        btn_frame = ctk.CTkFrame(ana, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(25, 0))

        ctk.CTkButton(
            btn_frame, text="💾 Kaydet",
            fg_color=COLORS["success"],
            hover_color=COLORS["success_light"],
            command=self._kaydet,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame, text="🗑️ İlerlemeyi Sıfırla",
            fg_color=COLORS["error"],
            hover_color="#E53935",
            command=self._sifirla,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame, text="İptal",
            fg_color=COLORS["btn_locked"],
            command=self.destroy,
        ).pack(side="right")

    def _basinc_degisti(self, deger: float) -> None:
        self.basinc_label.configure(text=f"{deger:.2f}")

    def _kaydet(self) -> None:
        self.app.com_port = self.com_var.get()
        try:
            self.app.baud_rate = int(self.baud_var.get())
        except ValueError:
            pass
        self.app.pressure_threshold = self.basinc_var.get()
        self.app.progress.username = self.isim_var.get()
        self.app.progress.kaydet()

        # Ayarları dosyaya kaydet
        ayarlar = {
            "com_port": self.app.com_port,
            "baud_rate": self.app.baud_rate,
            "pressure_threshold": self.app.pressure_threshold,
        }
        try:
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(ayarlar, f, indent=2)
        except Exception:
            pass

        self.app.sol_panel.guncelle()
        self.destroy()

    def _sifirla(self) -> None:
        try:
            dialog = CTkMessagebox(
                title="Onay",
                message="Tüm ilerleme sıfırlanacak!\nEmin misin?",
                icon="warning",
                option_1="Evet, Sıfırla",
                option_2="İptal",
            )
            if dialog.get() == "Evet, Sıfırla":
                self.app.progress.profil_sifirla()
                self.app.sol_panel.guncelle()
                self.destroy()
        except Exception:
            self.app.progress.profil_sifirla()
            self.app.sol_panel.guncelle()
            self.destroy()


# ═════════════════════════════════════════════════════════════════════
# REFERANS KAYDET DİALOG
# ═════════════════════════════════════════════════════════════════════

class ReferansKaydetDialog(ctk.CTkToplevel):
    """Referans harf kaydı penceresi."""

    def __init__(self, master: "HandwritingApp", **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.app = master
        self.title("💾 Referans Kaydet")
        self.geometry("350x300")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self._arayuz_olustur()

    def _arayuz_olustur(self) -> None:
        ana = ctk.CTkFrame(self, fg_color="transparent")
        ana.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            ana, text="Referans Harf Kaydı",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(0, 15))

        ctk.CTkLabel(ana, text="Kaydetmek istediğin harfi seç:").pack(pady=(0, 5))

        self.harf_var = ctk.StringVar(value=self.app.hedef_harf)
        self.harf_menu = ctk.CTkOptionMenu(
            ana,
            variable=self.harf_var,
            values=TUM_HARFLER,
            width=200,
        )
        self.harf_menu.pack(pady=5)

        self.durum_lbl = ctk.CTkLabel(
            ana, text="Harfi seç ve kayıt başlat",
            text_color=COLORS["text_muted"],
        )
        self.durum_lbl.pack(pady=10)

        ctk.CTkButton(
            ana,
            text="🎙️ Kayıt Başlat",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=40,
            command=self._kayit_baslat,
        ).pack(fill="x", pady=5)

        ctk.CTkButton(
            ana,
            text="İptal",
            fg_color=COLORS["btn_locked"],
            command=self.destroy,
        ).pack(fill="x")

    def _kayit_baslat(self) -> None:
        harf = self.harf_var.get()
        self.durum_lbl.configure(
            text="Geri sayım... Hazır ol!",
            text_color=COLORS["warning"],
        )
        self.update()

        # Geri sayım
        for i in [3, 2, 1]:
            self.durum_lbl.configure(text=f"  {i}  ")
            self.update()
            time.sleep(0.7)

        self.durum_lbl.configure(text="✍️ ŞİMDİ YAZ!", text_color=COLORS["success"])
        self.update()

        def _kayit_thread():
            try:
                veri = get_sensor_data()
                if veri is not None and veri.size > 0:
                    yol = referans_kaydet(harf, veri)
                    self.after(0, lambda: self.durum_lbl.configure(
                        text=f"✅ Kaydedildi: {yol}",
                        text_color=COLORS["success"],
                    ))
                else:
                    self.after(0, lambda: self.durum_lbl.configure(
                        text="❌ Veri alınamadı",
                        text_color=COLORS["error"],
                    ))
            except Exception as e:
                self.after(0, lambda: self.durum_lbl.configure(
                    text=f"❌ Hata: {str(e)[:30]}",
                    text_color=COLORS["error"],
                ))

        threading.Thread(target=_kayit_thread, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ═════════════════════════════════════════════════════════════════════

class HandwritingApp(ctk.CTk):
    """El Yazısı Öğrenme Uygulaması — Ana Pencere.

    IMU sensör verisini DTW mesafesiyle analiz ederek [1] kullanıcının
    el yazısı kalitesini değerlendirir. 6 seviyeli ilerleme sistemi,
    kelime modu ve detaylı geri bildirim sunar.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Tema ayarları ──
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Pencere ayarları ──
        self.title(APP_TITLE)
        self.geometry(APP_GEOMETRY)
        self.minsize(*MIN_SIZE)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── Uygulama durumu ──
        self.hedef_harf: str = "O"
        self.hedef_kelime: str = ""
        self.aktif_mod: str = "letter"  # "letter" veya "word"
        self.is_recording: bool = False
        self.is_processing: bool = False
        self.kalibre_edildi: bool = False

        # ── Ayarlar ──
        self.com_port: str = DEFAULT_COM_PORT
        self.baud_rate: int = DEFAULT_BAUD_RATE
        self.pressure_threshold: float = DEFAULT_PRESSURE_THRESHOLD
        self._ayarlari_yukle()

        # ── Son sonuçlar ──
        self.son_skor_sonuc: Optional[Dict[str, Any]] = None
        self.son_kullanici_verisi: Optional[np.ndarray] = None

        # ── Progress Manager ──
        self.progress = ProgressManager(
            username="Kullanıcı",
            profil_dosyasi="user_progress.json",
            otomatik_yukle=True,
        )

        # İlk hedefi belirle
        gorev = self.progress.get_next_task()
        self.hedef_harf = gorev.get("target", "O")

        # ── Arayüz oluştur ──
        self._arayuz_olustur()

        # İlk güncelleme
        self.after(100, self._ilk_yukleme)

    def _ayarlari_yukle(self) -> None:
        """settings.json'dan ayarları yükler."""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r", encoding="utf-8") as f:
                    ayarlar = json.load(f)
                self.com_port = ayarlar.get("com_port", self.com_port)
                self.baud_rate = ayarlar.get("baud_rate", self.baud_rate)
                self.pressure_threshold = ayarlar.get(
                    "pressure_threshold", self.pressure_threshold
                )
        except Exception:
            pass

    def _arayuz_olustur(self) -> None:
        """Ana arayüz düzenini oluşturur: sol, orta, sağ paneller."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sol Panel ──
        self.sol_panel = SolPanel(self, app=self)
        self.sol_panel.grid(row=0, column=0, sticky="nsew")

        # ── Orta Panel ──
        self.orta_panel = OrtaPanel(self, app=self)
        self.orta_panel.grid(row=0, column=1, sticky="nsew")

        # ── Sağ Panel ──
        self.sag_panel = SagPanel(self, app=self)
        self.sag_panel.grid(row=0, column=2, sticky="nsew")

    def _ilk_yukleme(self) -> None:
        """Program açılışında ilk yapılacak işlemler."""
        self.orta_panel.hedef_guncelle(self.hedef_harf, self.aktif_mod)
        self.sol_panel.guncelle()

        # Referans şeklini göster
        self._referans_goster()

        # Kalibrasyon durumunu kontrol et
        self.orta_panel.durum_mesaji(
            "💡 İlk kullanımda kalibrasyon yapmanızı öneririz.",
            COLORS["warning"],
        )

    def _referans_goster(self) -> None:
        """Hedef harf için referans yörünge grafiğini gösterir."""
        # Referans verisini yükle
        ref_data = referans_yukle(self.hedef_harf)
        if ref_data is not None:
            try:
                rx, ry, rp, rv = calculate_trajectory(ref_data)
                fig = plot_trajectory(
                    rx, ry, rp, rv,
                    title=f"Referans: {self.hedef_harf}",
                )
                self.orta_panel.grafik_goster(fig)
                return
            except Exception:
                pass

        # Şablon referans
        try:
            rx, ry = get_reference_shape(self.hedef_harf)
            if rx.size > 0:
                fig = plot_trajectory(
                    rx, ry, title=f"Referans: {self.hedef_harf} (şablon)"
                )
                self.orta_panel.grafik_goster(fig)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # Harf / Kelime Seçimi
    # ─────────────────────────────────────────────────────────────────

    def harf_sec(self, harf: str) -> None:
        """Kullanıcı sol panelden harf seçtiğinde çağrılır."""
        if self.is_recording or self.is_processing:
            return

        self.hedef_harf = harf.upper()
        self.aktif_mod = "letter"
        self.orta_panel.hedef_guncelle(self.hedef_harf, "letter")
        self.sol_panel.guncelle()
        self._referans_goster()
        self.sag_panel.mini_grafik_guncelle(self.hedef_harf)

    def kelime_moduna_gec(self) -> None:
        """Kelime moduna geçiş yapar."""
        if not self.progress.kelime_modu_acik_mi():
            return

        kelimeler = self.progress.kullanilabilir_kelimeler()
        if not kelimeler:
            self.orta_panel.durum_mesaji(
                "Uygun kelime bulunamadı", COLORS["warning"]
            )
            return

        self.aktif_mod = "word"
        self.hedef_kelime = kelimeler[0]
        self.orta_panel.hedef_guncelle(self.hedef_kelime, "word")
        self.orta_panel.durum_mesaji(
            f"Kelime modu: '{self.hedef_kelime}' yazılacak",
            COLORS["btn_active"],
        )

    def sonraki_harf(self) -> None:
        """İlerleme yöneticisinden sonraki görevi alıp gösterir."""
        gorev = self.progress.get_next_task()
        if gorev["type"] == "letter":
            self.harf_sec(gorev["target"])
        else:
            self.aktif_mod = "word"
            self.hedef_kelime = gorev["target"]
            self.orta_panel.hedef_guncelle(self.hedef_kelime, "word")

        self.orta_panel.durum_mesaji(
            f"🎯 {gorev.get('reason', '')}",
            COLORS["text_secondary"],
        )

    def tekrar_dene(self) -> None:
        """Aynı hedefi tekrar denemek için arayüzü sıfırlar."""
        if self.aktif_mod == "letter":
            self._referans_goster()
        self.orta_panel.durum_mesaji("Hazır — tekrar dene!", COLORS["text_secondary"])

    # ─────────────────────────────────────────────────────────────────
    # Yazma Akışı
    # ─────────────────────────────────────────────────────────────────

    def yazmaya_basla(self) -> None:
        """'Yazmaya Başla' butonuna tıklanınca çağrılır.

        Geri sayım gösterir, sensör verisini ayrı thread'de okur,
        işleme pipeline'ını çalıştırır. DTW karşılaştırması
        dtw_analysis.py modülü tarafından yapılır [1].
        """
        if self.is_recording or self.is_processing:
            return

        self.is_recording = True
        self.orta_panel.yaz_btn.configure(state="disabled")
        self.sag_panel.tekrar_btn.configure(state="disabled")
        self.sag_panel.sonraki_btn.configure(state="disabled")

        # Geri sayımı başlat
        self._geri_sayim(3)

    def _geri_sayim(self, kalan: int) -> None:
        """Geri sayım (GUI donmadan, self.after ile)."""
        if kalan > 0:
            self.orta_panel.geri_sayim_goster(str(kalan))
            self.after(800, lambda: self._geri_sayim(kalan - 1))
        else:
            self.orta_panel.geri_sayim_goster("✍️ YAZ!")
            self.after(500, self._kayit_baslat)

    def _kayit_baslat(self) -> None:
        """Sensör verisini ayrı thread'de okumaya başlar."""
        self.orta_panel.bekleme_goster("✍️ Yazıyor... Harfi tamamla")
        self.orta_panel.durum_mesaji("Sensör verisi okunuyor...", COLORS["warning"])

        def _sensor_thread():
            try:
                veri = get_sensor_data()
                self.after(0, lambda: self._veri_alindi(veri))
            except Exception as e:
                self.after(0, lambda: self._veri_hatasi(str(e)))

        threading.Thread(target=_sensor_thread, daemon=True).start()

    def _veri_alindi(self, veri: Optional[np.ndarray]) -> None:
        """Sensör verisi başarıyla alındığında çağrılır."""
        self.is_recording = False

        if veri is None or veri.size == 0:
            self._veri_hatasi("Sensör verisi alınamadı.")
            return

        self.son_kullanici_verisi = veri
        self.is_processing = True
        self.orta_panel.durum_mesaji("İşleniyor...", COLORS["btn_active"])

        # İşleme pipeline'ını ayrı thread'de çalıştır
        def _isleme_thread():
            try:
                sonuc = self._pipeline_calistir(veri)
                self.after(0, lambda: self._sonuc_goster(sonuc))
            except Exception as e:
                self.after(0, lambda: self._veri_hatasi(str(e)))

        threading.Thread(target=_isleme_thread, daemon=True).start()

    def _pipeline_calistir(self, veri: np.ndarray) -> Dict[str, Any]:
        """Tam işleme pipeline'ını çalıştırır.

        Pipeline: ön işleme → DTW analizi [1] → skorlama → görselleştirme.

        Args:
            veri: Ham sensör verisi, shape (N, 6+).

        Returns:
            Pipeline sonuç sözlüğü.
        """
        sonuc: Dict[str, Any] = {}

        # 1. Ön işleme
        filtreli = filter_data(veri)
        vuruslar = segment_strokes(filtreli)
        normalizeli = normalize_data(filtreli)

        # 2. Referans verisi yükle
        ref_data = referans_yukle(self.hedef_harf)

        # 3. DTW analizi — [1]'deki fonksiyonlar kullanılır
        if ref_data is not None:
            dtw_mesafe = compare_with_reference(normalizeli, ref_data)
            tanimlama = identify_letter(normalizeli)
        else:
            dtw_mesafe = 50.0  # Referans yoksa yüksek mesafe
            tanimlama = {"letter": self.hedef_harf, "confidence": 0.5}

        sonuc["tanimlama"] = tanimlama
        sonuc["dtw_mesafe"] = dtw_mesafe

        # 4. Referans vuruşları
        if ref_data is not None:
            ref_vuruslar = segment_strokes(ref_data)
        else:
            ref_vuruslar = [(0, 10)]

        # 5. Skorlama
        guven = tanimlama.get("confidence", 0.5)
        ref_for_scoring = ref_data if ref_data is not None else normalizeli
        skor_sonuc = calculate_quality_score(
            user_data=normalizeli,
            ref_data=ref_for_scoring,
            user_strokes=vuruslar,
            ref_strokes=ref_vuruslar,
            dtw_distance=dtw_mesafe,
            identification_confidence=guven,
        )
        sonuc["skor"] = skor_sonuc

        # 6. Yörünge hesaplama
        try:
            user_px, user_py, user_pr, user_vr = calculate_trajectory(filtreli)
        except Exception:
            user_px = np.cumsum(filtreli[:, 0]) * 0.001
            user_py = np.cumsum(filtreli[:, 1]) * 0.001
            user_pr = filtreli[:, 6] if filtreli.shape[1] > 6 else np.ones(len(filtreli))
            user_vr = [(0, len(filtreli) - 1)]

        sonuc["user_traj"] = (user_px, user_py, user_pr, user_vr)

        # 7. Referans yörüngesi
        if ref_data is not None:
            try:
                ref_px, ref_py, ref_pr, ref_vr = calculate_trajectory(ref_data)
            except Exception:
                ref_px, ref_py = get_reference_shape(self.hedef_harf)
                ref_pr = np.ones(len(ref_px))
                ref_vr = [(0, len(ref_px) - 1)]
        else:
            ref_px, ref_py = get_reference_shape(self.hedef_harf)
            ref_pr = np.ones(max(1, len(ref_px)))
            ref_vr = [(0, max(0, len(ref_px) - 1))]

        sonuc["ref_traj"] = (ref_px, ref_py)

        # 8. Görselleştirme figürü oluştur
        try:
            toplam_skor = skor_sonuc["total_score"]
            fig = plot_comparison(
                ref_px, ref_py, user_px, user_py,
                score=toplam_skor,
            )
        except Exception:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
            a1.plot(ref_px, ref_py, "gray", lw=2)
            a1.set_title("Referans")
            a1.set_aspect("equal")
            a2.plot(user_px, user_py, "b-", lw=2)
            a2.set_title("Senin Yazın")
            a2.set_aspect("equal")
            fig.tight_layout()

        sonuc["figure"] = fig

        return sonuc

    def _sonuc_goster(self, sonuc: Dict[str, Any]) -> None:
        """Pipeline sonucunu GUI'de gösterir ve ilerlemeyi kaydeder."""
        self.is_processing = False
        skor_sonuc = sonuc.get("skor", {})
        toplam_skor = skor_sonuc.get("total_score", 0)

        # Grafiği göster
        fig = sonuc.get("figure")
        if fig is not None:
            self.orta_panel.grafik_goster(fig)

        # Skor çubuğunu güncelle
        self.orta_panel.skor_cubugu_guncelle(toplam_skor)

        # Sağ panel skor gösterimi
        self.sag_panel.skor_goster(skor_sonuc)

        # İlerlemeye kaydet
        if self.aktif_mod == "letter":
            kayit_sonuc = self.progress.deneme_kaydet(
                self.hedef_harf, toplam_skor
            )
            self.orta_panel.durum_mesaji(
                kayit_sonuc.get("mesaj", "").split("\n")[0],
                skor_rengi(toplam_skor),
            )
        else:
            kayit_sonuc = self.progress.kelime_deneme_kaydet(
                self.hedef_kelime, toplam_skor
            )
            self.orta_panel.durum_mesaji(
                kayit_sonuc.get("mesaj", ""),
                skor_rengi(toplam_skor),
            )

        # Mini grafik güncelle
        self.sag_panel.mini_grafik_guncelle(self.hedef_harf)

        # Sol panel güncelle
        self.sol_panel.guncelle()

        # Butonları aktifle
        self.orta_panel.yaz_btn.configure(state="normal")
        self.son_skor_sonuc = skor_sonuc

    def _veri_hatasi(self, mesaj: str) -> None:
        """Veri okuma/işleme hatasını gösterir."""
        self.is_recording = False
        self.is_processing = False
        self.orta_panel.yaz_btn.configure(state="normal")
        self.orta_panel.durum_mesaji(f"❌ {mesaj}", COLORS["error"])
        self.orta_panel.bekleme_goster(f"❌ Hata\n{mesaj}")

    # ─────────────────────────────────────────────────────────────────
    # Kalibrasyon
    # ─────────────────────────────────────────────────────────────────

    def kalibrasyon_baslat(self) -> None:
        """Sensör kalibrasyonunu başlatır."""
        if self.is_recording:
            return

        self.orta_panel.durum_mesaji(
            "🎯 Kalibrasyon — Kalemi sabit tut...", COLORS["warning"]
        )
        self.orta_panel.bekleme_goster("🎯 Kalibrasyon\nKalemi sabit tut...")

        def _kal_thread():
            try:
                basarili = sensor_calibrate()
                if basarili:
                    self.kalibre_edildi = True
                    self.after(0, lambda: self.orta_panel.durum_mesaji(
                        "✅ Kalibrasyon başarılı!", COLORS["success"]
                    ))
                    self.after(0, self._referans_goster)
                else:
                    self.after(0, lambda: self.orta_panel.durum_mesaji(
                        "❌ Kalibrasyon başarısız", COLORS["error"]
                    ))
            except Exception as e:
                self.after(0, lambda: self.orta_panel.durum_mesaji(
                    f"❌ Kalibrasyon hatası: {e}", COLORS["error"]
                ))

        threading.Thread(target=_kal_thread, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────
    # Yardımcı Pencereler
    # ─────────────────────────────────────────────────────────────────

    def referans_kaydet_dialog(self) -> None:
        """Referans kaydetme penceresini açar."""
        ReferansKaydetDialog(self)

    def ayarlar_ac(self) -> None:
        """Ayarlar penceresini açar."""
        AyarlarPenceresi(self)

    def rapor_goster(self) -> None:
        """İlerleme raporunu ayrı pencerede gösterir."""
        rapor_pencere = ctk.CTkToplevel(self)
        rapor_pencere.title("📊 İlerleme Raporu")
        rapor_pencere.geometry("550x600")
        rapor_pencere.transient(self)

        metin = ctk.CTkTextbox(
            rapor_pencere,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
        )
        metin.pack(fill="both", expand=True, padx=10, pady=10)
        metin.insert("1.0", self.progress.ozet_rapor())
        metin.configure(state="disabled")

    # ─────────────────────────────────────────────────────────────────
    # Kapanış
    # ─────────────────────────────────────────────────────────────────

    def on_closing(self) -> None:
        """Uygulama kapanırken çağrılır — profili kaydet, temizle."""
        try:
            self.progress.kaydet()
        except Exception:
            pass
        plt.close("all")
        self.destroy()
        sys.exit(0)


# ═════════════════════════════════════════════════════════════════════
# UYGULAMA BAŞLATICI
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    """Uygulamayı başlatır."""
    # letters_db dizinini oluştur
    os.makedirs(LETTERS_DB, exist_ok=True)

    print("=" * 50)
    print(f"  {APP_TITLE}")
    print("  DTW Tabanlı El Yazısı Analiz Sistemi")
    print("=" * 50)
    print("  Modüller yükleniyor...")

    app = HandwritingApp()
    print("  ✅ Uygulama hazır!")
    print("=" * 50)
    app.mainloop()


if __name__ == "__main__":
    main()