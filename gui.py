"""
gui.py — El Yazısı Öğrenme Uygulaması Ana Arayüzü

CustomTkinter + Matplotlib tabanlı.
Buton ile kayıt, DTW tabanlı analiz, ilerleme takibi.
"""

import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import os
import sys
import time
import threading
import tkinter.simpledialog
import tkinter.messagebox

from sensor_data import get_sensor_data, connect_serial
try:
    from sensor_data import calibrate as sensor_calibrate
except ImportError:
    sensor_calibrate = None

from dtw_analysis import (
    preprocess_data,
    calculate_score,
    calculate_2d_trajectory,
    load_references,
    save_reference as dtw_save_reference,
    identify_letter,
)
from preprocessing import process_pipeline, filter_data
from scoring import calculate_quality_score

try:
    from progress import ProgressManager
    HAS_PROGRESS = True
except ImportError:
    HAS_PROGRESS = False

try:
    from trajectory import plot_trajectory, plot_comparison
    HAS_TRAJECTORY = True
except ImportError:
    HAS_TRAJECTORY = False

# ── Tema ──
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "accent": "#0f3460",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "text": "#ecf0f1",
    "text_muted": "#7f8c8d",
    "chart_bg": "#1e1e2e",
}


def skor_rengi(skor: float) -> str:
    """Skora göre renk döndürür."""
    if skor >= 8:
        return COLORS["success"]
    elif skor >= 5:
        return COLORS["warning"]
    else:
        return COLORS["danger"]


class HandwritingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Handwriting Studio v4.0 — Buton Tabanlı")
        self.geometry("1300x850")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Veri
        self.references = load_references()
        self.ser = None  # Seri bağlantı
        self.is_recording = False
        self.current_target = None
        self.last_score = None

        # İlerleme yöneticisi
        if HAS_PROGRESS:
            self.progress = ProgressManager()
            self.progress.yukle()
        else:
            self.progress = None

        # Arayüz oluştur
        self._build_ui()
        self._refresh_letter_buttons()

    # ──────────────────────────────────────
    # Arayüz Oluşturma
    # ──────────────────────────────────────

    def _build_ui(self):
        """Ana arayüzü oluşturur."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sol Panel ──
        self.sol_panel = ctk.CTkFrame(self, width=260, fg_color=COLORS["panel"])
        self.sol_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.sol_panel.grid_propagate(False)

        ctk.CTkLabel(
            self.sol_panel, text="📝 Harfler",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(15, 5))

        # İlerleme çubuğu
        self.ilerleme_label = ctk.CTkLabel(
            self.sol_panel, text="İlerleme: 0%",
            font=ctk.CTkFont(size=12),
        )
        self.ilerleme_label.pack(pady=(5, 2))

        self.ilerleme_bar = ctk.CTkProgressBar(self.sol_panel, width=220)
        self.ilerleme_bar.pack(pady=(0, 10))
        self.ilerleme_bar.set(0)

        # Harf butonları çerçevesi
        self.harf_frame = ctk.CTkScrollableFrame(
            self.sol_panel, width=230, height=400,
            fg_color=COLORS["panel"],
        )
        self.harf_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Alt butonlar
        ctk.CTkButton(
            self.sol_panel, text="📥 Referans Kaydet",
            command=self._referans_kaydet_dialog,
            fg_color=COLORS["accent"],
        ).pack(fill="x", padx=10, pady=(5, 2))

        ctk.CTkButton(
            self.sol_panel, text="🔧 Kalibrasyon",
            command=self._kalibrasyon_yap,
            fg_color=COLORS["accent"],
        ).pack(fill="x", padx=10, pady=(2, 2))

        ctk.CTkButton(
            self.sol_panel, text="🔄 Referansları Yenile",
            command=self._referanslari_yenile,
            fg_color=COLORS["accent"],
        ).pack(fill="x", padx=10, pady=(2, 10))

        # ── Orta Panel ──
        self.orta_panel = ctk.CTkFrame(self, fg_color=COLORS["bg"])
        self.orta_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.orta_panel.grid_rowconfigure(1, weight=1)
        self.orta_panel.grid_columnconfigure(0, weight=1)

        # Hedef gösterimi
        self.hedef_frame = ctk.CTkFrame(self.orta_panel, fg_color=COLORS["panel"], height=80)
        self.hedef_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.hedef_label = ctk.CTkLabel(
            self.hedef_frame,
            text="Bir harf seçin veya 'Yazmaya Başla' butonuna basın",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.hedef_label.pack(pady=15)

        # Grafik alanı
        self.grafik_frame = ctk.CTkFrame(
            self.orta_panel, fg_color=COLORS["chart_bg"], corner_radius=12,
        )
        self.grafik_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Matplotlib figure
        self.fig, (self.ax_ref, self.ax_user) = plt.subplots(
            1, 2, figsize=(10, 5), facecolor=COLORS["chart_bg"]
        )
        self.fig.subplots_adjust(wspace=0.3)
        self._temiz_grafik()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.grafik_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Alt kontrol çubuğu
        self.kontrol_frame = ctk.CTkFrame(self.orta_panel, fg_color=COLORS["panel"], height=60)
        self.kontrol_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))

        self.basla_btn = ctk.CTkButton(
            self.kontrol_frame, text="🖊️ Yazmaya Başla",
            command=self._yazmaya_basla,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45, fg_color="#2980b9",
        )
        self.basla_btn.pack(side="left", padx=20, pady=8)

        self.durum_label = ctk.CTkLabel(
            self.kontrol_frame,
            text="Hazır. Butona basarak yazın.",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_muted"],
        )
        self.durum_label.pack(side="left", padx=20, pady=8)

        # ── Sağ Panel ──
        self.sag_panel = ctk.CTkFrame(self, width=300, fg_color=COLORS["panel"])
        self.sag_panel.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.sag_panel.grid_propagate(False)

        ctk.CTkLabel(
            self.sag_panel, text="📊 Skor",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(15, 5))

        # Büyük skor göstergesi
        self.skor_label = ctk.CTkLabel(
            self.sag_panel, text="—",
            font=ctk.CTkFont(size=64, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        self.skor_label.pack(pady=(5, 5))

        self.not_label = ctk.CTkLabel(
            self.sag_panel, text="",
            font=ctk.CTkFont(size=16),
        )
        self.not_label.pack(pady=(0, 10))

        # Geri bildirim
        ctk.CTkLabel(
            self.sag_panel, text="Geri Bildirim:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(10, 2))

        self.feedback_text = ctk.CTkTextbox(
            self.sag_panel, height=200,
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["bg"],
        )
        self.feedback_text.pack(fill="x", padx=15, pady=(0, 10))

        # Detay skorları
        ctk.CTkLabel(
            self.sag_panel, text="Detay Skorları:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(5, 2))

        self.detay_frame = ctk.CTkFrame(self.sag_panel, fg_color=COLORS["bg"])
        self.detay_frame.pack(fill="x", padx=15, pady=(0, 10))

        # Tanıma sonucu
        self.tanima_label = ctk.CTkLabel(
            self.sag_panel, text="",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text_muted"],
        )
        self.tanima_label.pack(pady=(5, 10))

    # ──────────────────────────────────────
    # Grafik Yardımcıları
    # ──────────────────────────────────────

    def _temiz_grafik(self):
        """Grafikleri temizler."""
        for ax in [self.ax_ref, self.ax_user]:
            ax.clear()
            ax.set_facecolor(COLORS["chart_bg"])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        self.ax_ref.set_title("Referans", color=COLORS["text"], fontsize=12)
        self.ax_user.set_title("Senin Yazın", color=COLORS["text"], fontsize=12)

    def _grafik_guncelle(self, ref_data: np.ndarray, user_data: np.ndarray, skor: float):
        """Grafikleri günceller — referans ve kullanıcı izleri."""
        self._temiz_grafik()

        # Referans iz
        try:
            ref_x, ref_y = calculate_2d_trajectory(ref_data)
            self.ax_ref.plot(ref_x, ref_y, color="#2ecc71", linewidth=2)
            self.ax_ref.set_aspect("equal")
            self.ax_ref.set_title("Referans", color=COLORS["text"], fontsize=12)
        except Exception as e:
            self.ax_ref.text(0.5, 0.5, f"Hata: {e}", transform=self.ax_ref.transAxes,
                            ha="center", color="red", fontsize=9)

        # Kullanıcı izi
        try:
            user_x, user_y = calculate_2d_trajectory(user_data)
            renk = skor_rengi(skor / 10.0 if skor > 10 else skor)
            self.ax_user.plot(user_x, user_y, color=renk, linewidth=2)
            self.ax_user.set_aspect("equal")
            self.ax_user.set_title(f"Senin Yazın", color=COLORS["text"], fontsize=12)
        except Exception as e:
            self.ax_user.text(0.5, 0.5, f"Hata: {e}", transform=self.ax_user.transAxes,
                            ha="center", color="red", fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()

    # ──────────────────────────────────────
    # Harf Butonları
    # ──────────────────────────────────────

    def _refresh_letter_buttons(self):
        """Harf butonlarını günceller."""
        for widget in self.harf_frame.winfo_children():
            widget.destroy()

        # Mevcut referansları göster
        harfler = sorted(self.references.keys())

        if not harfler:
            ctk.CTkLabel(
                self.harf_frame,
                text="Henüz referans yok.\n'Referans Kaydet' ile\nbir harf kaydedin.",
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_muted"],
            ).pack(pady=20)
            return

        # Izgara düzeninde harfler (6 sütun)
        cols = 5
        for i, harf in enumerate(harfler):
            row = i // cols
            col = i % cols

            ref_sayisi = len(self.references.get(harf, []))
            btn_text = f"{harf}\n({ref_sayisi})"

            btn = ctk.CTkButton(
                self.harf_frame, text=btn_text, width=45, height=45,
                font=ctk.CTkFont(size=11),
                command=lambda h=harf: self._harf_sec(h),
                fg_color=COLORS["accent"],
            )
            btn.grid(row=row, column=col, padx=2, pady=2)

        # İlerleme güncelle
        toplam = 29  # Türk alfabesi
        acik = len(harfler)
        oran = min(acik / toplam, 1.0)
        self.ilerleme_bar.set(oran)
        self.ilerleme_label.configure(text=f"İlerleme: {acik}/{toplam} harf")

    def _harf_sec(self, harf: str):
        """Bir harf seçildiğinde."""
        self.current_target = harf
        ref_sayisi = len(self.references.get(harf, []))
        self.hedef_label.configure(
            text=f"Hedef: {harf}   ({ref_sayisi} referans)"
        )
        self.durum_label.configure(text=f"'{harf}' harfini yazmak için butona basın.")

        # Referans izi göster
        if harf in self.references and self.references[harf]:
            ref = self.references[harf][0]
            self._temiz_grafik()
            try:
                ref_x, ref_y = calculate_2d_trajectory(ref)
                self.ax_ref.plot(ref_x, ref_y, color="#2ecc71", linewidth=2)
                self.ax_ref.set_aspect("equal")
                self.ax_ref.set_title(f"Referans: {harf}", color=COLORS["text"])
            except Exception:
                pass
            self.canvas.draw()

    # ──────────────────────────────────────
    # Kayıt ve Analiz
    # ──────────────────────────────────────

    def _yazmaya_basla(self):
        """Yazmaya başla butonuna tıklanınca."""
        if self.is_recording:
            return

        if not self.references:
            tkinter.messagebox.showwarning(
                "Referans Yok",
                "Henüz hiç referans kaydedilmemiş!\n"
                "'Referans Kaydet' ile önce birkaç harf kaydedin."
            )
            return

        self.is_recording = True
        self.basla_btn.configure(state="disabled", text="⏺ Kayıt Bekleniyor...")
        self.durum_label.configure(text="Butona basarak yazmaya başlayın...")
        self.skor_label.configure(text="...", text_color=COLORS["text_muted"])

        # Ayrı thread'de kayıt
        thread = threading.Thread(target=self._kayit_thread, daemon=True)
        thread.start()

    def _kayit_thread(self):
        """Sensör verisini ayrı thread'de toplar."""
        try:
            user_data = get_sensor_data(is_reference=False)

            if user_data.shape[0] == 0:
                self.after(0, self._kayit_hata, "Veri toplanamadı!")
                return

            # Analiz
            self.after(0, self._analiz_yap, user_data)

        except Exception as e:
            self.after(0, self._kayit_hata, str(e))

    def _analiz_yap(self, user_data: np.ndarray):
        """Toplanan veriyi analiz eder ve sonuçları gösterir."""
        try:
            self.durum_label.configure(text="Analiz ediliyor...")

            # 1. Harf tanımlama
            taninan_harf, guven, mesafeler = identify_letter(user_data, self.references)

            # 2. Hedef harf belirleme
            hedef = self.current_target if self.current_target else taninan_harf

            # 3. Skor hesaplama
            if hedef in self.references and self.references[hedef]:
                ref_data = self.references[hedef][0]

                # DTW skor (geriye uyumlu — 2 değer döner)
                final_score, raw_distance = calculate_score(
                    preprocess_data(ref_data),
                    preprocess_data(user_data),
                )

                # Kalite skoru
                kalite = calculate_quality_score(
                    user_data=user_data,
                    ref_data=ref_data,
                    dtw_distance=raw_distance,
                    identification_confidence=guven,
                )

                # Grafik güncelle
                self._grafik_guncelle(ref_data, user_data, kalite["total_score"])

                # Skor göster
                toplam_skor = kalite["total_score"]
                renk = skor_rengi(toplam_skor)
                self.skor_label.configure(
                    text=f"{toplam_skor:.1f}",
                    text_color=renk,
                )
                self.not_label.configure(
                    text=kalite["grade"],
                    text_color=renk,
                )

                # Tanıma bilgisi
                self.tanima_label.configure(
                    text=f"Tanınan: {taninan_harf} (güven: {guven*100:.0f}%)",
                )

                # Geri bildirim
                self.feedback_text.delete("1.0", "end")
                for msg in kalite["feedback"]:
                    self.feedback_text.insert("end", f"• {msg}\n")

                # Detay skorları
                self._detay_goster(kalite["details"])

                # İlerleme güncelle
                if self.progress and hedef:
                    self.progress.deneme_kaydet(hedef, toplam_skor)

                self.durum_label.configure(
                    text=f"Tamamlandı! Skor: {toplam_skor:.1f}/10 — {kalite['grade']}"
                )
            else:
                self.durum_label.configure(
                    text=f"'{hedef}' için referans bulunamadı."
                )
                self.tanima_label.configure(
                    text=f"Tanınan: {taninan_harf} (güven: {guven*100:.0f}%)",
                )

        except Exception as e:
            self.durum_label.configure(text=f"Analiz hatası: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_recording = False
            self.basla_btn.configure(state="normal", text="🖊️ Yazmaya Başla")

    def _detay_goster(self, detaylar: dict):
        """Detay skorlarını gösterir."""
        for widget in self.detay_frame.winfo_children():
            widget.destroy()

        isimler = {
            "shape_similarity": "Şekil",
            "stroke_accuracy": "Vuruş",
            "smoothness": "Akıcılık",
            "speed_consistency": "Hız",
            "confidence": "Güven",
        }

        for key, baslik in isimler.items():
            skor = detaylar.get(key, 0)
            renk = skor_rengi(skor)

            frame = ctk.CTkFrame(self.detay_frame, fg_color="transparent")
            frame.pack(fill="x", pady=1)

            ctk.CTkLabel(
                frame, text=f"{baslik}:", width=60,
                font=ctk.CTkFont(size=11),
                anchor="w",
            ).pack(side="left", padx=(5, 5))

            bar = ctk.CTkProgressBar(frame, width=120, height=12)
            bar.pack(side="left", padx=5)
            bar.set(skor / 10.0)

            ctk.CTkLabel(
                frame, text=f"{skor:.1f}",
                font=ctk.CTkFont(size=11),
                text_color=renk,
            ).pack(side="left", padx=5)

    def _kayit_hata(self, mesaj: str):
        """Kayıt hatası gösterir."""
        self.durum_label.configure(text=f"Hata: {mesaj}")
        self.is_recording = False
        self.basla_btn.configure(state="normal", text="🖊️ Yazmaya Başla")

    # ──────────────────────────────────────
    # Referans Kaydetme
    # ──────────────────────────────────────

    def _referans_kaydet_dialog(self):
        """Referans kaydetme diyaloğu."""
        harf = tkinter.simpledialog.askstring(
            "Referans Kaydet",
            "Hangi harfi kaydetmek istiyorsunuz?\n(Tek harf girin, ör: A)",
            parent=self,
        )

        if not harf or not harf.strip():
            return

        harf = harf.strip().upper()
        if len(harf) != 1 or not harf.isalpha():
            tkinter.messagebox.showerror("Hata", "Lütfen tek bir harf girin!")
            return

        self.durum_label.configure(text=f"'{harf}' referansı için butona basın...")
        self.basla_btn.configure(state="disabled")

        def kayit_thread():
            try:
                data = get_sensor_data(is_reference=True)
                if data.shape[0] == 0:
                    self.after(0, lambda: self.durum_label.configure(
                        text="Referans kaydedilemedi — veri yok!"
                    ))
                    return

                # Kaydet
                dtw_save_reference(harf, data)

                # Referansları yenile
                self.after(0, self._referanslari_yenile)
                self.after(0, lambda: self.durum_label.configure(
                    text=f"'{harf}' referansı başarıyla kaydedildi!"
                ))

            except Exception as e:
                self.after(0, lambda: self.durum_label.configure(
                    text=f"Hata: {e}"
                ))
            finally:
                self.after(0, lambda: self.basla_btn.configure(state="normal"))

        thread = threading.Thread(target=kayit_thread, daemon=True)
        thread.start()

    def _referanslari_yenile(self):
        """Referans veritabanını yeniden yükler."""
        self.references = load_references()
        self._refresh_letter_buttons()
        self.durum_label.configure(
            text=f"{len(self.references)} harf referansı yüklendi."
        )

    def _kalibrasyon_yap(self):
        """Kalibrasyon başlatır."""
        if sensor_calibrate is None:
            tkinter.messagebox.showinfo("Bilgi", "Kalibrasyon fonksiyonu bulunamadı.")
            return

        self.durum_label.configure(text="Kalibrasyon... Kalemi sabit tutun (3s)...")

        def cal_thread():
            try:
                offsets = sensor_calibrate(duration=3.0)
                self.after(0, lambda: self.durum_label.configure(
                    text=f"Kalibrasyon tamamlandı! Offsets: {offsets}"
                ))
            except Exception as e:
                self.after(0, lambda: self.durum_label.configure(
                    text=f"Kalibrasyon hatası: {e}"
                ))

        thread = threading.Thread(target=cal_thread, daemon=True)
        thread.start()

    # ──────────────────────────────────────
    # Kapanış
    # ──────────────────────────────────────

    def on_closing(self):
        """Program kapanırken temizlik yapar."""
        if self.progress:
            try:
                self.progress.kaydet()
            except Exception:
                pass

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

        plt.close("all")
        self.destroy()
        sys.exit(0)


# ── Ana Giriş Noktası ──
if __name__ == "__main__":
    os.makedirs("letters_db", exist_ok=True)
    app = HandwritingApp()
    app.mainloop()