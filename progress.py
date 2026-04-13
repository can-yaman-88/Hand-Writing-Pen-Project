"""
progress.py — Kullanıcı İlerleme ve Öğrenme Yönetim Sistemi

Türk alfabesi harflerini zorluk seviyelerine göre sıralayarak
kullanıcının öğrenme sürecini yönetir. Harf açma, kelime modu,
istatistik takibi ve profil kaydetme/yükleme işlemlerini içerir.

Ana analiz DTW ile yapılır (dtw_analysis.py [1]) ve skorlama
scoring.py tarafından hesaplanır. Bu modül, bu skorları kullanarak
ilerleme mantığını yönetir.

Bağımlılıklar:
    - json, datetime, os (standart kütüphane)
"""

import json
import os
from datetime import datetime, date
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Set,
)


# ─────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────

# Zorluk seviyeleri ve harf grupları
SEVIYELER: Dict[int, List[str]] = {
    1: ["O", "I", "C", "L"],
    2: ["U", "T", "V", "J"],
    3: ["A", "E", "M", "N", "K"],
    4: ["B", "D", "H", "P", "R"],
    5: ["F", "G", "S", "Y", "İ"],
    6: ["Ç", "Ğ", "Ö", "Ş", "Ü", "Z"],
}

# Tüm harfler (düz liste)
TUM_HARFLER: List[str] = []
for _seviye in sorted(SEVIYELER.keys()):
    TUM_HARFLER.extend(SEVIYELER[_seviye])

TOPLAM_HARF_SAYISI: int = len(TUM_HARFLER)

# Harf → seviye eşlemesi (hızlı erişim)
HARF_SEVIYE: Dict[str, int] = {}
for _sev, _harfler in SEVIYELER.items():
    for _h in _harfler:
        HARF_SEVIYE[_h] = _sev

# Başarı eşiği
BASARI_ESIGI: float = 7.0

# Ard arda kaç başarılı deneme gerekli
ARDARDA_BASARI_GEREKSIZ: int = 3

# Kelime modu açılma eşiği (minimum açılmış harf sayısı)
KELIME_MODU_ESIGI: int = 8

# Varsayılan profil dosyası
VARSAYILAN_PROFIL_DOSYASI: str = "user_progress.json"

# ─────────────────────────────────────────────────────────────────────
# Kelime Havuzu
# ─────────────────────────────────────────────────────────────────────

# Türkçe kelimeler — zorluk sırasına göre (kısa → uzun)
# Tüm harfler büyük harf olarak tutulur
KELIME_HAVUZU: Dict[int, List[str]] = {
    2: [
        "AL", "EL", "OL", "AT", "İL", "İN", "UN", "AV",
        "OT", "UZ", "OK", "AK", "EK", "İP", "OP", "EV",
        "İZ", "ÖN", "ÜÇ", "AŞ", "ÖZ", "US", "AÇ", "GÖ",
    ],
    3: [
        "ACI", "ALI", "OCA", "ULU", "ANA", "İLK", "ALT",
        "SON", "BİR", "GEL", "GİT", "VER", "BAK", "TAŞ",
        "KOL", "GÖL", "GÜL", "SOL", "TUZ", "KUŞ", "BUZ",
        "CAN", "DAL", "DİL", "TUĞ", "MİL", "PUL", "NEM",
        "FİL", "HAL", "SAÇ", "YOL", "YAZ", "KIŞ",
    ],
    4: [
        "KALE", "TAVA", "OKUL", "MASA", "GECE", "KONU",
        "DOLU", "SALI", "OTEL", "VALI", "KOVA", "İLAÇ",
        "TREN", "KAPI", "SPOR", "HAVA", "FAAL", "PLAN",
        "KRAL", "MAVI", "NOTA", "BÜFE", "GÜÇE", "ŞANS",
        "GAZI", "PUAN", "DÖRT", "SERT", "KURT", "BÖCE",
        "GENÇ", "HALK", "ÖDEV", "TÜRK", "YÜCE",
    ],
    5: [
        "ARABA", "ELMAS", "KALEM", "KITAP", "BULUT",
        "ÇOBAN", "DÜNYA", "ERKEK", "FOSIL", "GÜNEŞ",
        "HAFTA", "ILICA", "JAPON", "KALIP", "LIMON",
        "MASAL", "NEHIR", "ORMAN", "PRENS", "RÜZGA",
        "SINIF", "TAVUK", "ÜLKER", "VAKIT", "YÜZME",
        "ZAMAN", "BAHÇE", "ÇİÇEK", "KÖPEK", "ŞEKER",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Oturum Bilgisi
# ─────────────────────────────────────────────────────────────────────

class Oturum:
    """Mevcut oturumun anlık istatistiklerini tutar.

    Attributes:
        baslangic: Oturum başlangıç zamanı.
        denemeler: Bu oturumdaki deneme sayısı.
        skorlar: Bu oturumdaki tüm skorlar.
    """

    def __init__(self) -> None:
        self.baslangic: str = datetime.now().isoformat()
        self.denemeler: int = 0
        self.skorlar: List[float] = []

    def deneme_ekle(self, skor: float) -> None:
        """Oturuma yeni deneme ekler."""
        self.denemeler += 1
        self.skorlar.append(skor)

    def ortalama_skor(self) -> float:
        """Oturum ortalama skorunu döndürür."""
        if not self.skorlar:
            return 0.0
        return round(sum(self.skorlar) / len(self.skorlar), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Oturumu sözlük olarak döndürür (JSON kayıt için)."""
        return {
            "date": self.baslangic,
            "attempts": self.denemeler,
            "avg_score": self.ortalama_skor(),
        }


# ─────────────────────────────────────────────────────────────────────
# Ana İlerleme Sınıfı
# ─────────────────────────────────────────────────────────────────────

class ProgressManager:
    """Kullanıcı ilerleme ve öğrenme yönetim sistemi.

    Türk alfabesi harflerini zorluk seviyelerine göre sıralayarak
    kullanıcının öğrenme sürecini yönetir. Skor verileri
    dtw_analysis.py [1] ve scoring.py üzerinden hesaplanır;
    bu sınıf skorları alıp ilerleme mantığını işletir.

    Temel özellikler:
        - 6 seviyeli harf açma sistemi
        - Ard arda 3 başarılı deneme ile harf kilit açma
        - Kelime modu (8+ harf açılınca)
        - Detaylı istatistikler
        - JSON profil kaydetme/yükleme

    Attributes:
        username: Kullanıcı adı.
        profil_dosyasi: JSON profil dosya yolu.
        unlocked_letters: Açılmış harfler kümesi.
        current_level: Mevcut seviye (1-6).
        letter_scores: Harf → skor geçmişi.
        letter_streaks: Harf → ard arda başarılı deneme sayısı.
        mode: Mevcut mod ("letter" veya "word").
        total_attempts: Toplam deneme sayısı.
        sessions: Geçmiş oturum kayıtları.
        current_session: Mevcut aktif oturum.
        streak: Genel ard arda başarılı deneme sayısı.
    """

    def __init__(
        self,
        username: str = "Kullanıcı",
        profil_dosyasi: str = VARSAYILAN_PROFIL_DOSYASI,
        otomatik_yukle: bool = True,
    ) -> None:
        """İlerleme yöneticisini başlatır.

        Args:
            username: Kullanıcı adı.
            profil_dosyasi: JSON profil dosya yolu.
            otomatik_yukle: True ise dosyadan otomatik yükle.
        """
        self.username: str = username
        self.profil_dosyasi: str = profil_dosyasi

        # ── İlerleme durumu ──
        self.unlocked_letters: Set[str] = set(SEVIYELER[1])  # Seviye 1 açık başlar
        self.current_level: int = 1
        self.letter_scores: Dict[str, List[float]] = {}
        self.letter_streaks: Dict[str, int] = {}  # Harf → ard arda başarılı
        self.mode: str = "letter"
        self.total_attempts: int = 0
        self.sessions: List[Dict[str, Any]] = []
        self.streak: int = 0  # Genel ard arda başarı serisi

        # ── Mevcut oturum ──
        self.current_session: Oturum = Oturum()

        # ── Kelime geçmişi ──
        self.word_scores: Dict[str, List[float]] = {}
        self.completed_words: Set[str] = set()

        # ── Otomatik yükleme ──
        if otomatik_yukle:
            self.yukle()

    # ─────────────────────────────────────────────────────────────────
    # Harf Kilidi Yönetimi
    # ─────────────────────────────────────────────────────────────────

    def harf_acik_mi(self, harf: str) -> bool:
        """Harfin açılmış olup olmadığını kontrol eder.

        Args:
            harf: Kontrol edilecek harf (büyük harf).

        Returns:
            True ise harf açık.
        """
        return harf.upper() in self.unlocked_letters

    def seviye_tamam_mi(self, seviye: int) -> bool:
        """Belirtilen seviyedeki tüm harflerin açılıp açılmadığını kontrol eder.

        Args:
            seviye: Seviye numarası (1-6).

        Returns:
            True ise seviyedeki tüm harfler açık.
        """
        if seviye not in SEVIYELER:
            return False
        return all(h in self.unlocked_letters for h in SEVIYELER[seviye])

    def _harf_ac(self, harf: str) -> bool:
        """Bir harfi açar (kilidini kaldırır).

        Args:
            harf: Açılacak harf.

        Returns:
            True ise yeni açıldı, False ise zaten açıktı.
        """
        h = harf.upper()
        if h in self.unlocked_letters:
            return False
        self.unlocked_letters.add(h)
        return True

    def _seviye_guncelle(self) -> None:
        """Mevcut seviyeyi kontrol edip gerekirse yükseltir.

        Bir seviyedeki tüm harfler açılınca sonraki seviye
        harfleri erişilebilir hale gelir.
        """
        for seviye in sorted(SEVIYELER.keys()):
            if self.seviye_tamam_mi(seviye):
                yeni_seviye = seviye + 1
                if yeni_seviye in SEVIYELER and yeni_seviye > self.current_level:
                    self.current_level = yeni_seviye
                    # Yeni seviyenin harflerini henüz açmıyoruz —
                    # ard arda başarı ile açılacaklar.
                    # Ancak erişilebilir (denenebilir) hale gelirler.

    def erisilebilir_harfler(self) -> List[str]:
        """Şu an denenebilir (seviyesi açılmış) harfleri döndürür.

        Kullanıcının mevcut seviyesi ve altındaki tüm harfler
        denenebilir durumdadır.

        Returns:
            Erişilebilir harflerin listesi.
        """
        harfler: List[str] = []
        for seviye in sorted(SEVIYELER.keys()):
            if seviye <= self.current_level:
                harfler.extend(SEVIYELER[seviye])
        return harfler

    def kilitli_harfler(self) -> List[str]:
        """Henüz açılmamış ama erişilebilir harfleri döndürür.

        Returns:
            Kilitli harflerin listesi.
        """
        erisim = self.erisilebilir_harfler()
        return [h for h in erisim if h not in self.unlocked_letters]

    # ─────────────────────────────────────────────────────────────────
    # Deneme Kayıt
    # ─────────────────────────────────────────────────────────────────

    def deneme_kaydet(
        self,
        harf: str,
        skor: float,
        otomatik_kaydet: bool = True,
    ) -> Dict[str, Any]:
        """Bir harf denemesinin sonucunu kaydeder ve ilerlemeyi günceller.

        İşlem sırası:
            1. Skoru kaydet
            2. Başarılı mı kontrol et (>= 7.0)
            3. Ard arda sayacı güncelle
            4. Gerekirse harf aç
            5. Gerekirse seviye yükselt
            6. Profili kaydet

        Skor, dtw_analysis.py [1] üzerinden hesaplanan DTW mesafesi
        ve scoring.py'deki çok boyutlu değerlendirme sonucudur.

        Args:
            harf: Denenen harf (büyük harf).
            skor: 0-10 arası kalite skoru.
            otomatik_kaydet: True ise dosyaya otomatik kaydet.

        Returns:
            Sonuç sözlüğü:
            {
                "harf": str,
                "skor": float,
                "basarili": bool,
                "streak": int,
                "yeni_acilan": List[str],
                "seviye_atladi": bool,
                "mesaj": str,
            }
        """
        h = harf.upper()
        skor = round(max(0.0, min(10.0, skor)), 2)

        # ── Skor geçmişine ekle ──
        if h not in self.letter_scores:
            self.letter_scores[h] = []
        self.letter_scores[h].append(skor)

        # ── Oturum ve genel sayaçlar ──
        self.total_attempts += 1
        self.current_session.deneme_ekle(skor)

        basarili = skor >= BASARI_ESIGI

        # ── Genel streak ──
        if basarili:
            self.streak += 1
        else:
            self.streak = 0

        # ── Harf bazlı streak ──
        if h not in self.letter_streaks:
            self.letter_streaks[h] = 0

        if basarili:
            self.letter_streaks[h] += 1
        else:
            self.letter_streaks[h] = 0

        # ── Harf açma kontrolü ──
        yeni_acilan: List[str] = []
        eski_seviye = self.current_level

        if self.letter_streaks[h] >= ARDARDA_BASARI_GEREKSIZ:
            if self._harf_ac(h):
                yeni_acilan.append(h)

        # ── Seviye güncelleme ──
        self._seviye_guncelle()
        seviye_atladi = self.current_level > eski_seviye

        # Yeni seviyedeki harflerin ilk erişimini sağla
        if seviye_atladi:
            pass  # Harfler erişilebilir ama kilitli — streak ile açılacak

        # ── Mod güncelleme ──
        if len(self.unlocked_letters) >= KELIME_MODU_ESIGI:
            # Kelime modu artık kullanılabilir (mod değiştirmek opsiyonel)
            pass

        # ── Mesaj üret ──
        mesaj = self._deneme_mesaji(h, skor, basarili, yeni_acilan, seviye_atladi)

        # ── Otomatik kaydet ──
        if otomatik_kaydet:
            self.kaydet()

        return {
            "harf": h,
            "skor": skor,
            "basarili": basarili,
            "streak": self.streak,
            "harf_streak": self.letter_streaks.get(h, 0),
            "yeni_acilan": yeni_acilan,
            "seviye_atladi": seviye_atladi,
            "yeni_seviye": self.current_level,
            "mesaj": mesaj,
        }

    def kelime_deneme_kaydet(
        self,
        kelime: str,
        skor: float,
        otomatik_kaydet: bool = True,
    ) -> Dict[str, Any]:
        """Kelime denemesinin sonucunu kaydeder.

        Args:
            kelime: Denenen kelime (büyük harf).
            skor: 0-10 arası ortalama kalite skoru.
            otomatik_kaydet: Dosyaya otomatik kaydet.

        Returns:
            Sonuç sözlüğü.
        """
        k = kelime.upper()
        skor = round(max(0.0, min(10.0, skor)), 2)

        if k not in self.word_scores:
            self.word_scores[k] = []
        self.word_scores[k].append(skor)

        self.total_attempts += 1
        self.current_session.deneme_ekle(skor)

        basarili = skor >= BASARI_ESIGI
        if basarili:
            self.streak += 1
            self.completed_words.add(k)
        else:
            self.streak = 0

        if otomatik_kaydet:
            self.kaydet()

        return {
            "kelime": k,
            "skor": skor,
            "basarili": basarili,
            "streak": self.streak,
            "mesaj": self._kelime_mesaji(k, skor, basarili),
        }

    def _deneme_mesaji(
        self,
        harf: str,
        skor: float,
        basarili: bool,
        yeni_acilan: List[str],
        seviye_atladi: bool,
    ) -> str:
        """Deneme sonucuna göre Türkçe mesaj üretir."""
        parcalar: List[str] = []

        # Skor geri bildirimi
        if skor >= 9.0:
            parcalar.append(f"🎉 Muhteşem! '{harf}' harfinde {skor:.1f} puan!")
        elif skor >= 7.0:
            parcalar.append(f"👏 Güzel! '{harf}' harfinde {skor:.1f} puan.")
        elif skor >= 5.0:
            parcalar.append(f"💪 İyi gidiyorsun! '{harf}' harfinde {skor:.1f} puan.")
        elif skor >= 3.0:
            parcalar.append(f"📝 '{harf}' harfinde {skor:.1f} puan. Pratik yapmaya devam!")
        else:
            parcalar.append(
                f"🔄 '{harf}' harfinde {skor:.1f} puan. Referansı incele ve tekrar dene."
            )

        # Streak bilgisi
        harf_str = self.letter_streaks.get(harf, 0)
        if basarili and harf_str < ARDARDA_BASARI_GEREKSIZ:
            kalan = ARDARDA_BASARI_GEREKSIZ - harf_str
            parcalar.append(
                f"  → Kilidi açmak için {kalan} başarılı deneme daha gerekli."
            )

        # Yeni açılan harfler
        for yh in yeni_acilan:
            parcalar.append(f"  🔓 '{yh}' harfinin kilidi açıldı!")

        # Seviye atlama
        if seviye_atladi:
            parcalar.append(
                f"  🏆 TEBRİKLER! Seviye {self.current_level}'e yükseldin!"
            )
            yeni_harfler = SEVIYELER.get(self.current_level, [])
            if yeni_harfler:
                parcalar.append(
                    f"  📖 Yeni harfler: {', '.join(yeni_harfler)}"
                )

        # Genel streak
        if self.streak >= 5:
            parcalar.append(f"  🔥 {self.streak} ard arda başarı!")

        # Kelime modu bildirimi
        if len(self.unlocked_letters) == KELIME_MODU_ESIGI:
            parcalar.append(
                "  📝 Kelime modu açıldı! Artık kelime yazma pratiği yapabilirsin."
            )

        return "\n".join(parcalar)

    def _kelime_mesaji(self, kelime: str, skor: float, basarili: bool) -> str:
        """Kelime denemesi için mesaj üretir."""
        if basarili:
            return f"✅ '{kelime}' kelimesini başarıyla yazdın! Skor: {skor:.1f}"
        else:
            return f"📝 '{kelime}' kelimesinde {skor:.1f} puan. Tekrar dene!"

    # ─────────────────────────────────────────────────────────────────
    # Kelime Modu
    # ─────────────────────────────────────────────────────────────────

    def kelime_modu_acik_mi(self) -> bool:
        """Kelime modunun kullanılabilir olup olmadığını kontrol eder.

        Returns:
            True ise 8+ harf açılmış ve kelime modu kullanılabilir.
        """
        return len(self.unlocked_letters) >= KELIME_MODU_ESIGI

    def kullanilabilir_kelimeler(
        self,
        max_uzunluk: Optional[int] = None,
    ) -> List[str]:
        """Açılmış harflerden oluşturulabilecek kelimeleri döndürür.

        Sadece tüm harfleri açılmış olan kelimeler listelenir.

        Args:
            max_uzunluk: Maksimum kelime uzunluğu. None ise sınırsız.

        Returns:
            Uygun kelimelerin listesi (kısa → uzun sıralı).
        """
        if not self.kelime_modu_acik_mi():
            return []

        acik = self.unlocked_letters
        uygun: List[str] = []

        for uzunluk in sorted(KELIME_HAVUZU.keys()):
            if max_uzunluk is not None and uzunluk > max_uzunluk:
                break
            for kelime in KELIME_HAVUZU[uzunluk]:
                # Kelimenin her harfi açılmış mı?
                if all(c in acik for c in kelime):
                    uygun.append(kelime)

        return uygun

    def tamamlanmamis_kelimeler(self) -> List[str]:
        """Henüz başarılı tamamlanmamış kullanılabilir kelimeleri döndürür.

        Returns:
            Tamamlanmamış kelimelerin listesi.
        """
        tum = self.kullanilabilir_kelimeler()
        return [k for k in tum if k not in self.completed_words]

    # ─────────────────────────────────────────────────────────────────
    # İstatistikler
    # ─────────────────────────────────────────────────────────────────

    def harf_ortalama_skor(self, harf: str) -> float:
        """Bir harfin ortalama skorunu döndürür.

        Args:
            harf: Harf karakteri.

        Returns:
            Ortalama skor. Deneme yoksa 0.0.
        """
        h = harf.upper()
        skorlar = self.letter_scores.get(h, [])
        if not skorlar:
            return 0.0
        return round(sum(skorlar) / len(skorlar), 2)

    def harf_deneme_sayisi(self, harf: str) -> int:
        """Bir harf için toplam deneme sayısını döndürür."""
        return len(self.letter_scores.get(harf.upper(), []))

    def harf_en_iyi_skor(self, harf: str) -> float:
        """Bir harfin en iyi skorunu döndürür."""
        skorlar = self.letter_scores.get(harf.upper(), [])
        return round(max(skorlar), 2) if skorlar else 0.0

    def harf_son_skor(self, harf: str) -> float:
        """Bir harfin en son skorunu döndürür."""
        skorlar = self.letter_scores.get(harf.upper(), [])
        return round(skorlar[-1], 2) if skorlar else 0.0

    def en_zayif_harfler(self, n: int = 5) -> List[Tuple[str, float]]:
        """En düşük ortalama skorlu harfleri döndürür.

        Sadece en az 1 denemesi olan harfler dahil edilir.

        Args:
            n: Döndürülecek harf sayısı.

        Returns:
            [(harf, ortalama_skor), ...] listesi, düşükten yükseğe.
        """
        ortalamalar: List[Tuple[str, float]] = []
        for harf, skorlar in self.letter_scores.items():
            if skorlar:
                ort = sum(skorlar) / len(skorlar)
                ortalamalar.append((harf, round(ort, 2)))

        ortalamalar.sort(key=lambda x: x[1])
        return ortalamalar[:n]

    def en_guclu_harfler(self, n: int = 5) -> List[Tuple[str, float]]:
        """En yüksek ortalama skorlu harfleri döndürür.

        Args:
            n: Döndürülecek harf sayısı.

        Returns:
            [(harf, ortalama_skor), ...] listesi, yüksekten düşüğe.
        """
        ortalamalar: List[Tuple[str, float]] = []
        for harf, skorlar in self.letter_scores.items():
            if skorlar:
                ort = sum(skorlar) / len(skorlar)
                ortalamalar.append((harf, round(ort, 2)))

        ortalamalar.sort(key=lambda x: x[1], reverse=True)
        return ortalamalar[:n]

    def toplam_ilerleme(self) -> float:
        """Açılan harf / toplam harf oranını yüzde olarak döndürür.

        Returns:
            İlerleme yüzdesi (0.0 – 100.0).
        """
        return round(
            (len(self.unlocked_letters) / TOPLAM_HARF_SAYISI) * 100.0, 1
        )

    def istatistikler(self) -> Dict[str, Any]:
        """Tüm istatistikleri tek sözlükte döndürür.

        Returns:
            Kapsamlı istatistik sözlüğü.
        """
        # Tüm skorlar
        tum_skorlar: List[float] = []
        for skorlar in self.letter_scores.values():
            tum_skorlar.extend(skorlar)

        genel_ortalama = (
            round(sum(tum_skorlar) / len(tum_skorlar), 2) if tum_skorlar else 0.0
        )

        return {
            "kullanici": self.username,
            "mevcut_seviye": self.current_level,
            "toplam_ilerleme_yuzde": self.toplam_ilerleme(),
            "acilan_harf_sayisi": len(self.unlocked_letters),
            "toplam_harf_sayisi": TOPLAM_HARF_SAYISI,
            "acilan_harfler": sorted(self.unlocked_letters),
            "toplam_deneme": self.total_attempts,
            "genel_ortalama_skor": genel_ortalama,
            "en_zayif_5": self.en_zayif_harfler(5),
            "en_guclu_5": self.en_guclu_harfler(5),
            "ard_arda_basari": self.streak,
            "kelime_modu_acik": self.kelime_modu_acik_mi(),
            "tamamlanan_kelime_sayisi": len(self.completed_words),
            "oturum": {
                "denemeler": self.current_session.denemeler,
                "ortalama_skor": self.current_session.ortalama_skor(),
            },
            "harf_detay": {
                h: {
                    "deneme_sayisi": len(s),
                    "ortalama": round(sum(s) / len(s), 2) if s else 0,
                    "en_iyi": round(max(s), 2) if s else 0,
                    "son_skor": round(s[-1], 2) if s else 0,
                    "streak": self.letter_streaks.get(h, 0),
                    "acik": h in self.unlocked_letters,
                }
                for h, s in self.letter_scores.items()
            },
        }

    # ─────────────────────────────────────────────────────────────────
    # Sonraki Görev Belirleme
    # ─────────────────────────────────────────────────────────────────

    def get_next_task(self) -> Dict[str, Any]:
        """Mevcut duruma göre kullanıcıya sonraki görevi önerir.

        Öncelik sırası:
            1. Kilitli harfler varsa → en az streak'e sahip kilitli harf
            2. Zayıf harfler varsa → en düşük ortalamalı açık harf
            3. Kelime modu açıksa → tamamlanmamış kelime
            4. Tüm harfler güçlüyse → rastgele tekrar

        Bu görev önerisi, dtw_analysis.py [1] üzerinden yapılan
        DTW analizinin sonuçlarına dayanır.

        Returns:
            Görev sözlüğü:
            {
                "type": "letter" | "word",
                "target": str,
                "level": int (harf için),
                "reason": str,
            }
        """
        # ── Öncelik 1: Kilitli harfler (mevcut seviyede) ──
        kilitli = self.kilitli_harfler()
        if kilitli:
            # En yakın olanı öner (en çok streak birikmiş)
            en_yakin = max(
                kilitli,
                key=lambda h: self.letter_streaks.get(h, 0),
            )
            # Eğer hiçbirinin streak'i yoksa seviye sırasına göre
            if self.letter_streaks.get(en_yakin, 0) == 0:
                en_yakin = kilitli[0]

            return {
                "type": "letter",
                "target": en_yakin,
                "level": HARF_SEVIYE.get(en_yakin, self.current_level),
                "reason": f"'{en_yakin}' harfinin kilidini açmak için pratik yap.",
            }

        # ── Öncelik 2: Zayıf harfler (ortalaması < 7 olan açık harfler) ──
        zayif_harfler: List[Tuple[str, float]] = []
        for harf in self.unlocked_letters:
            ort = self.harf_ortalama_skor(harf)
            deneme = self.harf_deneme_sayisi(harf)
            if deneme > 0 and ort < BASARI_ESIGI:
                zayif_harfler.append((harf, ort))

        if zayif_harfler:
            zayif_harfler.sort(key=lambda x: x[1])
            hedef = zayif_harfler[0][0]
            return {
                "type": "letter",
                "target": hedef,
                "level": HARF_SEVIYE.get(hedef, 1),
                "reason": (
                    f"'{hedef}' harfini güçlendir "
                    f"(ortalama: {zayif_harfler[0][1]:.1f})."
                ),
            }

        # ── Öncelik 3: Hiç denenmemiş açık harfler ──
        denenmemis = [
            h for h in self.unlocked_letters
            if h not in self.letter_scores or not self.letter_scores[h]
        ]
        if denenmemis:
            hedef = denenmemis[0]
            return {
                "type": "letter",
                "target": hedef,
                "level": HARF_SEVIYE.get(hedef, 1),
                "reason": f"'{hedef}' harfini henüz denemedin. Bir dene!",
            }

        # ── Öncelik 4: Kelime modu ──
        if self.kelime_modu_acik_mi():
            tamamlanmamis = self.tamamlanmamis_kelimeler()
            if tamamlanmamis:
                # Kısa kelimelerden başla
                hedef_kelime = tamamlanmamis[0]
                return {
                    "type": "word",
                    "target": hedef_kelime,
                    "level": len(hedef_kelime),
                    "reason": f"'{hedef_kelime}' kelimesini yazmayı dene!",
                }

        # ── Öncelik 5: Tekrar — en düşük son skorlu harf ──
        tum_acik = list(self.unlocked_letters)
        if tum_acik:
            # En düşük son skora sahip harfi bul
            tum_acik.sort(
                key=lambda h: self.harf_son_skor(h) if self.harf_deneme_sayisi(h) > 0 else 999
            )
            hedef = tum_acik[0]
            return {
                "type": "letter",
                "target": hedef,
                "level": HARF_SEVIYE.get(hedef, 1),
                "reason": f"'{hedef}' harfini tekrar ederek ustalaş!",
            }

        # Fallback
        return {
            "type": "letter",
            "target": "O",
            "level": 1,
            "reason": "'O' harfiyle başla!",
        }

    # ─────────────────────────────────────────────────────────────────
    # Mod Değiştirme
    # ─────────────────────────────────────────────────────────────────

    def mod_degistir(self, yeni_mod: str) -> bool:
        """Çalışma modunu değiştirir.

        Args:
            yeni_mod: "letter" veya "word".

        Returns:
            True ise mod değiştirildi, False ise değiştirilemedi.
        """
        yeni_mod = yeni_mod.lower()

        if yeni_mod == "word" and not self.kelime_modu_acik_mi():
            return False

        if yeni_mod in ("letter", "word"):
            self.mode = yeni_mod
            return True

        return False

    # ─────────────────────────────────────────────────────────────────
    # Kaydetme / Yükleme (JSON)
    # ─────────────────────────────────────────────────────────────────

    def kaydet(self, dosya_yolu: Optional[str] = None) -> bool:
        """Profili JSON dosyasına kaydeder.

        Her deneme sonrası otomatik çağrılır.

        Args:
            dosya_yolu: Hedef dosya yolu. None ise varsayılan kullanılır.

        Returns:
            True ise başarılı.
        """
        yol = dosya_yolu or self.profil_dosyasi

        profil: Dict[str, Any] = {
            "username": self.username,
            "unlocked_letters": sorted(self.unlocked_letters),
            "current_level": self.current_level,
            "letter_scores": self.letter_scores,
            "letter_streaks": self.letter_streaks,
            "mode": self.mode,
            "total_attempts": self.total_attempts,
            "streak": self.streak,
            "word_scores": self.word_scores,
            "completed_words": sorted(self.completed_words),
            "sessions": self.sessions + [self.current_session.to_dict()],
            "last_saved": datetime.now().isoformat(),
        }

        try:
            # Dizin yoksa oluştur
            dizin = os.path.dirname(yol)
            if dizin:
                os.makedirs(dizin, exist_ok=True)

            with open(yol, "w", encoding="utf-8") as f:
                json.dump(profil, f, ensure_ascii=False, indent=2)
            return True

        except (OSError, IOError) as e:
            print(f"⚠ Profil kaydedilemedi: {e}")
            return False

    def yukle(self, dosya_yolu: Optional[str] = None) -> bool:
        """Profili JSON dosyasından yükler.

        Program açıldığında otomatik çağrılır.

        Args:
            dosya_yolu: Kaynak dosya yolu. None ise varsayılan kullanılır.

        Returns:
            True ise başarılı yüklendi, False ise dosya bulunamadı
            veya hatalı.
        """
        yol = dosya_yolu or self.profil_dosyasi

        if not os.path.exists(yol):
            return False

        try:
            with open(yol, "r", encoding="utf-8") as f:
                profil = json.load(f)

            # ── Alanları yükle (geriye uyumlu — eksik alanlar için varsayılan) ──
            self.username = profil.get("username", self.username)

            yuklu_harfler = profil.get("unlocked_letters", [])
            # Seviye 1 her zaman açık
            self.unlocked_letters = set(yuklu_harfler) | set(SEVIYELER[1])

            self.current_level = profil.get("current_level", 1)
            self.letter_scores = profil.get("letter_scores", {})
            self.letter_streaks = profil.get("letter_streaks", {})
            self.mode = profil.get("mode", "letter")
            self.total_attempts = profil.get("total_attempts", 0)
            self.streak = profil.get("streak", 0)

            self.word_scores = profil.get("word_scores", {})
            self.completed_words = set(profil.get("completed_words", []))

            self.sessions = profil.get("sessions", [])

            # Yeni oturum başlat
            self.current_session = Oturum()

            return True

        except (json.JSONDecodeError, OSError, IOError, KeyError) as e:
            print(f"⚠ Profil yüklenemedi: {e}")
            return False

    def profil_sifirla(self, otomatik_kaydet: bool = True) -> None:
        """Profili tamamen sıfırlar (yeni kullanıcı gibi).

        Args:
            otomatik_kaydet: True ise sıfırlamayı dosyaya kaydet.
        """
        self.unlocked_letters = set(SEVIYELER[1])
        self.current_level = 1
        self.letter_scores = {}
        self.letter_streaks = {}
        self.mode = "letter"
        self.total_attempts = 0
        self.streak = 0
        self.word_scores = {}
        self.completed_words = set()
        self.sessions = []
        self.current_session = Oturum()

        if otomatik_kaydet:
            self.kaydet()

    # ─────────────────────────────────────────────────────────────────
    # Seviye Bilgisi
    # ─────────────────────────────────────────────────────────────────

    def seviye_bilgisi(self, seviye: Optional[int] = None) -> Dict[str, Any]:
        """Belirtilen seviyenin detaylı bilgisini döndürür.

        Args:
            seviye: Seviye numarası. None ise mevcut seviye.

        Returns:
            Seviye bilgi sözlüğü.
        """
        sev = seviye if seviye is not None else self.current_level

        if sev not in SEVIYELER:
            return {"hata": f"Seviye {sev} bulunamadı."}

        harfler = SEVIYELER[sev]
        acik = [h for h in harfler if h in self.unlocked_letters]
        kilitli = [h for h in harfler if h not in self.unlocked_letters]
        tamam = len(acik) == len(harfler)

        return {
            "seviye": sev,
            "harfler": harfler,
            "acik": acik,
            "kilitli": kilitli,
            "ilerleme": f"{len(acik)}/{len(harfler)}",
            "tamamlandi": tamam,
            "erisebilir": sev <= self.current_level,
        }

    # ─────────────────────────────────────────────────────────────────
    # Özet Rapor
    # ─────────────────────────────────────────────────────────────────

    def ozet_rapor(self) -> str:
        """Kullanıcı ilerlemesinin Türkçe metin özetini döndürür.

        Returns:
            Çok satırlı özet metni.
        """
        ist = self.istatistikler()
        satirlar: List[str] = []

        satirlar.append("=" * 55)
        satirlar.append(f"  📊 İLERLEME RAPORU — {self.username}")
        satirlar.append("=" * 55)

        satirlar.append(f"  Seviye        : {ist['mevcut_seviye']} / 6")
        satirlar.append(f"  İlerleme      : %{ist['toplam_ilerleme_yuzde']}")
        satirlar.append(
            f"  Açılan harfler: {ist['acilan_harf_sayisi']} / {ist['toplam_harf_sayisi']}"
        )
        satirlar.append(f"  Toplam deneme : {ist['toplam_deneme']}")
        satirlar.append(f"  Genel ortalama: {ist['genel_ortalama_skor']:.1f} / 10")
        satirlar.append(f"  Ard arda başarı: {ist['ard_arda_basari']}")

        satirlar.append("-" * 55)
        satirlar.append("  SEVİYE DURUMU:")
        for sev in sorted(SEVIYELER.keys()):
            bilgi = self.seviye_bilgisi(sev)
            durum = "✅" if bilgi["tamamlandi"] else ("🔓" if bilgi["erisebilir"] else "🔒")
            satirlar.append(
                f"    {durum} Seviye {sev}: {bilgi['ilerleme']} "
                f"({', '.join(bilgi['harfler'])})"
            )

        if ist["en_guclu_5"]:
            satirlar.append("-" * 55)
            satirlar.append("  💪 EN GÜÇLÜ HARFLER:")
            for harf, ort in ist["en_guclu_5"]:
                satirlar.append(f"    {harf}: {ort:.1f}")

        if ist["en_zayif_5"]:
            satirlar.append("  📝 GELİŞTİRİLECEK HARFLER:")
            for harf, ort in ist["en_zayif_5"]:
                satirlar.append(f"    {harf}: {ort:.1f}")

        if ist["kelime_modu_acik"]:
            satirlar.append("-" * 55)
            satirlar.append(
                f"  📖 Kelime modu: AKTİF "
                f"({ist['tamamlanan_kelime_sayisi']} kelime tamamlandı)"
            )

        satirlar.append("-" * 55)
        satirlar.append(
            f"  Bu oturum: {ist['oturum']['denemeler']} deneme, "
            f"ortalama {ist['oturum']['ortalama_skor']:.1f}"
        )

        # Sonraki görev
        gorev = self.get_next_task()
        satirlar.append("-" * 55)
        satirlar.append(f"  🎯 ÖNERİLEN GÖREV: {gorev['reason']}")

        satirlar.append("=" * 55)

        return "\n".join(satirlar)

    # ─────────────────────────────────────────────────────────────────
    # String Temsil
    # ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ProgressManager("
            f"user={self.username!r}, "
            f"level={self.current_level}, "
            f"unlocked={len(self.unlocked_letters)}/{TOPLAM_HARF_SAYISI}, "
            f"attempts={self.total_attempts})"
        )


# ─────────────────────────────────────────────────────────────────────
# Modül Seviyesinde Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────

def harf_seviyesi(harf: str) -> int:
    """Bir harfin zorluk seviyesini döndürür.

    Args:
        harf: Harf karakteri.

    Returns:
        Seviye numarası (1-6). Bulunamazsa -1.
    """
    return HARF_SEVIYE.get(harf.upper(), -1)


def seviye_harfleri(seviye: int) -> List[str]:
    """Belirtilen seviyedeki harfleri döndürür.

    Args:
        seviye: Seviye numarası (1-6).

    Returns:
        Harf listesi. Geçersiz seviyede boş liste.
    """
    return SEVIYELER.get(seviye, [])


# ─────────────────────────────────────────────────────────────────────
# Modül Testi
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  PROGRESS.PY — Modül Testi")
    print("=" * 60)

    # Test profil dosyası
    TEST_DOSYA = "test_progress.json"

    # Varsa önceki test dosyasını sil
    if os.path.exists(TEST_DOSYA):
        os.remove(TEST_DOSYA)

    # ── Yeni profil oluştur ──
    pm = ProgressManager(
        username="Test Kullanıcı",
        profil_dosyasi=TEST_DOSYA,
        otomatik_yukle=False,
    )
    print(f"\n  {pm}")
    print(f"  Seviye 1 harfleri (açık): {sorted(pm.unlocked_letters)}")

    # ── Seviye 1 harflerinde başarılı denemeler ──
    print("\n--- Seviye 1 Denemeler ---")
    for harf in SEVIYELER[1]:
        for deneme in range(3):
            skor = random.uniform(7.5, 9.5)
            sonuc = pm.deneme_kaydet(harf, skor, otomatik_kaydet=False)
            if sonuc["yeni_acilan"]:
                print(f"  🔓 {harf} açıldı! (deneme {deneme + 1})")
        print(f"  {harf}: ortalama={pm.harf_ortalama_skor(harf):.1f}, "
              f"en_iyi={pm.harf_en_iyi_skor(harf):.1f}")

    print(f"\n  Seviye tamam mı: {pm.seviye_tamam_mi(1)}")
    print(f"  Mevcut seviye: {pm.current_level}")
    print(f"  Açık harfler: {sorted(pm.unlocked_letters)}")

    # ── Seviye 2 denemeler ──
    print("\n--- Seviye 2 Denemeler ---")
    for harf in SEVIYELER[2]:
        # İlk deneme başarısız
        sonuc = pm.deneme_kaydet(harf, 5.0, otomatik_kaydet=False)
        # 3 başarılı deneme
        for _ in range(3):
            sonuc = pm.deneme_kaydet(harf, random.uniform(7.0, 8.5), otomatik_kaydet=False)
        if sonuc["yeni_acilan"]:
            print(f"  🔓 {harf} açıldı!")

    print(f"  Mevcut seviye: {pm.current_level}")

    # ── Kelime modu kontrolü ──
    print(f"\n  Kelime modu açık mı: {pm.kelime_modu_acik_mi()}")
    print(f"  Açık harf sayısı: {len(pm.unlocked_letters)}")

    if pm.kelime_modu_acik_mi():
        kelimeler = pm.kullanilabilir_kelimeler(max_uzunluk=3)
        print(f"  Kullanılabilir kelimeler (≤3 harf): {kelimeler[:10]}")

        if kelimeler:
            sonuc = pm.kelime_deneme_kaydet(kelimeler[0], 8.0, otomatik_kaydet=False)
            print(f"  Kelime denemesi: {sonuc['mesaj']}")

    # ── İstatistikler ──
    print("\n--- İstatistikler ---")
    ist = pm.istatistikler()
    print(f"  Toplam ilerleme: %{ist['toplam_ilerleme_yuzde']}")
    print(f"  Toplam deneme: {ist['toplam_deneme']}")
    print(f"  Genel ortalama: {ist['genel_ortalama_skor']:.1f}")
    print(f"  En güçlü: {ist['en_guclu_5'][:3]}")
    print(f"  En zayıf: {ist['en_zayif_5'][:3]}")

    # ── Sonraki görev ──
    print("\n--- Sonraki Görev ---")
    gorev = pm.get_next_task()
    print(f"  Tip: {gorev['type']}")
    print(f"  Hedef: {gorev['target']}")
    print(f"  Sebep: {gorev['reason']}")

    # ── Kaydet ve yükle ──
    print("\n--- Kaydet / Yükle ---")
    pm.kaydet()
    print(f"  Profil kaydedildi: {TEST_DOSYA}")

    # Yeni instance ile yükle
    pm2 = ProgressManager(
        username="Yüklenen",
        profil_dosyasi=TEST_DOSYA,
        otomatik_yukle=True,
    )
    print(f"  Yüklenen profil: {pm2}")
    print(f"  Kullanıcı adı: {pm2.username}")
    print(f"  Açık harfler: {sorted(pm2.unlocked_letters)}")
    print(f"  Toplam deneme: {pm2.total_attempts}")

    # ── Özet rapor ──
    print("\n" + pm.ozet_rapor())

    # ── Temizlik ──
    if os.path.exists(TEST_DOSYA):
        os.remove(TEST_DOSYA)
        print(f"\n  Test dosyası silindi: {TEST_DOSYA}")

    print("\n" + "=" * 60)
    print("  Test tamamlandı!")
    print("=" * 60)