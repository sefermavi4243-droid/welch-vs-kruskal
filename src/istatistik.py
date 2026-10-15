"""Grup karşılaştırmaları için ortak yardımcı fonksiyonlar.

Varsayım kontrolüne göre uygun testi seçer, etki büyüklüğünü ve post-hoc
karşılaştırmaları hesaplar. Notebook'lar bu modülü kullanır.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# --------------------------------------------------------------------------- biçim
def p_yaz(p):
    """APA biçiminde p değeri: p < .001 / p = .034"""
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def evet_hayir(kosul):
    return "Evet" if kosul else "Hayır"


def etki_yorumu(deger, esikler):
    """esikler: (küçük, orta, büyük) alt sınırları."""
    deger = abs(deger)
    if deger >= esikler[2]:
        return "büyük"
    if deger >= esikler[1]:
        return "orta"
    if deger >= esikler[0]:
        return "küçük"
    return "önemsiz"


# --------------------------------------------------------------------------- etki büyüklüğü
def eta_kare(*gruplar):
    """Tek yönlü ANOVA için η² = SS_arası / SS_toplam."""
    tum = np.concatenate(gruplar)
    ss_arasi = sum(len(g) * (np.mean(g) - tum.mean()) ** 2 for g in gruplar)
    return ss_arasi / ((tum - tum.mean()) ** 2).sum()


def epsilon_kare(h, n_toplam):
    """Kruskal-Wallis için ε² = H / (N − 1)."""
    return h / (n_toplam - 1)


def welch_payda_sd(*gruplar):
    """Welch ANOVA payda serbestlik derecesi."""
    n = np.array([len(g) for g in gruplar], float)
    w = n / np.array([np.var(g, ddof=1) for g in gruplar])
    k = len(gruplar)
    a = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    return (k ** 2 - 1) / (3 * a)


def omega_kare_welch(f, k, n_toplam):
    """Welch ANOVA için yaklaşık ω² (Carroll & Nordholm, 1975)."""
    return max(0.0, (k - 1) * (f - 1) / ((k - 1) * (f - 1) + n_toplam))


# --------------------------------------------------------------------------- çoklu karşılaştırma
def holm(p_degerleri):
    """Holm-Bonferroni düzeltmesi."""
    p = np.asarray(p_degerleri, dtype=float)
    m = len(p)
    duzeltilmis = np.empty(m)
    enbuyuk = 0.0
    for k, i in enumerate(np.argsort(p)):
        enbuyuk = max(enbuyuk, (m - k) * p[i])
        duzeltilmis[i] = min(enbuyuk, 1.0)
    return duzeltilmis


def _ciftler(adlar):
    return [(i, j) for i in range(len(adlar)) for j in range(i + 1, len(adlar))]


def tukey(gruplar, alpha=ALPHA):
    adlar, veriler = list(gruplar), [np.asarray(v, float) for v in gruplar.values()]
    sonuc = stats.tukey_hsd(*veriler)
    ga = sonuc.confidence_interval(confidence_level=1 - alpha)
    return pd.DataFrame([{
        "Grup 1": adlar[i], "Grup 2": adlar[j],
        "Ort. farkı": veriler[i].mean() - veriler[j].mean(),
        "GA alt": ga.low[i, j], "GA üst": ga.high[i, j],
        "p (düzeltilmiş)": sonuc.pvalue[i, j],
    } for i, j in _ciftler(adlar)])


def games_howell(gruplar, alpha=ALPHA):
    """Varyanslar eşit olmadığında kullanılan post-hoc test."""
    adlar, veriler = list(gruplar), [np.asarray(v, float) for v in gruplar.values()]
    k = len(veriler)
    satirlar = []
    for i, j in _ciftler(adlar):
        a, b = veriler[i], veriler[j]
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        fark = a.mean() - b.mean()
        sd = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        q = abs(fark) / np.sqrt((va + vb) / 2)
        q_kritik = stats.studentized_range.ppf(1 - alpha, k, sd)
        yaricap = q_kritik * np.sqrt((va + vb) / 2)
        satirlar.append({
            "Grup 1": adlar[i], "Grup 2": adlar[j], "Ort. farkı": fark,
            "GA alt": fark - yaricap, "GA üst": fark + yaricap,
            "p (düzeltilmiş)": stats.studentized_range.sf(q, k, sd),
        })
    return pd.DataFrame(satirlar)


def dunn(gruplar):
    """Kruskal-Wallis sonrası Dunn testi (bağ düzeltmeli) + Holm."""
    adlar, veriler = list(gruplar), [np.asarray(v, float) for v in gruplar.values()]
    tum = np.concatenate(veriler)
    n_toplam = len(tum)
    siralar = stats.rankdata(tum)
    _, bag = np.unique(tum, return_counts=True)
    bag_duzeltme = (bag ** 3 - bag).sum() / (12 * (n_toplam - 1))
    sinirlar = np.cumsum([0] + [len(v) for v in veriler])
    ort_sira = [siralar[sinirlar[g]:sinirlar[g + 1]].mean() for g in range(len(veriler))]

    satirlar, ham_p = [], []
    for i, j in _ciftler(adlar):
        se = np.sqrt((n_toplam * (n_toplam + 1) / 12 - bag_duzeltme)
                     * (1 / len(veriler[i]) + 1 / len(veriler[j])))
        z = (ort_sira[i] - ort_sira[j]) / se
        ham_p.append(2 * stats.norm.sf(abs(z)))
        satirlar.append({"Grup 1": adlar[i], "Grup 2": adlar[j],
                         "Medyan farkı": np.median(veriler[i]) - np.median(veriler[j]),
                         "z": z})
    tablo = pd.DataFrame(satirlar)
    tablo["p (düzeltilmiş)"] = holm(ham_p)
    return tablo


# --------------------------------------------------------------------------- ana analizler
@dataclass
class GrupSonucu:
    normallik: pd.DataFrame
    levene_p: float
    yontem: str
    test_adi: str
    istatistik: float
    sd: tuple
    p: float
    etki_adi: str
    etki: float
    etki_duzeyi: str
    posthoc_adi: str
    posthoc: pd.DataFrame
    en_iyi: str
    en_iyi_kesin: bool

    def rapor(self):
        sd = ", ".join(f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}" for x in self.sd)
        harf = "H" if self.yontem == "kruskal" else "F"
        return (f"{self.test_adi}: {harf}({sd}) = {self.istatistik:.2f}, {p_yaz(self.p)}, "
                f"{self.etki_adi} = {self.etki:.3f} ({self.etki_duzeyi} etki)")


def grup_karsilastir(gruplar, alpha=ALPHA, buyuk_iyi=True, zorla=None):
    """Bağımsız k grubu karşılaştırır; testi varsayımlara göre seçer.

    gruplar: {ad: değerler} sözlüğü.
    Normal + homojen -> ANOVA + Tukey; normal + homojen değil -> Welch + Games-Howell;
    normal değil -> Kruskal-Wallis + Dunn.
    zorla: "anova", "welch" veya "kruskal" verilirse varsayım sonucundan bağımsız o test kullanılır.
    """
    adlar = list(gruplar)
    veriler = [np.asarray(v, float) for v in gruplar.values()]
    k, n_toplam = len(veriler), sum(len(v) for v in veriler)

    normallik = pd.DataFrame([{"Grup": a, "n": len(v), "W": stats.shapiro(v).statistic,
                               "p": stats.shapiro(v).pvalue} for a, v in zip(adlar, veriler)])
    normallik["Normal mi?"] = (normallik["p"] > alpha).map(evet_hayir)
    normal_mi = (normallik["p"] > alpha).all()
    levene_p = stats.levene(*veriler).pvalue
    homojen_mi = levene_p > alpha

    if zorla is None:
        zorla = "anova" if normal_mi and homojen_mi else "welch" if normal_mi else "kruskal"

    if zorla == "anova":
        sonuc = stats.f_oneway(*veriler)
        yontem, test_adi, sd = "anova", "Tek yönlü ANOVA", (k - 1, n_toplam - k)
        etki_adi, etki, esik = "η²", eta_kare(*veriler), (0.01, 0.06, 0.14)
        posthoc_adi, posthoc = "Tukey HSD", tukey(gruplar, alpha)
    elif zorla == "welch":
        sonuc = stats.f_oneway(*veriler, equal_var=False)
        yontem, test_adi = "welch", "Welch ANOVA"
        sd = (k - 1, welch_payda_sd(*veriler))
        etki_adi, etki, esik = "ω²", omega_kare_welch(sonuc.statistic, k, n_toplam), (0.01, 0.06, 0.14)
        posthoc_adi, posthoc = "Games-Howell", games_howell(gruplar, alpha)
    else:
        sonuc = stats.kruskal(*veriler)
        yontem, test_adi, sd = "kruskal", "Kruskal-Wallis", (k - 1,)
        etki_adi, etki, esik = "ε²", epsilon_kare(sonuc.statistic, n_toplam), (0.01, 0.08, 0.26)
        posthoc_adi, posthoc = "Dunn + Holm", dunn(gruplar)

    posthoc["Anlamlı mı?"] = (posthoc["p (düzeltilmiş)"] < alpha).map(evet_hayir)

    # En iyi grup: ortalaması (Kruskal-Wallis'te ortalama sırası) en yüksek olan;
    # diğer HER gruptan anlamlı olarak yüksekse "kesin" kabul edilir
    if yontem == "kruskal":
        siralar = stats.rankdata(np.concatenate(veriler))
        sinir = np.cumsum([0] + [len(v) for v in veriler])
        merkez = [siralar[sinir[g]:sinir[g + 1]].mean() for g in range(k)]
    else:
        merkez = [v.mean() for v in veriler]
    sira = np.argsort(merkez)[::-1] if buyuk_iyi else np.argsort(merkez)
    en_iyi = adlar[sira[0]]
    ilgili = posthoc[(posthoc["Grup 1"] == en_iyi) | (posthoc["Grup 2"] == en_iyi)]
    en_iyi_kesin = bool(sonuc.pvalue < alpha and (ilgili["p (düzeltilmiş)"] < alpha).all())

    return GrupSonucu(normallik, levene_p, yontem, test_adi, float(sonuc.statistic), sd,
                      float(sonuc.pvalue), etki_adi, float(etki), etki_yorumu(etki, esik),
                      posthoc_adi, posthoc, en_iyi, en_iyi_kesin)


@dataclass
class EslesikSonuc:
    n: int
    fark_normal_p: float
    yontem: str
    istatistik: float
    p: float
    etki_adi: str
    etki: float
    etki_duzeyi: str
    ort_fark: float
    medyan_fark: float
    ga: tuple

    def rapor(self):
        if self.yontem == "t":
            return (f"Bağımlı t-testi: t({self.n - 1}) = {self.istatistik:.2f}, {p_yaz(self.p)}, "
                    f"{self.etki_adi} = {self.etki:.2f} ({self.etki_duzeyi} etki), "
                    f"ortalama fark = {self.ort_fark:+.2f} [%95 GA {self.ga[0]:.2f}, {self.ga[1]:.2f}]")
        return (f"Wilcoxon işaretli sıralar: W = {self.istatistik:.1f}, {p_yaz(self.p)}, "
                f"{self.etki_adi} = {self.etki:.2f} ({self.etki_duzeyi} etki), "
                f"medyan fark = {self.medyan_fark:+.2f}, ortalama fark = {self.ort_fark:+.2f} "
                f"[%95 bootstrap GA {self.ga[0]:.2f}, {self.ga[1]:.2f}]")


def eslesik_karsilastir(once, sonra, alpha=ALPHA, seed=0):
    """Aynı kişilerin iki ölçümünü karşılaştırır; farkların normalliğine göre test seçer."""
    once, sonra = np.asarray(once, float), np.asarray(sonra, float)
    fark = sonra - once
    normal_p = stats.shapiro(fark).pvalue

    if normal_p > alpha:
        sonuc = stats.ttest_rel(sonra, once)
        ga = sonuc.confidence_interval(confidence_level=1 - alpha)
        dz = fark.mean() / fark.std(ddof=1)
        return EslesikSonuc(len(fark), normal_p, "t", float(sonuc.statistic), float(sonuc.pvalue), "d_z", dz,
                            etki_yorumu(dz, (0.2, 0.5, 0.8)), fark.mean(), np.median(fark),
                            (ga.low, ga.high))

    sonuc = stats.wilcoxon(sonra, once)
    sifirsiz = fark[fark != 0]
    siralar = stats.rankdata(np.abs(sifirsiz))
    r = (siralar[sifirsiz > 0].sum() - siralar[sifirsiz < 0].sum()) / siralar.sum()
    boot = stats.bootstrap((fark,), np.mean, confidence_level=1 - alpha,
                           rng=np.random.default_rng(seed)).confidence_interval
    return EslesikSonuc(len(fark), normal_p, "wilcoxon", float(sonuc.statistic), float(sonuc.pvalue),
                        "r (sıra-çift serisel)", r, etki_yorumu(r, (0.1, 0.3, 0.5)), fark.mean(),
                        np.median(fark), (boot.low, boot.high))
