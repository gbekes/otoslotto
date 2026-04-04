"""
Ötös Lottó elemzés: Csalnak-e a húzásnál?
Adatforrás: szerencsejatek.hu (otos.csv)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 5),
    "font.size": 11,
})

# ── 1. Beolvasás ──────────────────────────────────────────────
raw = pd.read_csv(
    OUT / "otos_raw.csv",
    sep=";",
    header=None,
    encoding="utf-8-sig",
    names=[
        "ev", "het", "datum",
        "ot_db", "ot_dij",
        "negy_db", "negy_dij",
        "harom_db", "harom_dij",
        "ketto_db", "ketto_dij",
        "sz1", "sz2", "sz3", "sz4", "sz5",
    ],
)

# Számok tisztítása (néhol szóköz van)
for c in ["sz1", "sz2", "sz3", "sz4", "sz5"]:
    raw[c] = pd.to_numeric(raw[c].astype(str).str.strip(), errors="coerce")

raw = raw.dropna(subset=["sz1", "sz2", "sz3", "sz4", "sz5"])
for c in ["sz1", "sz2", "sz3", "sz4", "sz5"]:
    raw[c] = raw[c].astype(int)

n_draws = len(raw)
print(f"Húzások száma: {n_draws}")

# Összes húzott szám
all_numbers = pd.concat([raw[c] for c in ["sz1", "sz2", "sz3", "sz4", "sz5"]])
freq = Counter(all_numbers)

# ── 2. Alapstatisztikák ──────────────────────────────────────
numbers = list(range(1, 91))
observed = np.array([freq.get(n, 0) for n in numbers])
total_balls = observed.sum()
expected_per_number = total_balls / 90  # egyenletes eloszlás

print(f"Összes húzott golyó: {total_balls}")
print(f"Várt gyakoriság számonként: {expected_per_number:.1f}")
print(f"Leggyakoribb: {observed.max()} (szám {numbers[observed.argmax()]})")
print(f"Legritkább:   {observed.min()} (szám {numbers[observed.argmin()]})")

# ── 3. Khi-négyzet próba ─────────────────────────────────────
expected = np.full(90, expected_per_number)
chi2, p_value = stats.chisquare(observed, f_exp=expected)
print(f"\nKhi-négyzet statisztika: {chi2:.2f}")
print(f"Szabadsági fok: 89")
print(f"p-érték: {p_value:.4f}")

# ── 4. Kolmogorov–Szmirnov teszt ─────────────────────────────
# Az empirikus eloszlást hasonlítjuk a diszkrét egyenleteshez
ks_stat, ks_p = stats.kstest(all_numbers, "uniform", args=(0.5, 90))
print(f"\nKS-teszt statisztika: {ks_stat:.4f}")
print(f"KS p-érték: {ks_p:.4f}")

# ── 5. Ábra 1: Számgyakoriság oszlopdiagram ──────────────────
fig, ax = plt.subplots()
colors = ["#e74c3c" if o > expected_per_number + 2 * np.sqrt(expected_per_number)
          else "#2ecc71" if o < expected_per_number - 2 * np.sqrt(expected_per_number)
          else "#3498db" for o in observed]
ax.bar(numbers, observed, color=colors, width=0.8, edgecolor="none")
ax.axhline(expected_per_number, color="black", linewidth=1.5, linestyle="--",
           label=f"Várt ({expected_per_number:.0f})")
sigma = np.sqrt(expected_per_number)
ax.axhline(expected_per_number + 2 * sigma, color="gray", linewidth=0.8,
           linestyle=":", label=f"±2σ ({expected_per_number + 2*sigma:.0f} / {expected_per_number - 2*sigma:.0f})")
ax.axhline(expected_per_number - 2 * sigma, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel("Lottószám (1–90)")
ax.set_ylabel("Húzások száma")
ax.set_title(f"Ötös Lottó – számgyakoriságok ({n_draws} húzás, 2004–2026)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "abra1_szamgyakorisag.png")
plt.close()
print("\n✓ abra1_szamgyakorisag.png mentve")

# ── 6. Ábra 2: Top 10 leggyakoribb és legritkább szám ────────
top10 = sorted(freq.items(), key=lambda x: -x[1])[:10]
bot10 = sorted(freq.items(), key=lambda x: x[1])[:10]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title, color in [
    (axes[0], top10, "10 leggyakoribb szám", "#e74c3c"),
    (axes[1], bot10, "10 legritkább szám", "#2ecc71"),
]:
    nums, freqs = zip(*data)
    ax.barh([str(n) for n in nums], freqs, color=color, edgecolor="none")
    ax.axvline(expected_per_number, color="black", linestyle="--", label=f"Várt ({expected_per_number:.0f})")
    ax.set_xlabel("Húzások száma")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
fig.suptitle(f"Ötös Lottó – szélső értékek ({n_draws} húzás)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "abra2_top_bottom10.png")
plt.close()
print("✓ abra2_top_bottom10.png mentve")

# ── 7. Ábra 3: Eltérés a várttól (standardizált) ─────────────
z_scores = (observed - expected_per_number) / np.sqrt(expected_per_number)
fig, ax = plt.subplots()
colors_z = ["#e74c3c" if abs(z) > 2 else "#3498db" for z in z_scores]
ax.bar(numbers, z_scores, color=colors_z, width=0.8, edgecolor="none")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(2, color="gray", linestyle=":", linewidth=0.8, label="±2σ határ")
ax.axhline(-2, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Lottószám (1–90)")
ax.set_ylabel("Standardizált eltérés (z-érték)")
ax.set_title("Eltérés az egyenletes eloszlástól")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "abra3_z_ertekek.png")
plt.close()
print("✓ abra3_z_ertekek.png mentve")

# ── 8. Összefoglaló táblázat mentése ─────────────────────────
results = {
    "Húzások száma": n_draws,
    "Összes húzott golyó": int(total_balls),
    "Várt gyakoriság/szám": round(expected_per_number, 1),
    "Leggyakoribb szám": f"{numbers[observed.argmax()]} ({observed.max()}x)",
    "Legritkább szám": f"{numbers[observed.argmin()]} ({observed.min()}x)",
    "Khi-négyzet": round(chi2, 2),
    "Khi-négyzet p-érték": round(p_value, 4),
    "KS-teszt statisztika": round(ks_stat, 4),
    "KS-teszt p-érték": round(ks_p, 4),
}

with open(OUT / "eredmenyek.txt", "w", encoding="utf-8") as f:
    f.write("Ötös Lottó statisztikai elemzés – összefoglaló\n")
    f.write("=" * 50 + "\n\n")
    for k, v in results.items():
        f.write(f"{k:30s}: {v}\n")
    f.write("\n\nTop 10 leggyakoribb szám:\n")
    for num, cnt in top10:
        f.write(f"  {num:3d}: {cnt}x\n")
    f.write("\nTop 10 legritkább szám:\n")
    for num, cnt in bot10:
        f.write(f"  {num:3d}: {cnt}x\n")
    f.write(f"\nKonklúzió: A khi-négyzet próba p-értéke {p_value:.4f}.\n")
    if p_value > 0.05:
        f.write("Ez nem szignifikáns (p > 0.05), tehát nincs statisztikai bizonyíték csalásra.\n")
    else:
        f.write("Ez szignifikáns (p < 0.05), ami további vizsgálatot igényel.\n")

print("✓ eredmenyek.txt mentve")

# ── 9. Tisztított adatfile mentése ────────────────────────────
clean = raw[["ev", "het", "datum", "sz1", "sz2", "sz3", "sz4", "sz5"]].copy()
clean.to_csv(OUT / "otos_clean.csv", index=False, encoding="utf-8")
print("✓ otos_clean.csv mentve")

print("\n── Kész! ──")
