"""
Ötös Lottó – webre optimalizált ábrák a cikkhez
4 ábra: gyakoriság, összeg-eloszlás, gap, időtrend
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from pathlib import Path

SRC = Path(__file__).parent
OUT = Path(r"C:\Users\bekes\Documents\GitHub\gaborbekes-site\assets\images\otoslotto")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# ── Beolvasás ─────────────────────────────────────────────────
df = pd.read_csv(SRC / "otos_clean.csv")
df["date"] = pd.to_datetime(df["datum"].str.strip().str.rstrip("."), format="%Y.%m.%d", errors="coerce")
df = df.sort_values("date").reset_index(drop=True)
n = len(df)
cols = ["sz1", "sz2", "sz3", "sz4", "sz5"]

all_numbers = pd.concat([df[c] for c in cols])
freq = Counter(all_numbers)
numbers = list(range(1, 91))
observed = np.array([freq.get(i, 0) for i in numbers])
expected = observed.sum() / 90
sigma = np.sqrt(expected)

# ══════════════════════════════════════════════════════════════
# 1. ÁBRA – Számgyakoriság
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#d63031" if o > expected + 2*sigma
          else "#00b894" if o < expected - 2*sigma
          else "#0984e3" for o in observed]
ax.bar(numbers, observed, color=colors, width=0.85, edgecolor="none")
ax.axhline(expected, color="#2d3436", linewidth=1.8, linestyle="--",
           label=f"Várt átlag: {expected:.0f}")
ax.axhline(expected + 2*sigma, color="#636e72", linewidth=0.9, linestyle=":",
           label=f"±2σ határ")
ax.axhline(expected - 2*sigma, color="#636e72", linewidth=0.9, linestyle=":")
ax.set_xlabel("Lottószám (1–90)")
ax.set_ylabel("Hányszor húzták ki")
ax.set_title(f"Melyik sz\u00e1mot h\u00e1nyszor h\u00fazt\u00e1k ki? ({n} h\u00faz\u00e1s, 1957\u20132026)")
ax.legend(fontsize=11)
ax.set_xlim(0, 91)
fig.savefig(OUT / "szamgyakorisag.png")
plt.close()
print("✓ szamgyakorisag.png")

# ══════════════════════════════════════════════════════════════
# 2. ÁBRA – Összeg eloszlás
# ══════════════════════════════════════════════════════════════
df["osszeg"] = df[cols].sum(axis=1)
expected_mean = 5 * 91 / 2
expected_std = np.sqrt(5 * (90**2 - 1) / 12 * (90 - 5) / (90 - 1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["osszeg"], bins=45, density=True, color="#0984e3", edgecolor="white",
        alpha=0.85, label="Tényleges húzások")
x = np.linspace(40, 420, 300)
ax.plot(x, stats.norm.pdf(x, expected_mean, expected_std), color="#d63031",
        linewidth=2.5, label=f"Elméleti haranggörbe\n(átlag={expected_mean:.0f}, szórás={expected_std:.0f})")
ax.set_xlabel("Az 5 kihúzott szám összege")
ax.set_ylabel("Sűrűség")
ax.set_title("Az ötös lottó számai pont úgy adódnak össze, ahogy kellene")
ax.legend(fontsize=11)
fig.savefig(OUT / "osszeg_eloszlas.png")
plt.close()
print("✓ osszeg_eloszlas.png")

# ══════════════════════════════════════════════════════════════
# 3. ÁBRA – Visszatérési idő
# ══════════════════════════════════════════════════════════════
all_gaps = []
for num in range(1, 91):
    indices = df.index[df[cols].eq(num).any(axis=1)].tolist()
    gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
    all_gaps.extend(gaps)
all_gaps = np.array(all_gaps)
p_geom = 5 / 90

fig, ax = plt.subplots(figsize=(10, 5))
max_gap = 70
bins = np.arange(1, max_gap + 2) - 0.5
ax.hist(all_gaps[all_gaps <= max_gap], bins=bins, density=True, color="#0984e3",
        edgecolor="white", alpha=0.85, label="Tényleges várakozási idők")
x_geom = np.arange(1, max_gap + 1)
ax.plot(x_geom, stats.geom.pmf(x_geom, p_geom), color="#d63031", linewidth=2.5,
        label=f"Elméleti eloszlás")
ax.set_xlabel("Hetek száma két húzás között")
ax.set_ylabel("Sűrűség")
ax.set_title("Mennyi idő telik el, míg egy szám újra kijön?")
ax.set_xlim(0, max_gap)
ax.legend(fontsize=11)
fig.savefig(OUT / "visszateresi_ido.png")
plt.close()
print("✓ visszateresi_ido.png")

# ══════════════════════════════════════════════════════════════
# 4. ÁBRA – Időbeli trend (3-as és 88-as)
# ══════════════════════════════════════════════════════════════
window = 200
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax, num, color, label in [
    (axes[0], 3, "#d63031", "A leggyakoribb: 3-as"),
    (axes[1], 88, "#00b894", "A legritkább: 88-as"),
]:
    hits = df[cols].eq(num).any(axis=1).astype(int)
    rolling = hits.rolling(window, min_periods=window).mean()
    expected_rate = 5 / 90
    sigma_rate = np.sqrt(expected_rate * (1 - expected_rate) / window)

    ax.plot(df["date"], rolling, color=color, linewidth=1.3, label=label)
    ax.axhline(expected_rate, color="#2d3436", linestyle="--", linewidth=1.2,
               label=f"Várt arány ({expected_rate:.3f})")
    ax.fill_between(df["date"],
                    expected_rate - 2*sigma_rate, expected_rate + 2*sigma_rate,
                    color="#b2bec3", alpha=0.3, label="Normál ingadozás (±2σ)")
    ax.set_ylabel("Kihúzás aránya")
    ax.set_title(label)
    ax.legend(loc="upper right", fontsize=10)

axes[1].set_xlabel("Év")
fig.suptitle('Van-e "forr\u00f3" \u00e9s "hideg" korszaka egy-egy sz\u00e1mnak?', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "idotrend.png")
plt.close()
print("✓ idotrend.png")

print("\n── Mind a 4 ábra kész! ──")
