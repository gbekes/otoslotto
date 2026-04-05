"""
Ötös Lottó elemzés – sanity check tesztek
Ellenőrzi, hogy az adatok és a számítások konzisztensek.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
from pathlib import Path

SRC = Path(__file__).parent
df = pd.read_csv(SRC / "otos_clean.csv")
cols = ["sz1", "sz2", "sz3", "sz4", "sz5"]
n = len(df)
all_numbers = pd.concat([df[c] for c in cols])
freq = Counter(all_numbers)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  -- {detail}")


print("=" * 60)
print("TEST 1: Adatintegritás")
print("=" * 60)

# 1a. Minden szám 1-90 között van
all_vals = all_numbers.values
check("Minden szám >= 1",
      all_vals.min() >= 1,
      f"Min = {all_vals.min()}")
check("Minden szám <= 90",
      all_vals.max() <= 90,
      f"Max = {all_vals.max()}")

# 1b. Soronként 5 különböző szám van (nincs duplikátum egy húzáson belül)
dupes = df[cols].apply(lambda row: len(set(row)) < 5, axis=1).sum()
check("Nincs duplikátum egy húzáson belül",
      dupes == 0,
      f"{dupes} húzásban van ismétlődés")

# 1c. Összes golyó = húzások * 5
total_balls = sum(freq.values())
check(f"Összes golyó = {n} * 5 = {n*5}",
      total_balls == n * 5,
      f"Kapott: {total_balls}")

# 1d. Az adatok 1957-ből indulnak (nem 2004!)
min_year = df["ev"].min()
check(f"Legkorábbi év = 1957 (kapott: {min_year})",
      min_year == 1957,
      f"A legkorábbi év {min_year}, nem 1957!")
max_year = df["ev"].max()
check(f"Legkésőbbi év = 2026 (kapott: {max_year})",
      max_year == 2026,
      f"Kapott: {max_year}")

# 1e. Mind a 90 szám előfordul legalább egyszer
missing = [i for i in range(1, 91) if freq.get(i, 0) == 0]
check("Mind a 90 szám előfordul",
      len(missing) == 0,
      f"Hiányzó számok: {missing}")


print()
print("=" * 60)
print("TEST 2: Gyakoriságok konzisztenciája")
print("=" * 60)

# 2a. A gyakoriságok összege = összes golyó
freq_sum = sum(freq[i] for i in range(1, 91))
check(f"Gyakoriságok összege = {total_balls}",
      freq_sum == total_balls,
      f"Kapott: {freq_sum}")

# 2b. A leggyakoribb szám a 3-as (240x) – egyezik-e a cikkel?
most_common_num, most_common_cnt = freq.most_common(1)[0]
check("Leggyakoribb szám = 3",
      most_common_num == 3,
      f"Kapott: {most_common_num}")
check("Leggyakoribb szám gyakorisága = 240",
      most_common_cnt == 240,
      f"Kapott: {most_common_cnt}")

# 2c. A legritkább szám a 88-as (158x) – egyezik-e a cikkel?
least_common = sorted(freq.items(), key=lambda x: x[1])[0]
check("Legritkább szám = 88",
      least_common[0] == 88,
      f"Kapott: {least_common[0]}")
check("Legritkább szám gyakorisága = 158",
      least_common[1] == 158,
      f"Kapott: {least_common[1]}")

# 2d. Átlagos gyakoriság ≈ 200.2
avg_freq = total_balls / 90
check(f"Átlagos gyakoriság ≈ 200.2 (kapott: {avg_freq:.1f})",
      abs(avg_freq - 200.2) < 0.1,
      f"Eltérés túl nagy: {avg_freq:.4f}")


print()
print("=" * 60)
print("TEST 3: Khi-négyzet próba reprodukálása")
print("=" * 60)

observed = np.array([freq.get(i, 0) for i in range(1, 91)])
expected_val = total_balls / 90
expected_arr = np.full(90, expected_val)
chi2, p_value = stats.chisquare(observed, f_exp=expected_arr)

check(f"Khi-négyzet ≈ 108.0 (kapott: {chi2:.2f})",
      abs(chi2 - 107.99) < 0.5,
      f"Eltérés: {abs(chi2 - 107.99):.2f}")
check(f"p-érték ≈ 0.084 (kapott: {p_value:.4f})",
      abs(p_value - 0.0835) < 0.005,
      f"Eltérés: {abs(p_value - 0.0835):.4f}")
check("p-érték > 0.05 (nem szignifikáns)",
      p_value > 0.05,
      f"p = {p_value:.4f} < 0.05!")


print()
print("=" * 60)
print("TEST 4: Összeg-eloszlás ellenőrzése")
print("=" * 60)

sums = df[cols].sum(axis=1)
obs_mean = sums.mean()
obs_std = sums.std()
theo_mean = 5 * 91 / 2  # = 227.5
theo_std = np.sqrt(5 * (90**2 - 1) / 12 * (90 - 5) / (90 - 1))  # ≈ 56.8

check(f"Összeg-átlag közel az elméletihez (kapott: {obs_mean:.1f}, várt: {theo_mean:.1f})",
      abs(obs_mean - theo_mean) < 5,
      f"Eltérés: {abs(obs_mean - theo_mean):.2f}")
check(f"Összeg-szórás közel az elméletihez (kapott: {obs_std:.1f}, várt: {theo_std:.1f})",
      abs(obs_std - theo_std) < 3,
      f"Eltérés: {abs(obs_std - theo_std):.2f}")

# Az összeg sosem lehet kisebb mint 1+2+3+4+5=15 vagy nagyobb mint 86+87+88+89+90=440
check(f"Min összeg >= 15 (kapott: {sums.min()})",
      sums.min() >= 15,
      f"Lehetetlen összeg: {sums.min()}")
check(f"Max összeg <= 440 (kapott: {sums.max()})",
      sums.max() <= 440,
      f"Lehetetlen összeg: {sums.max()}")


print()
print("=" * 60)
print("TEST 5: Visszatérési idő (gap) ellenőrzése")
print("=" * 60)

df_sorted = df.sort_values(["ev", "het"]).reset_index(drop=True)
all_gaps = []
for num in range(1, 91):
    indices = df_sorted.index[df_sorted[cols].eq(num).any(axis=1)].tolist()
    gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
    all_gaps.extend(gaps)
all_gaps = np.array(all_gaps)

gap_mean = all_gaps.mean()
theo_gap = 90 / 5  # = 18.0

check(f"Átlagos gap ≈ 18.0 (kapott: {gap_mean:.2f})",
      abs(gap_mean - theo_gap) < 0.5,
      f"Eltérés: {abs(gap_mean - theo_gap):.2f}")
check("Minden gap >= 1",
      all_gaps.min() >= 1,
      f"Min gap = {all_gaps.min()}")
check(f"Gap-ek száma konzisztens (kapott: {len(all_gaps)})",
      len(all_gaps) > 15000,
      f"Túl kevés gap: {len(all_gaps)}")


print()
print("=" * 60)
print(f"EREDMÉNY: {passed} PASS / {failed} FAIL  (összesen {passed+failed})")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    print("\nMinden teszt PASS -- a cikkben szereplő számok rendben vannak.")
