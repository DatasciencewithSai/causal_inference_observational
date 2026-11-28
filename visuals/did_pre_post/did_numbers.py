"""Numbers for did_pre_post.html: a simulated store panel analysed with causal.diff_in_diff.

Story: free delivery launches in 40 stores (treated) at week 9; 40 other stores don't get it.
Summer lifts orders everywhere (common trend) and treated stores were already bigger.
True effect: +8 orders / store / week.

Run: .venv/bin/python visuals/did_pre_post/did_numbers.py
"""

import numpy as np

from causal import diff_in_diff

rng = np.random.default_rng(20261133)
N_STORES, WEEKS, LAUNCH, TRUE_ATT = 80, 16, 9, 8.0

store = np.repeat(np.arange(N_STORES), WEEKS)
week = np.tile(np.arange(1, WEEKS + 1), N_STORES)
treated = (store < N_STORES // 2).astype(int)
post = (week >= LAUNCH).astype(int)

fe = rng.normal(0, 8, N_STORES)
fe[: N_STORES // 2] -= fe[: N_STORES // 2].mean()  # centre store effects within each group
fe[N_STORES // 2 :] -= fe[N_STORES // 2 :].mean()
season = 1.25 * (week - 8.5)  # summer: +10 between pre and post means
y = 110 + 20 * treated + fe[store] + season + TRUE_ATT * treated * post
y = y + rng.normal(0, 6, len(y))

r = diff_in_diff(y, treated, post, clusters=store)
cell = {(d, p): y[(treated == d) & (post == p)].mean() for d in (0, 1) for p in (0, 1)}
c_pre, c_post, t_pre, t_post = cell[0, 0], cell[0, 1], cell[1, 0], cell[1, 1]
print(f"cells  C pre {c_pre:.1f}  C post {c_post:.1f}  T pre {t_pre:.1f}  T post {t_post:.1f}")
print(f"naive pre/post (T)   {t_post - t_pre:+.1f}")
print(f"naive T - C (post)   {t_post - c_post:+.1f}")
print(f"betas b0 {c_pre:.1f} b1 {t_pre - c_pre:+.1f} b2 {c_post - c_pre:+.1f} b3 {r.att:+.1f}")
z = r.att / r.se
print(f"DiD {r.att:.2f}  SE {r.se:.2f}  CI [{r.ci_low:.2f}, {r.ci_high:.2f}]")
print(f"z {z:.2f}  p {r.p_value:.2g}")
print(f"relative to counterfactual {r.att / (t_post - r.att):.1%}")
print(f"naive pre/post / DiD {(t_post - t_pre) / r.att:.1f}x")
print("weekly means:")
for g in (0, 1):
    means = [float(round(y[(treated == g) & (week == w)].mean(), 1)) for w in range(1, WEEKS + 1)]
    print(" ", "T" if g else "C", means)

# Placebo pre-trend check: pretend launch was week 5, using pre-launch weeks only.
pre = week < LAUNCH
fake = diff_in_diff(y[pre], treated[pre], (week[pre] >= 5).astype(int), clusters=store[pre])
print(f"placebo DiD (wk 5, pre only) {fake.att:+.2f}  p {fake.p_value:.2f}")
