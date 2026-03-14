#!/usr/bin/env python3
"""
Final Fig 2 with refined C_E extrapolation.
Critical (Δ=0.5):  C_E(L) = a·ln(L) + b         [CFT log-law]
Gapped  (Δ=2.0):   C_E(L) = C∞ + B/L             [power-law finite-size correction]
"""
import math, json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

VAULT = Path("/sessions/beautiful-modest-einstein/mnt/Obsidian Vault")
ln2 = math.log(2)

# ── Known ED data (nats) ──────────────────────────────────────────────────────
L_ed  = np.array([8, 12, 16, 20])

S_c  = np.array([0.69709233*ln2, 0.81831258*ln2, 0.90232597*ln2, 0.669723])
CE_c = np.array([0.97145468, 1.00565539, 1.02306821, 1.034785])

S_g  = np.array([0.75923906*ln2, 0.91172657*ln2, 1.02442827*ln2, 0.77137315])
CE_g = np.array([0.89033379, 0.87098614, 0.83501226, 0.79518542])

# ── Model definitions ─────────────────────────────────────────────────────────
def log_model(L, a, b):     return a * np.log(L) + b
def inv_model(L, Cinf, B):  return Cinf + B / L

# ── Fit C_E ───────────────────────────────────────────────────────────────────
popt_CE_c, _ = curve_fit(log_model, L_ed, CE_c, p0=[0.1, 0.7])
popt_CE_g, _ = curve_fit(inv_model, L_ed, CE_g, p0=[0.5, 5.0])
a_c, b_c = popt_CE_c
Cinf_g, B_g = popt_CE_g

# ── Fit S ─────────────────────────────────────────────────────────────────────
popt_S_c, _ = curve_fit(log_model, L_ed, S_c, p0=[0.3, -0.1])
popt_S_g, _ = curve_fit(inv_model, L_ed, S_g, p0=[1.0, -5.0])

def R2(y, yfit): return 1 - np.var(y - yfit) / np.var(y)

print("Fit quality:")
print(f"  C_E, crit (log):  R²={R2(CE_c, log_model(L_ed,*popt_CE_c)):.5f}  "
      f"C_E(L) = {a_c:.4f}·ln(L) + {b_c:.4f}")
print(f"  C_E, gapped (1/L): R²={R2(CE_g, inv_model(L_ed,*popt_CE_g)):.5f}  "
      f"C_E(L) = {Cinf_g:.4f} + {B_g:.4f}/L")
print(f"  C_E(∞) gapped = {Cinf_g:.4f} nats")

# ── Full L grid ───────────────────────────────────────────────────────────────
L_all    = np.array([8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
L_extrap = np.array([24, 28, 32, 36, 40, 44])
L_fine   = np.linspace(6, 48, 400)

CE_c_all = log_model(L_all, *popt_CE_c)
CE_g_all = inv_model(L_all, *popt_CE_g)
S_c_all  = log_model(L_all, *popt_S_c)
S_g_all  = inv_model(L_all, *popt_S_g)

# Override exact ED points
for i, L in enumerate(L_all):
    if L in L_ed:
        idx = list(L_ed).index(L)
        CE_c_all[i] = CE_c[idx]; CE_g_all[i] = CE_g[idx]
        S_c_all[i]  = S_c[idx];  S_g_all[i]  = S_g[idx]

print("\n--- Full table ---")
print(f"{'L':>4} | {'S_crit':>9} {'CE_crit':>9} {'ratio':>7} | {'S_gap':>9} {'CE_gap':>9} {'ratio':>7}")
for i, L in enumerate(L_all):
    src = "  " if L in L_ed else " *"
    print(f"{L:>4}{src}| {S_c_all[i]:9.5f} {CE_c_all[i]:9.5f} {S_c_all[i]/CE_c_all[i]:7.4f} | "
          f"{S_g_all[i]:9.5f} {CE_g_all[i]:9.5f} {S_g_all[i]/CE_g_all[i]:7.4f}")
print("(* = extrapolated via model fit)")

# ── Save JSON ─────────────────────────────────────────────────────────────────
ce_extended = {
    "fit_params": {
        "critical_CE": {"model": "a*ln(L)+b", "a": float(a_c), "b": float(b_c)},
        "gapped_CE":   {"model": "Cinf+B/L",  "Cinf": float(Cinf_g), "B": float(B_g)}
    },
    "critical_0p5": [],
    "gapped_2p0":   []
}
for i, L in enumerate(L_all):
    src = "ED" if L in L_ed else "extrap"
    ce_extended["critical_0p5"].append({"L": int(L), "S_nats": float(S_c_all[i]),
                                         "CE": float(CE_c_all[i]), "source": src})
    ce_extended["gapped_2p0"].append(  {"L": int(L), "S_nats": float(S_g_all[i]),
                                         "CE": float(CE_g_all[i]), "source": src})

with open(VAULT / "ce_extended.json", "w") as f:
    json.dump(ce_extended, f, indent=2)
print("\nSaved ce_extended.json")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
blue, red = "#1f77b4", "#d62728"
LABEL_C = r"$\Delta=0.5$ (critical)"
LABEL_G = r"$\Delta=2.0$ (gapped)"

for panel_idx, (ax, yc, yg, ylabel, title) in enumerate(zip(
    axes,
    [S_c_all, CE_c_all, S_c_all/CE_c_all],
    [S_g_all, CE_g_all, S_g_all/CE_g_all],
    [r"$S$ (nats)", r"$C_E$ (nats)", r"$S/C_E$"],
    ["Entanglement Entropy", "Capacity of Entanglement", r"Ratio $S/C_E$"]
)):
    # smooth fit curves
    yc_fine = (log_model(L_fine, *popt_S_c) if panel_idx == 0 else
               log_model(L_fine, *popt_CE_c) if panel_idx == 1 else
               log_model(L_fine, *popt_S_c) / log_model(L_fine, *popt_CE_c))
    yg_fine = (inv_model(L_fine, *popt_S_g) if panel_idx == 0 else
               inv_model(L_fine, *popt_CE_g) if panel_idx == 1 else
               inv_model(L_fine, *popt_S_g) / inv_model(L_fine, *popt_CE_g))
    ax.plot(L_fine, yc_fine, "--", color=blue, lw=1.3, alpha=0.6)
    ax.plot(L_fine, yg_fine, "--", color=red,  lw=1.3, alpha=0.6)

    # ED points (solid circles)
    mask_ed = np.isin(L_all, L_ed)
    ax.plot(L_all[mask_ed],  yc[mask_ed],  "o", color=blue, ms=6.5, zorder=5, label=LABEL_C)
    ax.plot(L_all[mask_ed],  yg[mask_ed],  "o", color=red,  ms=6.5, zorder=5, label=LABEL_G)
    # Extrapolated (open squares)
    ax.plot(L_all[~mask_ed], yc[~mask_ed], "s", color=blue, ms=6.5, mfc="white", zorder=4)
    ax.plot(L_all[~mask_ed], yg[~mask_ed], "s", color=red,  ms=6.5, mfc="white", zorder=4)

    if panel_idx == 2:
        ax.axhline(1.0, color="gray", lw=0.9, ls=":", zorder=1)

    ax.set_xlabel(r"$L$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(5, 47)
    ax.tick_params(labelsize=10)
    if panel_idx == 0:
        ax.legend(fontsize=9, loc="upper left")

# Fit-info boxes
axes[1].text(0.97, 0.97,
    rf"crit: ${a_c:.3f}\ln L{b_c:+.3f}$" + "\n" + rf"gapped: ${Cinf_g:.3f}+{B_g:.3f}/L$",
    transform=axes[1].transAxes, fontsize=7.5, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.8))

# Shared legend for marker styles
leg_handles = [
    Line2D([0],[0], marker="o", color="gray", ms=6, ls="none", label="ED (exact)"),
    Line2D([0],[0], marker="s", color="gray", ms=6, ls="none", mfc="white", label="Extrapolated"),
    Line2D([0],[0], color="gray", lw=1.3, ls="--", label="Model fit"),
]
fig.legend(handles=leg_handles, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

plt.tight_layout(rect=[0, 0.07, 1, 1])
for ext in ["pdf", "png"]:
    p = VAULT / f"XXZ_Fig2_CE.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
plt.close()
print("Done.")
