import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

YIELD_CSV = "BulkYield_output.csv"
HIST_CSV  = "BulkYield_histogram.csv"
SE_THRESHOLD_EV = 50.0

yields = pd.read_csv(YIELD_CSV)
hist   = pd.read_csv(HIST_CSV)
hist["bin_center_ev"] = (hist["bin_min_ev"] + hist["bin_max_ev"]) / 2.0

phases   = sorted(hist["phase"].unique())
energies = sorted(hist["beam_energy_ev"].unique())
colors   = cm.viridis(np.linspace(0.1, 0.9, len(energies)))

def energy_label(ev):
    return f"{int(ev)} eV" if ev < 1000 else f"{int(ev/1000)} keV"

# ── Figure 1: BSE yield vs beam energy ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
for phase, grp in yields.groupby("phase"):
    ax1.plot(grp["BeamE_eV"] / 1000, grp["BSE_yield"], marker="o", label=phase)
ax1.set_xlabel("Beam energy (keV)")
ax1.set_ylabel("BSE yield")
ax1.set_title("Backscattered electron yield – Ni superalloy γ / γ′")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.4)
fig1.tight_layout()
fig1.savefig("BSE_yield_vs_energy.png", dpi=150)
print("Saved BSE_yield_vs_energy.png")

# ── Figure 2: SE yield vs beam energy ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
for phase, grp in yields.groupby("phase"):
    ax2.plot(grp["BeamE_eV"] / 1000, grp["SE_yield"], marker="o", label=phase)
ax2.set_xlabel("Beam energy (keV)")
ax2.set_ylabel("SE yield")
ax2.set_title("Secondary electron yield – Ni superalloy γ / γ′")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)
fig2.tight_layout()
fig2.savefig("SE_yield_vs_energy.png", dpi=150)
print("Saved SE_yield_vs_energy.png")

# ── Figure 3: BSE exit-energy histograms ─────────────────────────────────────
bse = hist[hist["bin_min_ev"] >= SE_THRESHOLD_EV].copy()

fig3, axes3 = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4.5), sharey=False)
if len(phases) == 1:
    axes3 = [axes3]
for ax, phase in zip(axes3, phases):
    sub = bse[bse["phase"] == phase]
    for color, beam_e in zip(colors, energies):
        row = sub[sub["beam_energy_ev"] == beam_e]
        if row.empty:
            continue
        ax.plot(row["bin_center_ev"], row["yield"],
                color=color, label=energy_label(beam_e), linewidth=1.2)
    ax.set_xlabel("Exit energy (eV)")
    ax.set_ylabel("Yield per bin")
    ax.set_title(f"BSE exit-energy spectrum – {phase}")
    ax.legend(title="Beam energy", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
fig3.tight_layout()
fig3.savefig("BSE_histogram.png", dpi=150)
print("Saved BSE_histogram.png")

# ── Figure 4: SE exit-energy histograms (0–50 eV, skip empty 0-10 bin) ───────
se = hist[(hist["bin_min_ev"] >= 10) & (hist["bin_max_ev"] <= SE_THRESHOLD_EV)].copy()

fig4, axes4 = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4.5), sharey=False)
if len(phases) == 1:
    axes4 = [axes4]
for ax, phase in zip(axes4, phases):
    sub = se[se["phase"] == phase]
    for color, beam_e in zip(colors, energies):
        row = sub[sub["beam_energy_ev"] == beam_e]
        if row.empty:
            continue
        ax.plot(row["bin_center_ev"], row["yield"],
                color=color, label=energy_label(beam_e), linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Exit energy (eV)")
    ax.set_ylabel("Yield per bin")
    ax.set_title(f"SE exit-energy spectrum – {phase}")
    ax.legend(title="Beam energy", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
fig4.tight_layout()
fig4.savefig("SE_histogram.png", dpi=150)
print("Saved SE_histogram.png")

plt.show()
