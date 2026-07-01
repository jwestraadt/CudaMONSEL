"""
Analyze the René 65 γ′ BSE-contrast studies (studies/1..4) and write a contrast
summary figure into each study folder. Reuses helpers from trajectory_plot.py.

    python tools/study_analyze.py            # analyze all studies it finds
    python tools/study_analyze.py energy     # just one (energy|carbon|oxide|filter)
Run from the CudaMONSEL project dir (paths are relative to it).
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trajectory_plot as tp

STUD = "studies"
HEINRICH = -5.8   # intrinsic Heinrich mass-avg BSE Z-contrast (gammaP vs gamma), %


def _contrast_grids(base, radius, core_f=0.6, mtx_pad=15.0):
    im = tp.read_image(base)
    if im is None:
        return None
    hw = im["hw"]
    core_r, mtx_r = core_f * radius, min(radius + mtx_pad, 0.95 * hw)
    out = {"im": im, "hw": hw}
    for k in ("SE", "BSE", "total"):
        out[k] = tp.region_contrast(im["grids"][k], hw, 0.0, 0.0, core_r, mtx_r)[2]
    return out


# --------------------------------------------------------------------------- #
def analyze_energy(folder):
    p = os.path.join(folder, "energy_sweep.csv")
    if not os.path.exists(p):
        print("  skip energy: no", p); return
    data = {}
    for r in csv.DictReader(open(p, newline="")):
        data.setdefault(r["phase"], {})[float(r["BeamE_eV"])] = {
            k: float(r[k + "_yield"]) for k in ("BSE", "SE", "total")}
    g, gp = "gamma_R65", "gammaP_R65"
    Es = sorted(data[g])
    def contrast(kind):
        return [100.0 * (data[gp][E][kind] - data[g][E][kind]) / data[g][E][kind] for E in Es]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for kind, c, m in [("BSE", "#2e8b3d", "s"), ("SE", "#e08a3c", "o"), ("total", "black", "^")]:
        a1.semilogx(Es, contrast(kind), "-" + m, color=c, lw=2, label=kind)
    a1.axhline(0, color="0.5", lw=1); a1.axhline(HEINRICH, color="green", ls=":", lw=1.5)
    a1.text(Es[-1], HEINRICH, " Heinrich Z-limit (-5.8%)", color="green", va="center", ha="right", fontsize=9)
    a1.set_xlabel("beam energy (eV)"); a1.set_ylabel("γ′ vs γ contrast (%)")
    a1.set_title("Intrinsic contrast vs energy (bulk γ vs γ′)"); a1.grid(True, ls=":", alpha=0.4); a1.legend()
    a2.semilogx(Es, [data[g][E]["BSE"] for E in Es], "-s", color="#7fbf7f", label="γ BSE η")
    a2.semilogx(Es, [data[gp][E]["BSE"] for E in Es], "-s", color="#2e8b3d", label="γ′ BSE η")
    a2.set_xlabel("beam energy (eV)"); a2.set_ylabel("backscatter yield η")
    a2.set_title("BSE yield vs energy"); a2.grid(True, ls=":", alpha=0.4); a2.legend()
    fig.suptitle("Exp 1 — René 65 γ/γ′ intrinsic BSE Z-contrast emerges with beam energy",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(folder, "energy_sweep_contrast.png"); fig.savefig(out, dpi=140); plt.close(fig)
    bse = contrast("BSE")
    print("wrote %s  (BSE contrast %+.1f%% @200eV -> %+.1f%% @%.0feV)" % (out, bse[0], bse[-1], Es[-1]))


def analyze_carbon(folder):
    toks = ["c0p0", "c0p5", "c1p0", "c2p0", "c3p0", "c5p0"]
    bases = [os.path.join(folder, "dc_" + t) for t in toks]
    geom = tp.read_geometry(os.path.join(folder, "diff_carbon"))
    R = geom["radius"] if geom else 30.0
    xs, se, bse, tot, ims = [], [], [], [], []
    for t, b in zip(toks, bases):
        c = _contrast_grids(b, R)
        xs.append(float(t[1:].replace("p", ".")))
        if c:
            se.append(c["SE"]); bse.append(c["BSE"]); tot.append(c["total"]); ims.append(c["im"])
        else:
            se.append(np.nan); bse.append(np.nan); tot.append(np.nan); ims.append(None)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.axhspan(0, 1e4, color="#e8f5e9"); ax.axhspan(-1e4, 0, color="#fde8e8")
    ax.axhline(0, color="0.4", lw=1)
    ax.plot(xs, se, "-o", color="#e08a3c", lw=2, label="SE")
    ax.plot(xs, bse, "-s", color="#2e8b3d", lw=2, label="BSE (>50 eV)")
    ax.plot(xs, tot, "-^", color="black", lw=2.4, label="total (T3)")
    ax.set_ylim(min(-2, np.nanmin(tot + bse) * 1.2), max(np.nanmax(se) * 1.15, 2))
    ax.set_xlabel("uniform carbon thickness (nm)  [cumulative-scanning proxy]")
    ax.set_ylabel("γ′ core vs γ matrix contrast (%)")
    ax.set_title("Exp 2 — dark-γ′ deepens as (uniform) carbon thickens @ 1 keV", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4); ax.legend()
    fig.tight_layout()
    out = os.path.join(folder, "diff_carbon_contrast.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s  (total %+.1f%% @0nm -> %+.1f%% @5nm)" % (out, tot[0], tot[-1]))


def analyze_oxide(folder):
    cases = [("gp_clean_200V", "clean γ′  200 eV"), ("gp_oxide_200V", "oxidized γ′  200 eV"),
             ("gp_clean_1kV", "clean γ′  1 keV"), ("gp_oxide_1kV", "oxidized γ′  1 keV")]
    geom = tp.read_geometry(os.path.join(folder, "gp_oxide"))
    R = geom["radius"] if geom else 30.0
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    for ax, (name, label) in zip(axes, cases):
        c = _contrast_grids(os.path.join(folder, name), R)
        if c is None:
            ax.set_axis_off(); continue
        hw = c["hw"]
        ax.imshow(c["im"]["grids"]["total"], origin="upper", extent=[-hw, hw, -hw, hw], cmap="gray")
        ax.add_patch(Circle((0, 0), R, fill=False, ec="cyan", ls="--", lw=1.2))
        ax.set_title("%s\ntotal %+.1f%%, BSE %+.1f%%" % (label, c["total"], c["BSE"]), fontsize=10)
        ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50])
    fig.suptitle("Exp 3 — a light (Al,Ti)-oxide skin turns γ′ strongly dark (bounding case)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(folder, "gp_oxide_images.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s" % out)


def analyze_filter(folder):
    fields = [("bse_50eV", 0.025), ("bse_25pct", 0.25), ("bse_50pct", 0.50), ("bse_75pct", 0.75)]
    geom = tp.read_geometry(os.path.join(folder, "bse_filter"))
    R = geom["radius"] if geom else 120.0
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for tag, e0 in [("bsf_2kV", 2000), ("bsf_5kV", 5000)]:
        p = os.path.join(folder, tag + ".csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, newline="")))
        xs = np.array(sorted(set(float(r["x_nm"]) for r in rows)))
        ys = np.array(sorted(set(float(r["y_nm"]) for r in rows)))
        nx, ny = len(xs), len(ys); hw = max(abs(xs).max(), abs(ys).max())
        fr, cs = [], []
        for name, frac in fields:
            col = "det_" + name
            if col not in rows[0]:
                continue
            grid = np.array([float(r[col]) for r in rows]).reshape(ny, nx)
            fr.append(frac); cs.append(tp.region_contrast(grid, hw, 0, 0, 0.6 * R, min(R + 15, 0.95 * hw))[2])
        ax.plot(fr, cs, "-o", lw=2, label="%g keV" % (e0 / 1000))
    ax.axhline(0, color="0.5", lw=1); ax.axhline(HEINRICH, color="green", ls=":", lw=1.5)
    ax.text(0.75, HEINRICH, " Heinrich Z-limit", color="green", va="center", fontsize=9)
    ax.set_xlabel("BSE energy window  (min escape energy / E$_0$)")
    ax.set_ylabel("γ′ core vs γ matrix contrast (%)")
    ax.set_title("Exp 4 — filtering to high-energy true backscatter sharpens dark-γ′", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4); ax.legend()
    fig.tight_layout()
    out = os.path.join(folder, "bse_filter_contrast.png"); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote %s" % out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"energy": (analyze_energy, "1_energy_sweep"), "carbon": (analyze_carbon, "2_diff_carbon"),
            "oxide": (analyze_oxide, "3_gp_oxide"), "filter": (analyze_filter, "4_bse_filter")}
    for key, (fn, sub) in jobs.items():
        if which in ("all", key):
            folder = os.path.join(STUD, sub)
            if os.path.isdir(folder):
                try:
                    fn(folder)
                except Exception as e:
                    print("  %s failed: %s" % (key, e))


if __name__ == "__main__":
    main()
