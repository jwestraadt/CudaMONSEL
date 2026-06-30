"""
Phase 1 detector post-processor for CudaMONSEL composite_image escape histograms.

Reads the per-pixel (escape-energy x take-off-angle) histogram written by the
GPU backend (``escape_histogram`` block in the JSON deck) and applies detector
acceptance windows in (energy, polar-angle beta) space to synthesize
detector-channel images WITHOUT re-running the Monte Carlo.

Usage:
    python detector_postprocess.py <hist>.json [--csv <run>.csv] [--out-prefix PFX]

beta = take-off polar angle from the outward optic axis (-z):
    beta = 0 deg  -> straight up the column (in-lens)
    beta = 90 deg -> grazing along the surface
"""
import argparse, json, csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_histogram(json_path):
    with open(json_path) as f:
        meta = json.load(f)
    bin_path = meta["bin_file"]
    if not os.path.isabs(bin_path):
        bin_path = os.path.join(os.path.dirname(os.path.abspath(json_path)), os.path.basename(bin_path))
    nx, ny = meta["nx"], meta["ny"]
    nE, nB = meta["energy_bins"], meta["angle_bins"]
    data = np.fromfile(bin_path, dtype=np.int32)
    expected = nx * ny * nE * nB
    if data.size != expected:
        raise ValueError("bin size %d != expected %d" % (data.size, expected))
    hist = data.reshape(ny, nx, nE, nB)          # [row][col][energy_bin][angle_bin]
    return meta, hist


def bin_centers(meta):
    eW = meta["energy_bin_width_ev"]
    bW = meta["angle_bin_width_deg"]
    e_centers = (np.arange(meta["energy_bins"]) + 0.5) * eW     # eV
    b_centers = (np.arange(meta["angle_bins"]) + 0.5) * bW      # deg
    return e_centers, b_centers


def wd_ring_to_beta(working_distance_mm, inner_mm, outer_mm):
    """Annulus radius range at a working distance -> polar take-off window (deg)."""
    lo = math.degrees(math.atan(inner_mm / working_distance_mm))
    hi = math.degrees(math.atan(outer_mm / working_distance_mm))
    return lo, hi


def detector_mask(meta, energy_window, beta_window):
    """Boolean [nE, nB] mask for an (energy_eV, beta_deg) acceptance box."""
    e_c, b_c = bin_centers(meta)
    e_lo, e_hi = energy_window
    b_lo, b_hi = beta_window
    e_mask = (e_c >= e_lo) & (e_c < e_hi)
    b_mask = (b_c >= b_lo) & (b_c <= b_hi)
    return np.outer(e_mask, b_mask)


def signal_map(hist, mask, traj):
    """Per-pixel detector signal = collected counts / trajectories."""
    return (hist * mask[None, None, :, :]).sum(axis=(2, 3)) / float(traj)


def region_means(arr, meta, core_r=15.0, mtx_r=60.0):
    ny, nx = arr.shape
    hw = meta["half_width_nm"]
    xs = -hw + np.arange(nx) * (2 * hw / (nx - 1))
    ys = hw - np.arange(ny) * (2 * hw / (ny - 1))
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X - meta.get("center_x_nm", 0.0), Y - meta.get("center_y_nm", 0.0))
    core = arr[R < core_r]
    mtx = arr[R > mtx_r]
    return float(core.mean()), float(mtx.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist_json")
    ap.add_argument("--csv", default=None, help="run CSV for full-acceptance validation")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    meta, hist = load_histogram(args.hist_json)
    traj = meta["trajectories_per_pixel"]
    se_thresh = meta["se_threshold_ev"]
    Emax = meta["energy_max_ev"]
    out_prefix = args.out_prefix or os.path.splitext(args.hist_json)[0]

    # ---- Validation: full-acceptance reconstruction vs the run CSV ----
    se_full = detector_mask(meta, (0.0, se_thresh), (0.0, 90.0))
    bse_full = detector_mask(meta, (se_thresh, Emax + 1.0), (0.0, 90.0))
    se_recon = signal_map(hist, se_full, traj)
    bse_recon = signal_map(hist, bse_full, traj)
    print("Histogram: %dx%d px, %d energy bins x %d angle bins, %d traj/px"
          % (meta["nx"], meta["ny"], meta["energy_bins"], meta["angle_bins"], traj))
    if args.csv and os.path.exists(args.csv):
        rows = list(csv.DictReader(open(args.csv, newline="")))
        nx = meta["nx"]
        se_csv = np.zeros((meta["ny"], nx)); bse_csv = np.zeros((meta["ny"], nx))
        for i, r in enumerate(rows):
            se_csv[i // nx, i % nx] = float(r["SE_yield"])
            bse_csv[i // nx, i % nx] = float(r["BSE_yield"])
        print("VALIDATION (full-acceptance reconstruction vs CSV):")
        print("  SE  max|diff| = %.3e   mean|diff| = %.3e" %
              (np.abs(se_recon - se_csv).max(), np.abs(se_recon - se_csv).mean()))
        print("  BSE max|diff| = %.3e   mean|diff| = %.3e" %
              (np.abs(bse_recon - bse_csv).max(), np.abs(bse_recon - bse_csv).mean()))

    # ---- Detector presets (energy_eV window, beta_deg window) ----
    annular_lo, annular_hi = wd_ring_to_beta(5.0, 3.0, 8.0)   # WD 5 mm, r 3-8 mm
    detectors = [
        ("T3 in-lens SE",  (0.0, se_thresh),       (0.0, 15.0)),
        ("Annular BSE",    (se_thresh, Emax + 1.0), (annular_lo, annular_hi)),
        ("Chamber ETD SE", (0.0, se_thresh),        (30.0, 70.0)),
    ]

    print("\nDetector channels (core r<15nm vs matrix r>60nm):")
    print("  %-16s %-22s %10s %10s %9s" % ("detector", "window", "core", "matrix", "contrast"))
    maps = []
    for name, ew, bw in detectors:
        m = signal_map(hist, detector_mask(meta, ew, bw), traj)
        core, mtx = region_means(m, meta)
        rel = 100.0 * (core - mtx) / mtx if mtx else float("nan")
        win = "E[%g,%g) b[%.0f,%.0f]" % (ew[0], min(ew[1], Emax), bw[0], bw[1])
        print("  %-16s %-22s %10.4f %10.4f %+8.1f%%" % (name, win, core, mtx, rel))
        maps.append((name, m, rel))

    # ---- Figure: detector-channel maps ----
    hw = meta["half_width_nm"]
    ext = [-hw, hw, -hw, hw]
    precip_r = None  # draw a guide circle if center looks like a precipitate scan
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.6))
    for ax, (name, m, rel) in zip(axes, maps):
        im = ax.imshow(m, origin="upper", extent=ext, cmap="inferno")
        ax.add_patch(Circle((meta.get("center_x_nm", 0.0), meta.get("center_y_nm", 0.0)),
                            30.0, fill=False, edgecolor="cyan", lw=1.0, ls="--"))
        ax.set_title("%s\ncore vs matrix: %+.1f%%" % (name, rel), fontsize=11, fontweight="bold")
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("CudaMONSEL Phase-1 detector channels from escape histogram (%s)"
                 % os.path.basename(args.hist_json), fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = out_prefix + "_detectors.png"
    fig.savefig(out_png, dpi=130)
    print("\nwrote", out_png)


if __name__ == "__main__":
    main()
