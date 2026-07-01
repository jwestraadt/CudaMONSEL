"""
Recreate Payton & Mills (2011) Fig. 5 from the CudaMONSEL void radial-eta scans.

For each (kV, R, depth) the 1-D eta(r) profile is revolved 360 deg into a 2-D
image (linear interpolation, extended with the bulk eta beyond 4R), plus uniform
+/-3 sigma noise (sigma = 0.008). Laid out as four blocks -- Planar Intersection
(analytic footprint) + 5 / 12 / 20 kV -- rows = centroid depth (+0.8R .. -1.0R),
cols = R (25 .. 275 nm), on a common spatial + grayscale scale.

    python studies/5_void_transfer/void_transfer_plot.py     # from the project dir
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RADII = [25, 75, 125, 175, 225, 275]
XFRACS = [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0]
KVS = [5, 12, 20]
H_NM = 340.0            # cell half-extent (common spatial scale)
NGRID = 130            # pixels per cell
SIGMA = 0.008
NOISE = 3 * SIGMA      # +/- range of the uniform noise
SEED = 12345


def xtag(xf):
    return ("p" if xf >= 0 else "m") + ("%.1f" % abs(xf)).replace(".", "")


def read_profile(kv, R, xf):
    p = os.path.join(DATA, "void_%dkV_R%d_x%s.csv" % (kv, R, xtag(xf)))
    if not os.path.exists(p):
        return None
    r, eta = [], []
    for row in csv.DictReader(open(p, newline="")):
        r.append(float(row["x_nm"])); eta.append(float(row["BSE_yield"]))
    if len(r) < 2:
        return None
    r = np.array(r); eta = np.array(eta)
    o = np.argsort(r)
    return r[o], eta[o]


def revolve(prof, rng):
    r, eta = prof
    bulk = float(np.median(eta[r > 0.5 * r.max()])) if r.max() > 0 else float(eta.mean())
    g = np.linspace(-H_NM, H_NM, NGRID)
    X, Y = np.meshgrid(g, g)
    RR = np.hypot(X, Y)
    img = np.interp(RR, r, eta, left=eta[0], right=bulk)
    img = img + rng.uniform(-NOISE, NOISE, img.shape)
    return img, bulk


def planar_disk(R, xf):
    # footprint where the void intersects the plane of polish: radius sqrt(R^2 - d^2)
    d = -xf * R                                   # centroid depth below surface
    rf = math.sqrt(max(0.0, R * R - d * d))
    g = np.linspace(-H_NM, H_NM, NGRID)
    X, Y = np.meshgrid(g, g)
    return (np.hypot(X, Y) <= rf).astype(float)   # 1 inside footprint (drawn black)


def main():
    rng = np.random.default_rng(SEED)
    # Load all sim images; collect a common grayscale range.
    sims = {}   # (kv,R,xf) -> img
    vals = []
    for kv in KVS:
        for R in RADII:
            for xf in XFRACS:
                prof = read_profile(kv, R, xf)
                if prof is None:
                    continue
                img, _ = revolve(prof, rng)
                sims[(kv, R, xf)] = img
                vals.append(img)
    if not vals:
        print("no void data found in", DATA); return
    # Paper-matched grayscale: bulk eta -> mid-gray, wide range so voids read as
    # gentle deviations (not black/white). Most pixels are background, so the
    # median approximates bulk eta.
    allv = np.concatenate([v.ravel() for v in vals])
    bulk = float(np.median(allv))
    vmin, vmax = 0.0, 2.0 * bulk

    nrow, ncol = len(XFRACS), len(RADII)
    blocks = ["Planar Intersection", "5 kV Simulation", "12 kV Simulation", "20 kV Simulation"]
    # figure geometry (figure-fraction placement with gaps between blocks)
    left, top = 0.055, 0.87
    cw, ch, bgap = 0.0355, 0.079, 0.018
    fig = plt.figure(figsize=(19.5, 9.2))

    for bi, title in enumerate(blocks):
        x0 = left + bi * (ncol * cw + bgap)
        for ri, xf in enumerate(XFRACS):
            for ci, R in enumerate(RADII):
                ax = fig.add_axes([x0 + ci * cw, top - (ri + 1) * ch, cw * 0.94, ch * 0.94])
                if bi == 0:
                    ax.imshow(planar_disk(R, xf), cmap="gray_r", vmin=0, vmax=1,
                              extent=[-H_NM, H_NM, -H_NM, H_NM], origin="upper")
                else:
                    kv = [5, 12, 20][bi - 1]
                    img = sims.get((kv, R, xf))
                    if img is None:
                        ax.set_facecolor("0.5")
                    else:
                        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                                  extent=[-H_NM, H_NM, -H_NM, H_NM], origin="upper")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if bi == 0 and ci == 0:
                    ax.set_ylabel("%+.1fR" % xf, rotation=0, ha="right", va="center", fontsize=9)
                if ri == 0:
                    ax.set_title("%d" % R, fontsize=8, pad=2)
        # block title + a single "R (nm)" hint under it
        fig.text(x0 + ncol * cw / 2, top + 0.045, title, ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
        fig.text(x0 + ncol * cw / 2, top + 0.028, "R (nm)", ha="center", va="bottom",
                 fontsize=8, color="0.35")

    fig.text(left - 0.028, top - nrow * ch / 2, "centroid depth x", rotation=90,
             ha="center", va="center", fontsize=10)
    fig.suptitle("Recreation of Payton & Mills (2011) Fig. 5 — BSE-coefficient images of "
                 "spherical voids (Rene 104), FWHM 3.2 nm", fontsize=13, fontweight="bold", y=0.995)
    out = os.path.join(HERE, "void_transfer_fig5.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote %s  (grayscale eta in [%.3f, %.3f]; %d/%d sim cells)"
          % (out, vmin, vmax, len(sims), 3 * nrow * ncol))


if __name__ == "__main__":
    main()
