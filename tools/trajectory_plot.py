"""
Trajectory / interaction-volume / escape-PSF visualizer for CudaMONSEL
single-probe captures.

Consumes the CSVs written by a ``composite_image`` deck with a
``trajectory_capture`` block (CPU backend, single probe over the matrix):

    <base>_traj.csv      full electron paths   -> 2D interaction-volume view (iv2d)
    <base>_escapes.csv   surface-escape events -> 3D escape views (escape3d, psf3d)

Views (``--mode``):
    iv2d      x-z interaction-volume trajectories, colored SE1/SE2/BSE + absorbed
    psf3d     radial areal-density escape PSF surface, log-z, colored by the
              dominant escaping type at each radius (SE1 core, BSE/SE2 skirt)
    escape3d  discrete take-off vectors at each surface exit point (quiver)
    both      iv2d + psf3d  (default)
    all       iv2d + psf3d + escape3d

Common-scale comparison across energies (``--grid``): pass several bases to get
side-by-side montages on shared axes (``*_iv2d_grid.png``, ``*_psf3d_grid.png``).

Geometry: surface at z = 0, material fills z > 0 (plotted downward as depth),
vacuum z < 0. Escape take-off directions are reconstructed from (theta, phi).

Usage:
    python trajectory_plot.py traj_r65g_200V --title "Rene 65 gamma - 200 eV"
    python trajectory_plot.py --grid traj_r65g_200V traj_r65g_1kV traj_r65g_2kV \\
        traj_r65g_4kV --labels "200 eV" "1 keV" "2 keV" "4 keV"
"""
import argparse, csv, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# 2D interaction-volume path styles: key -> (color, label, z-order, lw, alpha)
PATH_STYLE = {
    "BSE":      ("#d62728", "BSE (backscattered primary)", 5, 0.9, 0.9),
    "SE1":      ("#2ca02c", "SE1 (forward-parent secondary)", 4, 0.7, 0.85),
    "SE2":      ("#ff7f0e", "SE2 (backward-parent secondary)", 4, 0.7, 0.85),
    "ABSORBED": ("#9e9e9e", "absorbed primary", 1, 0.4, 0.35),
    "SE":       ("#8c564b", "low-energy primary escape", 3, 0.7, 0.7),
}
ESCAPE_COLOR = {"BSE": "#d62728", "SE1": "#2ca02c", "SE2": "#ff7f0e", "SE": "#8c564b"}

# PSF "dominant type" surface colors (match the reference psf_logz figure).
PSF_COLOR = {"SE1": "#3b6fb0", "SE2": "#ff7f0e", "BSE": "#2e8b3d"}
PSF_LABEL = {"SE1": "SE1-dominant (core)", "SE2": "SE2-dominant", "BSE": "BSE-dominant"}

# gamma matrix / gamma' precipitate phase shading for the 2D cross-section.
MATRIX_TINT = "#e7efe7"   # faint green
PRECIP_FILL = "#b9a7dd"   # light purple
PRECIP_EDGE = "#5b4a9e"
LAYER_FILL  = "#4d4d4d"   # carbon / surface overlayer band (z in [-t, 0])


def resolve_base(arg):
    for suf in ("_traj.csv", "_escapes.csv"):
        if arg.endswith(suf):
            return arg[: -len(suf)]
    if arg.endswith(".csv"):
        return os.path.splitext(arg)[0]
    return arg


def read_ntraj(base, override):
    """Best-effort: read trajectories_per_pixel from the deck <base>.json."""
    if override:
        return override
    cand = base + ".json"
    if os.path.exists(cand):
        try:
            sims = json.load(open(cand)).get("simulations", [])
            if sims:
                n = int(sims[0].get("trajectories_per_pixel", 0))
                return n or None
        except Exception:
            pass
    return None


def read_geometry(base):
    """Best-effort: read the precipitate geometry + phase names from <base>.json."""
    cand = base + ".json"
    if not os.path.exists(cand):
        return None
    try:
        sim = json.load(open(cand))["simulations"][0]
        pr = sim.get("precipitate", {})
        if pr.get("shape") != "sphere":
            return None
        sl = sim.get("surface_layer") or {}
        return {
            "radius": float(pr.get("radius_nm", 0.0)),
            "cx": float(pr.get("center_x_nm", 0.0)),
            "cy": float(pr.get("center_y_nm", 0.0)),
            "cz": float(pr.get("center_depth_nm", 0.0)),
            "precip_name": sim.get("precipitate_phase", {}).get("name", "precipitate"),
            "matrix_name": sim.get("matrix_phase", {}).get("name", "matrix"),
            "layer_t": float(sl.get("thickness_nm", 0.0) or 0.0),
            "layer_name": sl.get("name", "layer"),
        }
    except Exception:
        return None


def read_yields(base):
    """Read the single-probe yields (SE1/SE2/BSE/total) from <base>.csv."""
    p = base + ".csv"
    if not os.path.exists(p):
        return None
    try:
        rows = list(csv.DictReader(open(p, newline="")))
        if not rows:
            return None
        r = rows[0]
        return {k: float(r["%s_yield" % k]) for k in ("SE", "SE1", "SE2", "BSE", "total")}
    except Exception:
        return None


def pretty_energy(tok):
    t = str(tok).strip()
    low = t.lower()
    if low.endswith("kv"):
        return t[:-2] + " keV"
    if low.endswith("v"):
        return t[:-1] + " eV"
    return t


def carbon_nm(tok):
    """'0p6' -> 0.6 (carbon thickness in nm)."""
    try:
        return float(str(tok).replace("p", "."))
    except ValueError:
        return float("nan")


def geom_in_play(geom):
    """True only when the precipitate sits at/near the probe (not parked off-axis)."""
    return geom is not None and geom.get("radius", 0) > 0 and abs(geom.get("cx", 0.0)) <= 200.0


def pretty_phase(name):
    n = (name or "").lower()
    if "gammap" in n or "gamma_prime" in n or "gammaprime" in n or "_gp" in n:
        return "γ′"
    if "gamma" in n:
        return "γ"
    return name


def draw_precip(ax, geom):
    """Shade the material region (matrix tint), a surface layer, and the precipitate."""
    R, cx, cz = geom["radius"], geom["cx"], geom["cz"]
    ax.axhspan(0.0, 1e5, facecolor=MATRIX_TINT, alpha=0.6, zorder=-3, lw=0)
    t = geom.get("layer_t", 0.0)
    if t and t > 0:                       # surface (carbon) overlayer band z in [-t, 0]
        ax.axhspan(-t, 0.0, facecolor=LAYER_FILL, alpha=0.85, zorder=-1.5, lw=0)
    th = np.linspace(0.0, 2 * np.pi, 361)
    X, Z = cx + R * np.cos(th), cz + R * np.sin(th)
    keep = Z >= 0.0                                   # material side of the surface
    if np.any(keep):
        ax.fill(X[keep], Z[keep], facecolor=PRECIP_FILL, alpha=0.55, zorder=-2, lw=0)
        ax.plot(X[keep], Z[keep], color=PRECIP_EDGE, lw=1.3, ls="--", alpha=0.9, zorder=0.4)


# --------------------------------------------------------------------------- #
# 2D interaction volume
# --------------------------------------------------------------------------- #
def path_color_key(elec_type, exit_type):
    if elec_type in ("SE1", "SE2"):
        return elec_type
    return exit_type if exit_type in PATH_STYLE else "ABSORBED"


def load_paths(traj_csv):
    paths = {}
    with open(traj_csv, newline="") as f:
        for r in csv.DictReader(f):
            tid = r["traj_id"]
            p = paths.get(tid)
            if p is None:
                p = paths[tid] = {"x": [], "z": [],
                                  "key": path_color_key(r["elec_type"], r["exit_type"])}
            p["x"].append(float(r["x_nm"]))
            p["z"].append(float(r["z_nm"]))
    return paths


def draw_iv2d(ax, traj_csv, max_se, geom=None, zclip=-0.5):
    """Draw the interaction volume onto ax; return (xabs, zmax, counts, geom_drawn)."""
    geom_drawn = False
    if geom_in_play(geom):
        draw_precip(ax, geom)
        geom_drawn = True
    paths = load_paths(traj_csv)
    prim_keys = ("ABSORBED", "BSE", "SE")
    primaries = [p for p in paths.values() if p["key"] in prim_keys]
    secondaries = [p for p in paths.values() if p["key"] in ("SE1", "SE2")]
    if max_se and len(secondaries) > max_se:
        sel = np.linspace(0, len(secondaries) - 1, max_se).astype(int)
        secondaries = [secondaries[i] for i in sel]
    items = sorted(primaries + secondaries,
                   key=lambda p: PATH_STYLE.get(p["key"], PATH_STYLE["ABSORBED"])[2])

    counts, zmax, xabs = {}, 0.0, 0.5
    for p in items:
        xf = [x for x, z in zip(p["x"], p["z"]) if z >= zclip]
        zf = [z for z in p["z"] if z >= zclip]
        if len(zf) < 2:
            continue
        color, _lbl, zo, lw, al = PATH_STYLE.get(p["key"], PATH_STYLE["ABSORBED"])
        ax.plot(xf, zf, color=color, lw=lw, alpha=al, zorder=zo, solid_capstyle="round")
        counts[p["key"]] = counts.get(p["key"], 0) + 1
        zmax = max(zmax, max(zf))
        xabs = max(xabs, max(abs(v) for v in xf))
    return xabs, (zmax or 1.0), counts, geom_drawn


def geom_legend_handles(geom):
    handles = []
    if geom is None:
        return handles
    if geom.get("layer_t", 0) > 0:
        handles.append(Patch(facecolor=LAYER_FILL, edgecolor="none",
                             label="%s layer" % (geom.get("layer_name") or "surface")))
    handles.append(Patch(facecolor=PRECIP_FILL, edgecolor=PRECIP_EDGE, ls="--",
                         label="%s precipitate" % pretty_phase(geom["precip_name"])))
    handles.append(Patch(facecolor=MATRIX_TINT, edgecolor="#b8c7b8",
                         label="%s matrix" % pretty_phase(geom["matrix_name"])))
    return handles


def iv2d_legend(ax, counts, geom=None, loc="lower right"):
    handles = geom_legend_handles(geom)
    handles += [plt.Line2D([], [], color=PATH_STYLE[k][0], lw=2.4,
                           label="%s  (n=%d)" % (PATH_STYLE[k][1], counts.get(k, 0)))
                for k in ("BSE", "SE1", "SE2", "SE", "ABSORBED") if counts.get(k)]
    ax.legend(handles=handles, loc=loc, fontsize=8, framealpha=0.9)


def style_iv2d_axes(ax, xr, zmax, xlabel=True, ylabel=True, beam=True):
    ax.axhline(0.0, color="k", lw=1.4, zorder=6)
    zt = -0.10 * zmax
    if beam:
        ax.annotate("", xy=(0.0, 0.0), xytext=(0.0, zt),
                    arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.0), zorder=7)
        ax.text(0.0, zt, " beam ", color="#1f77b4", va="bottom", ha="center", fontsize=8)
    ax.set_xlim(-xr, xr)
    ax.set_ylim(zmax * 1.06, -0.16 * zmax)
    if xlabel:
        ax.set_xlabel("lateral x (nm)")
    if ylabel:
        ax.set_ylabel("depth z (nm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.3)


def _frame_with_geom(xr, zmax, geom, gd):
    """Expand the frame so an in-play precipitate silhouette is visible."""
    if gd:
        xr = max(xr, abs(geom["cx"]) + geom["radius"] + 4.0)
        zmax = max(zmax, geom["cz"] + geom["radius"] + 4.0)
    return xr, zmax


def plot_iv2d(traj_csv, out_png, title, max_se, geom=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    xabs, zmax, counts, gd = draw_iv2d(ax, traj_csv, max_se, geom)
    XR, ZM = _frame_with_geom(xabs * 1.1, zmax, geom, gd)
    style_iv2d_axes(ax, XR, ZM)
    iv2d_legend(ax, counts, geom if gd else None)
    ax.set_title(title or "Interaction volume (x-z projection)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("wrote %s  (%s)" % (out_png, ", ".join("%s=%d" % kv for kv in counts.items())))


def plot_iv2d_grid(entries, out_png, suptitle, max_se, xr=None, zmax=None):
    """entries: list of (traj_csv, label, geom). Shared x/z limits across panels."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 5.0), squeeze=False)
    axes = axes[0]
    tmp = []
    geom_any = None
    for (csvp, label, geom), ax in zip(entries, axes):
        xabs, zm, counts, gd = draw_iv2d(ax, csvp, max_se, geom)
        xabs, zm = _frame_with_geom(xabs, zm, geom, gd)   # include precipitate extent
        tmp.append((ax, label, counts, xabs, zm))
        if gd:
            geom_any = geom
    XR = xr or max(t[3] for t in tmp) * 1.1
    ZM = zmax or max(t[4] for t in tmp)
    present = set()
    for ax, label, counts, _xa, _zm in tmp:
        style_iv2d_axes(ax, XR, ZM)
        ax.set_title(label, fontsize=12, fontweight="bold")
        present.update(counts.keys())
    handles = []
    if geom_any is not None:
        handles.append(Patch(facecolor=PRECIP_FILL, edgecolor=PRECIP_EDGE, ls="--",
                             label="%s precipitate" % pretty_phase(geom_any["precip_name"])))
        handles.append(Patch(facecolor=MATRIX_TINT, edgecolor="#b8c7b8",
                             label="%s matrix" % pretty_phase(geom_any["matrix_name"])))
    handles += [plt.Line2D([], [], color=PATH_STYLE[k][0], lw=2.6, label=PATH_STYLE[k][1])
                for k in ("BSE", "SE1", "SE2", "SE", "ABSORBED") if k in present]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.05, right=0.98, wspace=0.22)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("wrote %s  (common scale: x +/-%.1f nm, depth %.1f nm)" % (out_png, XR, ZM))


# --------------------------------------------------------------------------- #
# Escape data + 3D views
# --------------------------------------------------------------------------- #
def load_escapes(esc_csv):
    x, y, th, ph, et, e = [], [], [], [], [], []
    with open(esc_csv, newline="") as f:
        for r in csv.DictReader(f):
            x.append(float(r["x_nm"])); y.append(float(r["y_nm"]))
            th.append(math.radians(float(r["theta_deg"])))
            ph.append(math.radians(float(r["phi_deg"])))
            et.append(r["exit_type"]); e.append(float(r["exit_energy_ev"]))
    return {"x": np.array(x), "y": np.array(y), "th": np.array(th),
            "ph": np.array(ph), "et": np.array(et, dtype=object), "e": np.array(e)}


def plot_escape3d(esc_csv, out_png, title, max_arrows, elev, azim):
    d = load_escapes(esc_csv)
    n = len(d["x"])
    if n == 0:
        print("  (no escape events in %s)" % esc_csv); return
    idx = np.arange(n)
    if max_arrows and n > max_arrows:
        idx = np.linspace(0, n - 1, max_arrows).astype(int)
    th, ph = d["th"][idx], d["ph"][idx]
    ux, uy, uz = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)
    px, py = d["x"][idx], d["y"][idx]
    pz = np.zeros(len(idx))                 # escape points sit on the surface (z = 0)
    rmax = max(3.0, (float(np.max(np.hypot(px, py))) * 1.2) if len(px) else 3.0)
    L = 0.16 * rmax
    fig = plt.figure(figsize=(8.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    gp = np.linspace(-rmax, rmax, 2); gx, gy = np.meshgrid(gp, gp)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="#cccccc", alpha=0.18, linewidth=0)
    counts = {}
    for et in ("BSE", "SE1", "SE2", "SE"):
        m = d["et"][idx] == et
        if not np.any(m):
            continue
        ax.quiver(px[m], py[m], pz[m], ux[m], uy[m], uz[m], length=L, normalize=True,
                  color=ESCAPE_COLOR[et], linewidth=0.7, alpha=0.85)
        counts[et] = int(np.count_nonzero(m))
    ax.scatter([0], [0], [0], color="#1f77b4", s=40, marker="o", depthshade=False)
    ax.quiver(0, 0, L * 2.2, 0, 0, -1, length=L * 2.2, normalize=True, color="#1f77b4", linewidth=2.2)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("height above surface (nm)")
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax); ax.set_zlim(0, 2.5 * L)
    ax.view_init(elev=elev, azim=azim)
    handles = [plt.Line2D([], [], color=ESCAPE_COLOR[k], lw=2.4, label="%s (n=%d)" % (k, counts[k]))
               for k in counts]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_title(title or "Escaping electrons (take-off directions)", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print("wrote %s  (%d/%d arrows)" % (out_png, len(idx), n))


# --------------------------------------------------------------------------- #
# Radial escape-density PSF surface (log-z, dominant-type colored)
# --------------------------------------------------------------------------- #
def radial_psf(esc_csv, rmax, nbins, ntraj):
    """Azimuthally-averaged areal density (1/nm^2 per incident e-) per type."""
    d = load_escapes(esc_csv)
    r = np.hypot(d["x"], d["y"])
    edges = np.linspace(0.0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)     # annulus area (nm^2)
    norm = area * float(ntraj if ntraj else 1)
    cnt_all, _ = np.histogram(r, bins=edges)
    total = cnt_all / norm
    sig = {}
    for t in ("SE1", "SE2", "BSE"):
        c, _ = np.histogram(r[d["et"] == t], bins=edges)
        sig[t] = c / norm
    stack = np.vstack([sig["SE1"], sig["SE2"], sig["BSE"]])
    names = np.array(["SE1", "SE2", "BSE"])
    dom = names[np.argmax(stack, axis=0)]
    return centers, total, dom, sig


def psf_surface(centers, total, dom, rmax, ngrid, floor, logz=True):
    g = np.linspace(-rmax, rmax, ngrid)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    Z = np.interp(R, centers, total, left=(total[0] if len(total) else 0.0), right=0.0)
    if logz:
        Zp = np.log10(np.clip(Z, floor, None))     # log density (compresses magnitude)
        gap = floor * 1.0001
    else:
        Zp = np.clip(Z, 0.0, None)                  # absolute density (linear z)
        gap = 0.0
    dr = rmax / max(len(centers), 1)
    idx = np.clip((R / dr).astype(int), 0, len(centers) - 1)
    domgrid = dom[idx]
    face = np.empty(X.shape + (4,))
    present = set()
    for name, col in PSF_COLOR.items():
        m = domgrid == name
        if np.any(m):
            face[m] = to_rgba(col)
            if np.any(m & (Z > gap)):
                present.add(name)
    beyond = (R > rmax) | (Z <= gap)
    face[beyond] = to_rgba("#dcdcdc", 0.5)
    return X, Y, Zp, face, present


def format_zlog(ax, floor, zmax_log):
    lo = int(np.floor(np.log10(floor)))
    hi = int(np.ceil(zmax_log)) if np.isfinite(zmax_log) else lo + 1
    hi = max(hi, lo + 1)
    ax.set_zlim(lo, hi)
    ticks = list(range(lo, hi + 1))
    ax.set_zticks(ticks)
    ax.set_zticklabels([r"$10^{%d}$" % k for k in ticks])
    ax.set_zlabel("areal density (1/nm$^2$)")


def format_zaxis(ax, floor, zmax, logz=True):
    if logz:
        format_zlog(ax, floor, zmax)
    else:
        ax.set_zlim(0.0, zmax * 1.05 if zmax > 0 else 1.0)
        ax.set_zlabel("areal density (1/nm$^2$)")


def draw_psf3d(ax, esc_csv, rmax, nbins, ntraj, floor, ngrid=141, logz=True):
    centers, total, dom, _sig = radial_psf(esc_csv, rmax, nbins, ntraj)
    X, Y, Zp, face, present = psf_surface(centers, total, dom, rmax, ngrid, floor, logz)
    ax.plot_surface(X, Y, Zp, facecolors=face, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    zmax = float(np.nanmax(Zp)) if Zp.size else (np.log10(floor) if logz else 0.0)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
    return present, zmax


def psf_legend(ax, present, loc="upper left"):
    handles = [plt.Line2D([], [], color=PSF_COLOR[k], lw=6, label=PSF_LABEL[k])
               for k in ("SE1", "SE2", "BSE") if k in present]
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=9)


def plot_psf3d(esc_csv, out_png, title, rmax, nbins, ntraj, floor, elev, azim):
    fig = plt.figure(figsize=(8.6, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    present, zmax_log = draw_psf3d(ax, esc_csv, rmax, nbins, ntraj, floor)
    format_zlog(ax, floor, zmax_log)
    ax.view_init(elev=elev, azim=azim)
    psf_legend(ax, present)
    ax.set_title(title or "Escape-density PSF (log density)", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print("wrote %s  (dominant: %s)" % (out_png, ",".join(sorted(present)) or "none"))


def plot_psf3d_grid(entries, out_png, suptitle, rmax, nbins, floor, elev, azim):
    """entries: list of (esc_csv, label, ntraj). Shared +/-rmax and log-z scale."""
    n = len(entries)
    fig = plt.figure(figsize=(6.2 * n, 6.6))
    panels = []
    for i, (esc, label, ntraj) in enumerate(entries):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        present, zmax_log = draw_psf3d(ax, esc, rmax, nbins, ntraj, floor)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=12, fontweight="bold")
        panels.append((ax, present, zmax_log))
    zmax_log = max(p[2] for p in panels)
    all_present = set().union(*[p[1] for p in panels]) if panels else set()
    for ax, _present, _zm in panels:
        format_zlog(ax, floor, zmax_log)
    if panels:
        psf_legend(panels[0][0], all_present, loc="upper left")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.04, left=0.02, right=0.98, wspace=0.12)
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print("wrote %s  (common +/-%.0f nm, log-z; dominant: %s)"
          % (out_png, rmax, ",".join(sorted(all_present))))


# --------------------------------------------------------------------------- #
# Carbon-sweep matrix (carbon thickness rows x beam-energy columns) + total signal
# --------------------------------------------------------------------------- #
def plot_carbon_matrix(prefix, energies, carbons, out_png, suptitle, max_se):
    """Rows = carbon thickness, cols = beam energy; common x/z scale; each cell is
    a 2D interaction-volume cross-section with the carbon band + precipitate shaded
    and annotated with the T3 total signal (SE1+SE2+BSE)."""
    nrows, ncols = len(carbons), len(energies)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.7 * nrows), squeeze=False)
    # Shared precipitate geometry + phase names come from the per-energy deck
    # (sim 0); each column's carbon thickness comes from its token. The frame is
    # kept at interaction-volume scale so the thin carbon band stays visible.
    base_geom = None
    for etok in energies:
        base_geom = read_geometry("%s_%s" % (prefix, etok))
        if base_geom:
            break
    cells, present, geom_any = [], set(), None
    for ri, ctok in enumerate(carbons):
        for ci, etok in enumerate(energies):
            base = "%s_%s_c%s" % (prefix, etok, ctok)
            ax = axes[ri][ci]
            traj = base + "_traj.csv"
            if not os.path.exists(traj):
                ax.set_axis_off()
                continue
            geom = None
            if base_geom:
                geom = dict(base_geom)
                geom["layer_t"] = carbon_nm(ctok)
                geom["layer_name"] = "carbon"
            xabs, zm, counts, gd = draw_iv2d(ax, traj, max_se, geom)
            cells.append((ax, ri, ci, xabs, zm, read_yields(base)))
            present.update(counts.keys())
            if gd:
                geom_any = geom
    if not cells:
        print("  (no carbon-matrix data for prefix %s)" % prefix)
        plt.close(fig)
        return
    XR = max(c[3] for c in cells) * 1.05
    ZM = max(c[4] for c in cells)
    for ax, ri, ci, _xa, _zm, y in cells:
        style_iv2d_axes(ax, XR, ZM, xlabel=(ri == nrows - 1), ylabel=False)
        if ri == 0:
            ax.set_title(pretty_energy(energies[ci]), fontsize=13, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("C %.1f nm" % carbon_nm(carbons[ri]), fontsize=11, fontweight="bold")
        if y:
            ax.text(0.97, 0.04, "T3 = %.3f" % y["total"], transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9))
    # Compact shared legend (short labels; wraps to keep within the figure width).
    handles = []
    if geom_any:
        if geom_any.get("layer_t", 0) > 0:
            handles.append(Patch(facecolor=LAYER_FILL, label="carbon"))
        handles.append(Patch(facecolor=PRECIP_FILL, edgecolor=PRECIP_EDGE, ls="--",
                             label="%s precip." % pretty_phase(geom_any["precip_name"])))
        handles.append(Patch(facecolor=MATRIX_TINT, edgecolor="#b8c7b8",
                             label="%s matrix" % pretty_phase(geom_any["matrix_name"])))
    short = {"BSE": "BSE", "SE1": "SE1", "SE2": "SE2", "SE": "PE→SE", "ABSORBED": "absorbed PE"}
    handles += [plt.Line2D([], [], color=PATH_STYLE[k][0], lw=2.6, label=short[k])
                for k in ("BSE", "SE1", "SE2", "SE", "ABSORBED") if k in present]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.999)
    fig.subplots_adjust(top=0.925, bottom=0.05, left=0.07, right=0.99, wspace=0.18, hspace=0.18)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("wrote %s  (%dx%d, common x +/-%.1f nm, depth %.1f nm)"
          % (out_png, nrows, ncols, XR, ZM))


def plot_total_signal(prefix, energies, carbons, out_png, title):
    """T3 total signal (SE1+SE2+BSE, full collection) vs carbon thickness, per energy."""
    xs = [carbon_nm(c) for c in carbons]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, etok in enumerate(energies):
        col = colors[i % len(colors)]
        tot, se, bse = [], [], []
        for ctok in carbons:
            y = read_yields("%s_%s_c%s" % (prefix, etok, ctok))
            tot.append(y["total"] if y else float("nan"))
            se.append(y["SE"] if y else float("nan"))
            bse.append(y["BSE"] if y else float("nan"))
        ax.plot(xs, tot, "-o", color=col, lw=2.4, label="%s  total (SE1+SE2+BSE)" % pretty_energy(etok))
        ax.plot(xs, se, "--", color=col, lw=1.3, alpha=0.7, label="%s  SE" % pretty_energy(etok))
        ax.plot(xs, bse, ":", color=col, lw=1.3, alpha=0.7, label="%s  BSE" % pretty_energy(etok))
    ax.set_xlabel("carbon overlayer thickness (nm)")
    ax.set_ylabel("signal per incident electron")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=len(energies))
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("wrote %s" % out_png)


# --------------------------------------------------------------------------- #
# Total-signal image (raster) reading + inversion demo + PSF-escape matrix
# --------------------------------------------------------------------------- #
def read_image(base):
    """Read a composite_image raster CSV into SE/SE1/SE2/BSE/total 2D grids."""
    p = base + ".csv"
    if not os.path.exists(p):
        return None
    xs, ys, cols = [], [], {"SE": [], "SE1": [], "SE2": [], "BSE": [], "total": []}
    det_cols = {}
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        det_names = [n for n in (rd.fieldnames or []) if n.startswith("det_")]
        det_cols = {n: [] for n in det_names}
        for r in rd:
            xs.append(float(r["x_nm"])); ys.append(float(r["y_nm"]))
            for k in cols:
                cols[k].append(float(r["%s_yield" % k]))
            for n in det_names:
                det_cols[n].append(float(r[n]))
    if not xs:
        return None
    ux, uy = sorted(set(xs)), sorted(set(ys))
    nx, ny = len(ux), len(uy)
    if nx * ny != len(xs):
        return None
    hw = max(abs(min(ux)), abs(max(ux)), abs(min(uy)), abs(max(uy)))
    grids = {k: np.array(v).reshape(ny, nx) for k, v in cols.items()}
    grids.update({k: np.array(v).reshape(ny, nx) for k, v in det_cols.items()})
    return {"grids": grids, "hw": hw, "nx": nx, "ny": ny}


def region_contrast(grid, hw, cx, cy, core_r, mtx_r):
    """Precip-core vs matrix mean and percent contrast for one yield grid."""
    ny, nx = grid.shape
    xs = np.linspace(-hw, hw, nx)
    ys = np.linspace(hw, -hw, ny)                 # row 0 = +hw (top)
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X - cx, Y - cy)
    core = grid[R < core_r]
    mtx = grid[R > mtx_r]
    cm, mm = float(core.mean()), float(mtx.mean())
    return cm, mm, (100.0 * (cm - mm) / mm if mm else float("nan"))


def plot_inversion_demo(prefix, etok, carbons, out_png, geom, core_r=20.0, mtx_r=45.0):
    """image_denseP-style figure: 2x3 total-signal image grid + precip-vs-matrix
    contrast (SE / BSE / total) vs carbon, with any bright<->dark inversion marked."""
    imgs = [read_image("%s_%s_c%s" % (prefix, etok, c)) for c in carbons]
    if not any(imgs):
        print("  (no image data for %s_%s)" % (prefix, etok)); return
    cx = geom["cx"] if geom else 0.0
    cy = geom.get("cy", 0.0) if geom else 0.0
    R = geom["radius"] if geom else 30.0
    xs = [carbon_nm(c) for c in carbons]
    se_c, bse_c, tot_c = [], [], []
    for im in imgs:
        if im is None:
            se_c.append(np.nan); bse_c.append(np.nan); tot_c.append(np.nan); continue
        se_c.append(region_contrast(im["grids"]["SE"], im["hw"], cx, cy, core_r, mtx_r)[2])
        bse_c.append(region_contrast(im["grids"]["BSE"], im["hw"], cx, cy, core_r, mtx_r)[2])
        tot_c.append(region_contrast(im["grids"]["total"], im["hw"], cx, cy, core_r, mtx_r)[2])

    from matplotlib.patches import Circle
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.55], wspace=0.28, hspace=0.28)
    # Left: 2x3 image grid, common grayscale per figure.
    ncol_img = 3
    vmax = max((np.nanpercentile(im["grids"]["total"], 99) for im in imgs if im), default=1.0)
    for i, (c, im) in enumerate(zip(carbons, imgs)):
        ax = fig.add_subplot(gs[i // ncol_img, i % ncol_img])
        if im is None:
            ax.set_axis_off(); continue
        hw = im["hw"]
        ax.imshow(im["grids"]["total"], origin="upper", extent=[-hw, hw, -hw, hw],
                  cmap="gray", vmin=0.0, vmax=vmax)
        ax.add_patch(Circle((cx, cy), R, fill=False, ec="cyan", ls="--", lw=1.2))
        tag = "BRIGHT" if tot_c[i] > 1.0 else ("DARK" if tot_c[i] < -1.0 else "~flat")
        ax.set_title("C %.1f nm | total %+.1f%% (%s)" % (carbon_nm(c), tot_c[i], tag), fontsize=10)
        ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50]); ax.tick_params(labelsize=8)

    # Right: contrast vs carbon. Fix the y-range first so the shaded bands and
    # the inversion label stay in bounds.
    axc = fig.add_subplot(gs[:, 3])
    allc = np.abs(np.array(se_c + bse_c + tot_c, dtype=float))
    ymax = float(np.nanmax(allc)) * 1.15 if np.any(np.isfinite(allc)) else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    axc.set_ylim(-ymax, ymax)
    axc.axhspan(0, ymax, color="#e8f5e9", zorder=0)
    axc.axhspan(-ymax, 0, color="#fde8e8", zorder=0)
    axc.axhline(0, color="0.4", lw=1)
    axc.plot(xs, se_c, "-o", color="#e08a3c", lw=2, label="SE")
    axc.plot(xs, bse_c, "-s", color="#2e8b3d", lw=2, label="BSE (>50 eV)")
    axc.plot(xs, tot_c, "-^", color="black", lw=2.4, label="Combined / total (T3)")
    # Mark the bright->dark inversion (first zero crossing of total).
    for i in range(1, len(xs)):
        a, b = tot_c[i - 1], tot_c[i]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            xc = xs[i - 1] + (xs[i] - xs[i - 1]) * a / (a - b)
            axc.axvline(xc, color="purple", ls="--", lw=1.6)
            axc.text(xc, ymax * 0.62, "  INVERSION\n  bright γ′ → dark γ′\n  %.2f nm" % xc,
                     color="purple", fontsize=10, fontweight="bold", va="top")
            break
    axc.text(0.98, 0.96, "γ′ BRIGHT", color="#2e8b3d", fontsize=11, ha="right",
             va="top", transform=axc.transAxes)
    axc.text(0.98, 0.04, "γ′ DARK", color="#c0392b", fontsize=11, ha="right",
             va="bottom", transform=axc.transAxes)
    axc.set_xlabel("carbon thickness (nm)")
    axc.set_ylabel("precip core vs matrix contrast (%)")
    axc.set_title("SE bright-γ′ + BSE dark-γ′ → combined T3 inverts as carbon strips SE", fontsize=11)
    axc.legend(loc="upper center", fontsize=9)
    axc.grid(True, ls=":", alpha=0.4)
    fig.suptitle("René 65 γ′-in-γ inversion demo @ %s — T3 total signal vs carbon" % pretty_energy(etok),
                 fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s  (total contrast %+.1f%% -> %+.1f%%)" % (out_png, tot_c[0], tot_c[-1]))


def first_zero_crossing(xs, ys):
    """Linear-interpolated x of the first sign change in ys, or None."""
    for i in range(1, len(xs)):
        a, b = ys[i - 1], ys[i]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * a / (a - b)
    return None


def plot_detector_inversion(prefix, etok, carbons, out_png, geom, channels=None,
                            core_r=20.0, mtx_r=45.0, images=False):
    """Detector-channel contrast-vs-carbon curves (one per det_* CSV column,
    ideal total_yield as black reference); each channel's bright->dark zero
    crossing is annotated in the legend and marked. With images=True, also
    writes a channel(row) x carbon(col) image-grid figure."""
    imgs = [read_image("%s_%s_c%s" % (prefix, etok, c)) for c in carbons]
    if not any(imgs):
        print("  (no image data for %s_%s)" % (prefix, etok)); return
    cx = geom["cx"] if geom else 0.0
    cy = geom.get("cy", 0.0) if geom else 0.0
    R = geom["radius"] if geom else 30.0
    first = next(im for im in imgs if im)
    found = [k for k in first["grids"] if k.startswith("det_")]
    if channels:
        want = [c if c.startswith("det_") else "det_" + c for c in channels]
        chans = [c for c in want if c in found]
    else:
        chans = found
    if not chans:
        print("  (no det_* channels in %s_%s CSVs)" % (prefix, etok)); return
    xs = [carbon_nm(c) for c in carbons]

    def contrast_series(key):
        out = []
        for im in imgs:
            if im is None or key not in im["grids"]:
                out.append(np.nan); continue
            out.append(region_contrast(im["grids"][key], im["hw"], cx, cy, core_r, mtx_r)[2])
        return out

    palette = ["#e08a3c", "#2e8b3d", "#7b52ab", "#c0392b", "#2471a3", "#7f8c52"]
    fig, axc = plt.subplots(figsize=(9.5, 7))
    tot_c = contrast_series("total")
    allvals = list(tot_c)
    curves = []
    for i, ch in enumerate(chans):
        yc = contrast_series(ch)
        allvals += yc
        curves.append((ch, yc, palette[i % len(palette)]))
    ymax = float(np.nanmax(np.abs(np.array(allvals, dtype=float)))) * 1.15
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    axc.set_ylim(-ymax, ymax)
    axc.axhspan(0, ymax, color="#e8f5e9", zorder=0)
    axc.axhspan(-ymax, 0, color="#fde8e8", zorder=0)
    axc.axhline(0, color="0.4", lw=1)
    xc = first_zero_crossing(xs, tot_c)
    lab = "total_yield (ideal)" + (" | inv %.2f nm" % xc if xc is not None else "")
    axc.plot(xs, tot_c, "-^", color="black", lw=2.4, label=lab)
    markers = "osdvP*"
    for i, (ch, yc, col) in enumerate(curves):
        xcc = first_zero_crossing(xs, yc)
        lab = ch[4:] + (" | inv %.2f nm" % xcc if xcc is not None else "")
        axc.plot(xs, yc, "-" + markers[i % len(markers)], color=col, lw=2, label=lab)
        if xcc is not None:
            axc.axvline(xcc, color=col, ls="--", lw=1.1, alpha=0.65)
    axc.text(0.98, 0.96, "γ′ BRIGHT", color="#2e8b3d", fontsize=11, ha="right",
             va="top", transform=axc.transAxes)
    axc.text(0.98, 0.04, "γ′ DARK", color="#c0392b", fontsize=11, ha="right",
             va="bottom", transform=axc.transAxes)
    axc.set_xlabel("carbon thickness (nm)")
    axc.set_ylabel("precip core vs matrix contrast (%)")
    axc.set_title("Detector-channel γ′ contrast vs carbon @ %s\n"
                  "(energy/angle windows, β = π − θ; 0° = up the column)"
                  % pretty_energy(etok), fontsize=12)
    axc.legend(loc="upper right", fontsize=9)
    axc.grid(True, ls=":", alpha=0.4)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    ends = ", ".join("%s %+.1f%%" % (ch[4:], yc[-1]) for ch, yc, _ in curves)
    print("wrote %s  (end contrast: %s)" % (out_png, ends))

    if images:
        from matplotlib.patches import Circle
        nrows, ncols = len(chans), len(carbons)
        fig = plt.figure(figsize=(2.6 * ncols, 2.8 * nrows))
        gs = fig.add_gridspec(nrows, ncols, wspace=0.08, hspace=0.22)
        for ri, ch in enumerate(chans):
            gmax = max((np.nanpercentile(im["grids"][ch], 99)
                        for im in imgs if im and ch in im["grids"]), default=1.0)
            for ci, (c, im) in enumerate(zip(carbons, imgs)):
                ax = fig.add_subplot(gs[ri, ci])
                if im is None or ch not in im["grids"]:
                    ax.set_axis_off(); continue
                hw = im["hw"]
                ax.imshow(im["grids"][ch], origin="upper", extent=[-hw, hw, -hw, hw],
                          cmap="gray", vmin=0.0, vmax=gmax)
                ax.add_patch(Circle((cx, cy), R, fill=False, ec="cyan", ls="--", lw=0.9))
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title("C %.1f nm" % carbon_nm(c), fontsize=9)
                if ci == 0:
                    ax.set_ylabel(ch[4:], fontsize=9)
        fig.suptitle("Detector-channel images @ %s (rows = channel, cols = carbon)"
                     % pretty_energy(etok), fontsize=12, fontweight="bold")
        img_png = out_png.replace(".png", "_images.png")
        fig.savefig(img_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote %s" % img_png)


def plot_psf3d_matrix(prefix, energies, carbons, out_png, suptitle, rmax, nbins, floor,
                      traj, elev, azim, logz=True):
    """Grid of escape-density PSF surfaces: rows = energy, cols = carbon thickness.
    logz=True: common log-z (shape); logz=False: absolute density, linear z scaled
    per energy row so the signal decrease with carbon is directly visible."""
    nrows, ncols = len(energies), len(carbons)
    fig = plt.figure(figsize=(3.3 * ncols, 3.3 * nrows))
    rows, all_present = [], set()
    for ri, etok in enumerate(energies):
        ntraj_e = read_ntraj("%s_%s" % (prefix, etok), traj)
        row_panels = []
        for ci, ctok in enumerate(carbons):
            base = "%s_%s_c%s" % (prefix, etok, ctok)
            esc = base + "_escapes.csv"
            ax = fig.add_subplot(nrows, ncols, ri * ncols + ci + 1, projection="3d")
            if not os.path.exists(esc):
                ax.set_axis_off(); row_panels.append(None); continue
            present, zmax = draw_psf3d(ax, esc, rmax, nbins, ntraj_e, floor, logz=logz)
            ax.view_init(elev=elev, azim=azim)
            if ri == 0:
                ax.set_title("C %.1f nm" % carbon_nm(ctok), fontsize=11, fontweight="bold")
            if ci == 0:
                ax.text2D(-0.15, 0.5, pretty_energy(etok), transform=ax.transAxes,
                          fontsize=12, fontweight="bold", rotation=90, va="center")
            row_panels.append((ax, zmax)); all_present |= present
        rows.append(row_panels)
    valid = [p for row in rows for p in row if p]
    if not valid:
        print("  (no escape data for psf3d matrix)"); plt.close(fig); return
    if logz:                                  # shared log-z across the whole grid
        zc = max(p[1] for p in valid)
        for p in valid:
            format_zaxis(p[0], floor, zc, logz=True)
    else:                                     # linear: per-energy-row peak (0-carbon)
        for row_panels in rows:
            rp = [p for p in row_panels if p]
            if not rp:
                continue
            zc = max(p[1] for p in rp)
            for p in rp:
                format_zaxis(p[0], floor, zc, logz=False)
    psf_legend(valid[0][0], all_present, loc="upper left")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.99, wspace=0.05, hspace=0.12)
    fig.savefig(out_png, dpi=135)
    plt.close(fig)
    print("wrote %s  (%dx%d PSF grid, %s-z)" % (out_png, nrows, ncols, "log" if logz else "linear"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bases", nargs="*", help="capture base name(s) or *_traj.csv / *_escapes.csv")
    ap.add_argument("--mode", choices=["iv2d", "escape3d", "psf3d", "both", "all"], default="both")
    ap.add_argument("--grid", action="store_true", help="montage all bases on a common scale")
    ap.add_argument("--carbon-matrix", action="store_true",
                    help="carbon(row) x energy(col) 2D matrix + total-signal plot from <prefix>_<energy>_c<t>")
    ap.add_argument("--prefix", default="traj_gpc", help="base prefix for --carbon-matrix")
    ap.add_argument("--energies", nargs="+", default=["200V", "1kV"], help="energy tokens (columns)")
    ap.add_argument("--carbons", nargs="+", default=["0p0", "0p2", "0p4", "0p6", "0p8", "1p0"],
                    help="carbon-thickness tokens (rows), e.g. 0p2 == 0.2 nm")
    ap.add_argument("--inversion", action="store_true",
                    help="image_denseP-style total-signal inversion demo per energy (needs GPU image CSVs)")
    ap.add_argument("--img-prefix", default="image_gpc", help="raster-image base prefix for --inversion")
    ap.add_argument("--det-inversion", action="store_true",
                    help="detector-channel contrast-vs-carbon curves from det_* CSV columns")
    ap.add_argument("--channels", nargs="+", default=None,
                    help="det_* channel names for --det-inversion (default: all found)")
    ap.add_argument("--det-images", action="store_true",
                    help="also write the channel x carbon image grid for --det-inversion")
    ap.add_argument("--labels", nargs="*", default=None, help="per-base titles (grid or single)")
    ap.add_argument("--title", default=None, help="single-figure title / grid suptitle")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--max-se", type=int, default=800, help="cap SE secondary paths drawn (2D)")
    ap.add_argument("--max-arrows", type=int, default=2500, help="cap escape arrows (escape3d)")
    ap.add_argument("--rmax", type=float, default=30.0, help="PSF radial half-extent (nm)")
    ap.add_argument("--nbins", type=int, default=60, help="PSF radial bins")
    ap.add_argument("--floor", type=float, default=1e-4, help="PSF log-density floor (1/nm^2)")
    ap.add_argument("--psf-scale", choices=["log", "linear", "both"], default="log",
                    help="PSF matrix z-axis: log (shape) / linear (absolute magnitude) / both")
    ap.add_argument("--traj", type=int, default=None, help="override trajectories/probe for PSF norm")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    args = ap.parse_args()

    if args.carbon_matrix or args.inversion or args.det_inversion:
        if args.carbon_matrix:
            out = args.out_prefix or args.prefix
            plot_carbon_matrix(args.prefix, args.energies, args.carbons, out + "_matrix.png",
                               args.title or "René 65 γ′ half-sphere: carbon effect on escape trajectories",
                               args.max_se)
            plot_total_signal(args.prefix, args.energies, args.carbons, out + "_total_signal.png",
                              "T3 total signal (SE1+SE2+BSE, immersion + beam deceleration)\n"
                              "René 65 γ′ half-sphere vs carbon overlayer")
            scales = {"log": [True], "linear": [False], "both": [True, False]}[args.psf_scale]
            for lz in scales:
                sfx = "" if lz else "_linear"
                scale_txt = ("log-z, common ±%.0f nm" % args.rmax if lz
                             else "absolute density, linear z per-energy row")
                plot_psf3d_matrix(args.prefix, args.energies, args.carbons,
                                  out + "_psf3d_matrix" + sfx + ".png",
                                  "René 65 γ′ escape-density PSF vs carbon (%s; "
                                  "rows = energy, cols = carbon)" % scale_txt,
                                  args.rmax, args.nbins, args.floor, args.traj,
                                  args.elev, args.azim, logz=lz)
        if args.inversion:
            for etok in args.energies:
                geom = read_geometry("%s_%s" % (args.img_prefix, etok))
                plot_inversion_demo(args.img_prefix, etok, args.carbons,
                                    "%s_%s_inversion.png" % (args.img_prefix, etok), geom)
        if args.det_inversion:
            for etok in args.energies:
                geom = read_geometry("%s_%s" % (args.img_prefix, etok))
                plot_detector_inversion(args.img_prefix, etok, args.carbons,
                                        "%s_%s_det_inversion.png" % (args.img_prefix, etok),
                                        geom, channels=args.channels, images=args.det_images)
        return

    if not args.bases:
        ap.error("provide capture base name(s), or use --grid / --carbon-matrix")
    bases = [resolve_base(b) for b in args.bases]
    labels = args.labels or [os.path.basename(b) for b in bases]

    if args.grid:
        prefix = args.out_prefix or (os.path.commonprefix(bases).rstrip("_") or "traj") + "_grid"
        if args.mode in ("iv2d", "both", "all"):
            entries = [(b + "_traj.csv", lab, read_geometry(b)) for b, lab in zip(bases, labels)
                       if os.path.exists(b + "_traj.csv")]
            if entries:
                plot_iv2d_grid(entries, prefix.replace("_grid", "") + "_iv2d_grid.png",
                               args.title or "Interaction volume vs beam energy (common scale)",
                               args.max_se)
        if args.mode in ("psf3d", "both", "all"):
            entries = [(b + "_escapes.csv", lab, read_ntraj(b, args.traj))
                       for b, lab in zip(bases, labels) if os.path.exists(b + "_escapes.csv")]
            if entries:
                plot_psf3d_grid(entries, prefix.replace("_grid", "") + "_psf3d_grid.png",
                                args.title or "Total electron-escape PSF colored by dominant type "
                                "vs beam energy (log density, common +/-%.0f nm)" % args.rmax,
                                args.rmax, args.nbins, args.floor, args.elev, args.azim)
        return

    for base, label in zip(bases, labels):
        title = args.title if (len(bases) == 1 and args.title) else label
        out = args.out_prefix or base
        traj_csv, esc_csv = base + "_traj.csv", base + "_escapes.csv"
        if args.mode in ("iv2d", "both", "all"):
            if os.path.exists(traj_csv):
                plot_iv2d(traj_csv, out + "_iv2d.png", title, args.max_se, read_geometry(base))
            else:
                print("  missing %s" % traj_csv)
        if args.mode in ("psf3d", "both", "all"):
            if os.path.exists(esc_csv):
                plot_psf3d(esc_csv, out + "_psf3d.png", title, args.rmax, args.nbins,
                           read_ntraj(base, args.traj), args.floor, args.elev, args.azim)
            else:
                print("  missing %s" % esc_csv)
        if args.mode in ("escape3d", "all"):
            if os.path.exists(esc_csv):
                plot_escape3d(esc_csv, out + "_escape3d.png", title, args.max_arrows, args.elev, args.azim)
            else:
                print("  missing %s" % esc_csv)


if __name__ == "__main__":
    main()
