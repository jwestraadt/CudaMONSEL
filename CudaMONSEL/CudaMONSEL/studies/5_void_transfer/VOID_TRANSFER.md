# Recreation of Payton & Mills (2011) Fig. 5 — BSE images of spherical voids

Reproduces *Materials Characterization* **62** (2011) 563–574, Fig. 5: BSE-
coefficient images of spherical **voids** (etched-out γ′) in a René 104 matrix,
across void radius R, centroid depth x, and accelerating voltage — using the
paper's efficient **1-D radial η(r) + 360° revolve** method (not a full 2-D
raster, which would be ~180× more expensive).

## Method
For each (kV, R, depth): scan the beam along a line from the void-center
projection out to 4R, compute η(r); then revolve 360°, linearly interpolate,
extend with the bulk η beyond 4R, and add uniform ±3σ noise (σ = 0.008). Lay out
as four blocks — analytic **Planar Intersection** (footprint radius √(R²−d²)) +
**5 / 12 / 20 kV** — rows = centroid depth (+0.8R…−1.0R), cols = R (25…275 nm),
on a common spatial + grayscale scale.

## Parameters
| | |
|---|---|
| Matrix | René 104 (at%): Ni 49.3, Co 20.1, Cr 14.5, Al 7.2, Ti 4.7, Mo 2.3, Ta 0.8, W 0.65; ρ 8250 kg/m³ |
| Voids | vacuum spheres; R = 25/75/125/175/225/275 nm |
| Depths x | +0.8R → −1.0R in 0.2R steps (x = centroid height above surface ⇒ `center_depth_nm = −x`) |
| Beam | Gaussian, FWHM 3.2 nm (σ = 1.36 nm), convergence 0; 5/12/20 kV |
| Sampling | 306 samples/µm along [0, 4R]; 5000 traj/sample (η stat. noise ≪ ±0.024 render noise) |
| η | BSE (>50 eV) yield; **BSE-only** (`track_secondaries: false`) — inelastic events don't touch the primary, so η is unchanged and ~SE overhead is skipped |

## Enabling features added to `composite_image`
- **`precipitate.void: true`** — precipitate becomes a true vacuum sphere (no scatter inside) on both backends; escape logic untouched.
- **`scan.radial: true` + `radial_max_nm`** — 1-D radial line scan: `nx` samples at (r, 0), r ∈ [0, radial_max].
- **`track_secondaries: false`** — BSE-only fast path (GPU): skips SE generation/tracking; η identical.

## Regenerate
```
python studies/5_void_transfer/gen_void_decks.py                     # writes void_{5,12,20}kV.json
..\..\x64\Release\CudaMONSEL.exe studies/5_void_transfer/void_5kV.json   # (and 12kV, 20kV)
python studies/5_void_transfer/void_transfer_plot.py                 # -> void_transfer_fig5.png
```

## What it shows (matches the paper's stereology)
- **Above-surface** voids (+x): shallow craters with **bright rims**, appearing *smaller* than their planar footprint (under-segmentation).
- **At / just below** surface: dark disk with a bright rim.
- **Sub-surface** voids (−x): dark cores that **over-project** — appearing *larger* than the true planar intersection — then fade toward −1.0R.
- **Higher kV**: larger interaction volume → more **delocalized**, lower-contrast, blurrier features (5 → 12 → 20 kV).

These are exactly the observation biases the paper's atanh transfer function corrects; small voids (≲ 32–38 nm) are unresolved.
