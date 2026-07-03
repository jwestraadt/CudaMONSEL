# SE Yield Simulations in CudaMONSEL — Summary

CudaMONSEL is a Monte Carlo electron-trajectory simulator (a CUDA port of NIST's JMONSEL) that computes **secondary electron (SE) and backscatter (BSE) yields** for SEM imaging. Electrons are launched into a material, scattered until they thermalize or escape, and the yield is the number of detected electrons per incident electron.

## The three simulation drivers

| Driver | What it does |
|--------|-------------|
| `BulkYield.cu` | SE/BSE yield vs. beam energy for homogeneous bulk materials |
| `CompositeYield.cu` | Yield for a matrix with an embedded spherical precipitate |
| `CompositeImage.cu` (+ GPU backend) | Raster-scanned 2D yield maps, split into SE1/SE2/BSE |

All live under `CudaMONSEL/CudaMONSEL/gov/nist/nanoscalemetrology/JMONSELTests/`.

## How yield is computed

Each trajectory chains three physics models (`MONSEL_MaterialScatterModel`):

1. **Elastic scattering** — NIST Mott cross-sections (`SelectableElasticSM`) set the direction changes.
2. **Inelastic scattering / SE creation** — `FittedInelSM` generates secondaries; new SE energy = `energySEgen + eFermi`.
3. **Continuous slowing-down (CSD)** — `JoyLuoNieminenCSD` bleeds energy along the path (Nieminen below `breakE`, Joy/Luo above).

Escape is gated by the **surface barrier** (`ExpQMBarrierSM`): only electrons whose **perpendicular** energy exceeds the barrier step (`cos²α·KE > ΔU`) transmit; the rest reflect. This is the single biggest lever on SE yield, since SEs are low-energy. Escaped electrons are tallied in `BackscatterStats`: those below the `seThreshold` (~50 eV) count as SE, above as BSE. SE1 (true secondaries, forward) vs. SE2 (BSE-induced, backward) are split by `cos(θ) > 0`.

## Key parameters driving the yield

### Material properties (the dominant knobs)

- **Work function** (~5.15 eV Ni) and **Fermi energy / conduction-band bottom** (`energyCBbottom`, ~−8.8 eV) — together set the escape barrier height. Lower barrier → higher SE yield.
- **`energySEgen`** — average energy to create one SE (~30 eV metals, ~65 eV organics). Lower value → more SEs generated per unit energy loss.
- **Bandgap** (0 for metals) and **density** — affect generation and stopping power.
- **Composition** (Z, A, weight fractions) — feeds cross-sections and CSD coefficients.

### Model parameters

- **`breakE`** (~45 eV) — Nieminen ↔ Joy/Luo CSD crossover.
- **Barrier model**: `u0` (height), `lambda` (width), and the `classical` vs. quantum-WKB transmission flag.

### Beam / simulation parameters

- **Beam energy** (200 eV–20 keV sweep) — yield peaks at low keV; the classic SE-yield-vs-energy curve.
- **Beam size** (~0.5 nm), **incidence/geometry**, **trajectory count** (~5000, statistics only), **`seThreshold`** (~50 eV SE/BSE cutoff), histogram bin size.

## Parameter reference table

| Parameter | Type | Typical value | Role |
|-----------|------|---------------|------|
| Work function | Material | 5.15 eV (Ni) | Energy to extract electron to vacuum |
| `energyCBbottom` (Fermi) | Material | −8.8 eV (Ni) | Sets escape barrier height |
| `energySEgen` | Model | ~30 eV (metals) | Energy to create one SE |
| `breakE` | Model | 45 eV | Nieminen ↔ Joy/Luo CSD crossover |
| Bandgap | Material | 0 (metals) | Generation / stopping |
| Density | Material | ~8700 kg/m³ (Ni) | Stopping power |
| Beam energy | Simulation | 200–20000 eV | Drives yield-vs-energy curve |
| Beam size | Simulation | ~0.5 nm | Gaussian beam sigma |
| `seThreshold` | Detection | 50 eV | SE/BSE energy cutoff |
| Trajectories | Simulation | 5000 | Statistics only |
| Barrier `u0` / `lambda` / `classical` | Barrier model | material-set | Transmission probability |
| SE1/SE2 split | Classification | `cos(θ) > 0` | True vs. BSE-induced secondary |

## Bottom line

SE yield is set primarily by the **surface barrier** (work function + Fermi level), the **SE generation energy**, and the **beam energy**, with material composition/density and the CSD/barrier model choices as secondary modifiers.
