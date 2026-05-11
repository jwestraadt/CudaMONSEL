# CUDA Implementation Plan

## Current status

The first CUDA path is implemented for `composite_image` simulations. It keeps the existing CPU/OpenMP implementation as the reference path and adds a JSON-selectable backend:

- `"backend": "cpu"` forces the existing OpenMP code.
- `"backend": "gpu"` forces the new CUDA backend and fails if CUDA is unavailable.
- `"backend": "auto"` tries CUDA first and falls back to CPU.

The CUDA implementation uses a flat, POD representation of the material tables, geometry, scan pixels, and counters. The kernel launches one CUDA thread per electron trajectory and accumulates per-pixel SE, SE1, SE2, BSE, and SE-generation counts with atomics.

## Implemented files

- `CudaMONSEL/CudaMONSEL/gov/nist/nanoscalemetrology/JMONSELTests/CompositeImage_GPU.cuh`
- `CudaMONSEL/CudaMONSEL/gov/nist/nanoscalemetrology/JMONSELTests/CompositeImage_GPU.cu`
- `CudaMONSEL/CudaMONSEL/gov/nist/nanoscalemetrology/JMONSELTests/CompositeImage.cu`
- `CudaMONSEL/CudaMONSEL/gov/nist/nanoscalemetrology/JMONSEL/ExpQMBarrierSM.cu`
- `CudaMONSEL/CudaMONSEL/CudaMONSEL.vcxproj`

## CUDA model mapping

The GPU path currently maps these CPU pieces:

- Geometry: chamber sphere, vacuum/material half-space, optional surface layer, and spherical precipitate.
- Elastic scattering: NIST Mott tables with Browning fallback below the tabulated range and screened Rutherford above it.
- Inelastic scattering: fitted SE generation rate from the Joy-Luo-Nieminen continuous slowing-down model.
- Barrier behavior: classical ExpQM-style transmission/reflection across material boundaries.
- Output: same CSV/PGM products as `composite_image`, plus a `genSeYield` diagnostic field.

This is intentionally not a direct port of the existing object graph. The CPU path uses STL containers, virtual dispatch, heap-allocated secondary electrons, action listeners, and thread-local CPU RNG. Those patterns are not suitable inside CUDA kernels, so the GPU path flattens the hot simulation state.

## Benchmarks

Benchmarks were run on an NVIDIA RTX 2000 Ada Generation Laptop GPU, driver 595.71, using the Release build.

| Workload | CPU/OpenMP | CUDA GPU | Speedup |
| --- | ---: | ---: | ---: |
| 16x16, 100 trajectories/pixel | 0.9 s | 2.0 s | 0.45x |
| 64x64, 100 trajectories/pixel | 14.5 s | 6.0 s | 2.4x |
| 64x64, 1000 trajectories/pixel | 160.6 s baseline | 45.3 s | 3.5x |

The GPU path is slower for tiny jobs because kernel launch, memory transfer, and setup overhead dominate. It becomes worthwhile once the image has enough independent trajectories. For the existing 64x64, 1000 trajectories/pixel run, the measured speedup is about 3.5x.

## Validation notes

`ExpQMBarrierSM` now recovers the actual ancestor boundary shape normal when a precipitate-region electron exits through the flat free surface. The recovery includes a direct `NormalMultiPlaneShape` endpoint-plane check so CPU hits whose plane intersection lands numerically just past `t = 1` do not fall back to the electron direction. The GPU `PRECIP -> VAC` and `PRECIP -> SL` transitions use the same physical flat-surface normal `{0,0,-1}`.

Latest clean Release validation used `gpu_smoke_1k_compare.json` at 16x16 pixels and 1000 trajectories/pixel:

| Metric | CPU | GPU | GPU-CPU |
| --- | ---: | ---: | ---: |
| Mean SE yield | 0.48096 | 0.47661 | -0.91% |
| Mean SE1 yield | 0.28781 | 0.28567 | -0.74% |
| Mean SE2 yield | 0.18673 | 0.18416 | -1.38% |
| Mean BSE yield | 0.32084 | 0.31884 | -0.62% |
| Mean genSE/traj | 25.7109 | 25.7554 | +0.17% |
| SE escape ratio | 1.87% | 1.85% | -0.02 percentage points |
| Runtime | 9.0 s | 4.3 s | 2.1x speedup |

The original CPU/GPU barrier-normal mismatch produced an SE split of about 0.62 absolute yield. The first parent-normal CPU fix reduced that to about 7% GPU-low SE for the surface-straddling case. The endpoint-plane fix removes the remaining CPU fallback-normal cases and reduces the 16x16, 1000 trajectories/pixel surface-straddling discrepancy to about 1% SE, with BSE within about 0.6%.

## Root cause of the previous SE discrepancy (fixed)

The original GPU used `{0,0,-1}` as the surface normal for `PRECIP -> VAC` barrier transitions, which is physically correct for the flat `z = 0` free surface. The CPU reference (`ExpQMBarrierSM`) used `currentRegion->getShape()` to find the boundary shape; for an electron in the precipitate region that shape is `precipSphere_t` (a `Sphere`), which does not implement `NormalShape`. The fallback was `nb = n0` (electron direction), giving `cosalpha = 1` and unconditional transmission whenever `kE > deltaU`.

The fix is to keep the existing CPU `getPreviousNormal()` path when the current region shape can provide a normal, then walk parent regions and call `getFirstNormal(pos0, pos1)` on ancestor `NormalShape` boundaries when it cannot. For a surface-straddling precipitate this finds the flat plane normal instead of falling back to the electron direction. A second explicit `NormalMultiPlaneShape` endpoint check handles the numerical edge case where `RegionBase::findEndOfStep()` has already truncated the step to the plane but `getFirstNormal()` rejects the crossing because its computed intersection is slightly greater than `1`.

This behavior matters for the surface-straddling geometry (`center_depth = 0`). When the precipitate is fully buried (`center_depth > 0`) there are no direct `PRECIP -> VAC` transitions, so this specific correction should not change those paths.

## Diagnostic output

After each CPU/GPU comparison run the console prints:

```
CPU mean genSE/traj=X.XXXX  SE escape ratio=X.XXXX
GPU mean SE yield=X.XXXX  BSE yield=X.XXXX  total=X.XXXX
GPU mean genSE/traj=X.XXXX  SE escape ratio=X.XXXX
```

`genSE/traj` counts inelastic scatter events per trajectory; `SE escape ratio = SE yield / genSE/traj`. These numbers are useful for separating generation-rate bugs from escape-probability bugs.

## Next optimization steps

1. Move read-only material and scattering tables into constant or texture memory where practical.
2. Convert selected trajectory math from `double` to `float` after validation tolerances are defined.
3. Reduce local-memory pressure from the fixed secondary-electron stack.
4. Add a benchmark mode that suppresses per-pixel CPU progress printing and records backend timing in machine-readable form.
5. Run a longer post-fix validation, preferably 64x64 at 1000 trajectories/pixel, to tighten the stochastic SE comparison after the CPU boundary-normal fix.
