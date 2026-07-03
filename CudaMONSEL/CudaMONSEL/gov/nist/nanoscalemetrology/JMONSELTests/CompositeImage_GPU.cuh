// file: gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage_GPU.cuh
//
// CUDA GPU acceleration for CompositeImage simulation.
// The implementation flattens the CPU object graph into POD material,
// geometry, and scan tables, then launches one CUDA thread per trajectory.
// It maps the same major physics models as the CPU path but is validated as
// an accelerated approximation rather than a bitwise-equivalent port.

#ifndef _COMPOSITE_IMAGE_GPU_CUH_
#define _COMPOSITE_IMAGE_GPU_CUH_

#include <vector>

namespace CompositeImageGPU
{
   // -----------------------------------------------------------------------
   // Per-element NIST Mott + Browning data uploaded to GPU global memory
   // -----------------------------------------------------------------------
   static const int SPWEM_LEN = 61;
   static const int X1_LEN    = 201;

   struct ElemTableGPU
   {
      double spwem[SPWEM_LEN];               // dimensionless (units of a0^2)
      double x1[SPWEM_LEN * X1_LEN];        // CDF table, row-major [energy][cdf]
      double MottXSatMin;                    // total XS at extrapolation boundary (m^2)
      double sfBrowning;                     // = MottXSatMin / browningXS(extraBelowE)
      double Zp17, Zp2, Zp3;               // Z^1.7, Z^2.0, Z^3.0 for Browning
      double extraBelowE;                    // 50 eV in SI (method-1 boundary)
      int    Z;
   };

   // -----------------------------------------------------------------------
   // Per-material data (4 slots: 0=vacuum, 1=SL, 2=bulk, 3=precipitate)
   // -----------------------------------------------------------------------
   static const int MAX_MAT_ELEM = 8;

   struct MatGPU
   {
      // Elastic scatter (SelectableElasticSM)
      int    nElems;
      int    elemIdx[MAX_MAT_ELEM];           // index into ElemTableGPU array
      double scalefactor[MAX_MAT_ELEM];       // 1000 * weightFrac / atomicWeight_SI
      double densityNa;                       // density * AvagadroNumber (m^-3)

      // Joy-Luo-Nieminen CSD
      int    nCSD;
      double recipJ[MAX_MAT_ELEM];
      double coefJL[MAX_MAT_ELEM];
      double betaJL[MAX_MAT_ELEM];
      double breakE;                          // SI
      double bhplus1eV;                       // SI
      double gammaN;                          // Nieminen proportionality constant

      // Tracking threshold
      double minEtrack;                       // SI (electrons below this are dropped)

      // FittedInelSM
      double energySEgen;                     // SI
      double eFermi;                          // SI

      // ExpQM barrier
      double energyCBbottom;                  // SI (negative for real materials)

      bool   isVacuum;
   };

   // -----------------------------------------------------------------------
   // Geometry: one or more precipitate spheres in the bulk half-space, with a
   // uniform-grid acceleration structure (CSR cell lists). Spheres never
   // intersect (asserted at parse time), so a precipitate region is exited
   // only through its own sphere surface. Each sphere is registered in every
   // grid cell its AABB overlaps, so the cell size is free to follow the
   // median radius even when radii span decades.
   // -----------------------------------------------------------------------
   struct SphereGPU
   {
      double x, y, z, r;                     // center (m) + radius (m)
   };

   struct GeomGPU
   {
      const SphereGPU* spheres;              // device array, nullptr if none
      int    nSpheres;
      bool   spheresAreVoid;                 // all spheres vacuum (etched-out)
      // Uniform grid over the sphere-populated AABB.
      double gridOx, gridOy, gridOz;         // grid origin (m)
      double cellInv;                        // 1 / cell edge (m^-1)
      int    ncx, ncy, ncz;
      const int* cellStart;                  // device, [ncx*ncy*ncz + 1]
      const int* cellItems;                  // device, sphere indices
      double slThick;                        // m (0 if no SL)
      bool   hasSL;
   };

   // -----------------------------------------------------------------------
   // Detector acceptance window in (exit energy, take-off polar angle beta).
   // beta is measured from the outward optic axis (-z): 0 = up the column,
   // 90 deg = grazing. Azimuth is integrated (annular / in-lens detectors).
   // -----------------------------------------------------------------------
   struct DetectorSpec
   {
      double eMinJ, eMaxJ;            // energy window (J), half-open [eMin, eMax)
      double betaMinRad, betaMaxRad;  // take-off polar-angle window (rad), inclusive
   };

   __host__ __device__ inline bool detectorAccepts(const DetectorSpec& d,
                                                   double energyJ, double betaRad)
   {
      return energyJ >= d.eMinJ && energyJ < d.eMaxJ
          && betaRad >= d.betaMinRad && betaRad <= d.betaMaxRad;
   }

   // -----------------------------------------------------------------------
   // Host-side config passed to runImageGPU()
   // -----------------------------------------------------------------------
   struct GPURunConfig
   {
      // Materials (indices match region IDs: 0=vacuum,1=SL,2=bulk,3=precip)
      MatGPU      mats[4];
      std::vector<ElemTableGPU> elems;       // all unique elements
      GeomGPU     geom;                      // device pointers patched by run()

      // Host-side geometry tables; run() uploads them and patches geom.
      std::vector<SphereGPU> spheres;
      std::vector<int> cellStart;            // CSR offsets, ncx*ncy*ncz + 1
      std::vector<int> cellItems;            // sphere indices per cell

      // Pixel scan grid
      int    nx, ny;
      std::vector<double> pixelX;            // [ny*nx] beam center x (m)
      std::vector<double> pixelY;            // [ny*nx] beam center y (m)

      // Beam
      double beamE;                          // J
      double beamSizeM;                      // sigma (m)
      double beamStartZ;                     // m (negative, in vacuum)

      // Simulation
      int    trajPerPixel;
      double seThresholdJ;                   // SE/BSE energy cutoff (J)
      bool   trackSecondaries = true;        // false => BSE-only (skip SE tracking)
      unsigned long long seed;               // deterministic base seed

      // Optional per-pixel (escape energy x take-off angle) histogram.
      // When histEnabled, every escaping electron is binned by exit energy and
      // by take-off polar angle beta (from the outward optic axis, -z).
      bool   histEnabled = false;
      int    histNEbins = 0;                  // energy bins, width = histEbinWidthJ
      int    histNBbins = 0;                  // polar-angle bins, 0..90 deg
      double histEbinWidthJ = 0.0;            // energy bin width (J)

      // Optional inline detector channels (energy x take-off-angle windows).
      std::vector<DetectorSpec> detectors;

      // Optional per-pixel radial escape-distance histogram (by type): bins the
      // lateral distance |escape_xy - beam_center_xy| to expose SE1 (narrow) vs
      // SE2 (wide) delocalization. Disabled when radialNBins == 0.
      int    radialNBins = 0;
      double radialMaxM = 0.0;                 // max radius (m); bin width = radialMaxM/radialNBins
   };

   // Output per pixel (yield = counts / trajPerPixel)
   struct GPUOutput
   {
      std::vector<double> seYield;
      std::vector<double> se1Yield;
      std::vector<double> se2Yield;
      std::vector<double> bseYield;
      std::vector<double> genSeYield; // SE generation events per trajectory

      // Optional escape histogram: raw counts, flattened
      // [pixel][type][energy_bin][angle_bin]; empty when histEnabled was false.
      // type: 0 = SE1, 1 = SE2, 2 = other (backscattered primary).
      std::vector<int> escapeHist;
      int histNTypes = 0;
      int histNEbins = 0;
      int histNBbins = 0;

      // Optional inline detector channels: yields flattened [pixel*nDet + d].
      std::vector<double> detYield;
      int nDet = 0;

      // Optional radial escape-distance histogram: counts flattened
      // [pixel][type][radial_bin]; empty when radialNBins was 0.
      std::vector<int> radialHist;
      int radialNTypes = 0;
      int radialNBins = 0;
      double radialMaxM = 0.0;
   };

   // Returns true if at least one CUDA device is present
   bool isAvailable();

   // Run the full image simulation on GPU.
   // Returns false on CUDA error (caller should fall back to CPU).
   bool run(const GPURunConfig& cfg, GPUOutput& out);

} // namespace CompositeImageGPU

#endif // _COMPOSITE_IMAGE_GPU_CUH_
