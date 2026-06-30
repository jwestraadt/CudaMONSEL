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
   // Geometry
   // -----------------------------------------------------------------------
   struct GeomGPU
   {
      double precipCx, precipCy, precipCz;   // m
      double precipR2, precipR;              // m^2, m
      double slThick;                        // m (0 if no SL)
      bool   hasSL;
   };

   // -----------------------------------------------------------------------
   // Host-side config passed to runImageGPU()
   // -----------------------------------------------------------------------
   struct GPURunConfig
   {
      // Materials (indices match region IDs: 0=vacuum,1=SL,2=bulk,3=precip)
      MatGPU      mats[4];
      std::vector<ElemTableGPU> elems;       // all unique elements
      GeomGPU     geom;

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
      unsigned long long seed;               // deterministic base seed

      // Optional per-pixel (escape energy x take-off angle) histogram.
      // When histEnabled, every escaping electron is binned by exit energy and
      // by take-off polar angle beta (from the outward optic axis, -z).
      bool   histEnabled = false;
      int    histNEbins = 0;                  // energy bins, width = histEbinWidthJ
      int    histNBbins = 0;                  // polar-angle bins, 0..90 deg
      double histEbinWidthJ = 0.0;            // energy bin width (J)
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
      // [pixel][energy_bin][angle_bin]; empty when histEnabled was false.
      std::vector<int> escapeHist;
      int histNEbins = 0;
      int histNBbins = 0;
   };

   // Returns true if at least one CUDA device is present
   bool isAvailable();

   // Run the full image simulation on GPU.
   // Returns false on CUDA error (caller should fall back to CPU).
   bool run(const GPURunConfig& cfg, GPUOutput& out);

} // namespace CompositeImageGPU

#endif // _COMPOSITE_IMAGE_GPU_CUH_
