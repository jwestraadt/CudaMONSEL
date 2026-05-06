// file: gov\nist\nanoscalemetrology\JMONSELTests\BulkYield.cu
//
// SE and BSE yield simulation for bulk homogeneous phases.
// Currently configured for a Ni-based superalloy gamma / gamma-prime system.
//
// Scatter stack: SelectableElasticSM (NISTMott) + FittedInelSM + JoyLuoNieminenCSD
// No JMONSEL tables required.
//
// NOTE: FittedInelSM was calibrated for organic materials. For metals it gives
// a plausible SE yield trend but the absolute values should be treated as
// semi-quantitative until calibrated against experiment or a full tabulated
// inelastic model (TabulatedInelasticSM + JMONSEL tables) is used.
//
// HOW TO ADAPT:
//   1. Edit the composition arrays and property constants below.
//   2. For metals set BANDGAP=0 and EFERMI to the actual Fermi energy (eV).
//   3. Adjust ENERGY_SE_GEN: ~30 eV for metals, ~65 eV for organics.
//   4. The polaron trap model is omitted here (metals only).

#include "gov\nist\nanoscalemetrology\JMONSELTests\BulkYield.cuh"

#include "gov\nist\microanalysis\EPQLibrary\ToSI.cuh"
#include "gov\nist\microanalysis\EPQLibrary\Element.cuh"
#include "gov\nist\microanalysis\NISTMonte\NullMaterialScatterModel.cuh"
#include "gov\nist\microanalysis\NISTMonte\MonteCarloSS.cuh"
#include "gov\nist\microanalysis\NISTMonte\GaussianBeam.cuh"
#include "gov\nist\microanalysis\NISTMonte\RegionBase.cuh"
#include "gov\nist\microanalysis\NISTMonte\Sphere.cuh"
#include "gov\nist\microanalysis\NISTMonte\BackscatterStats.cuh"
#include "gov\nist\microanalysis\Utility\Math2.cuh"
#include "gov\nist\microanalysis\Utility\Histogram.cuh"

#include "gov\nist\nanoscalemetrology\JMONSEL\SEmaterial.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\ExpQMBarrierSM.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\MONSEL_MaterialScatterModel.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\SelectableElasticSM.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\NISTMottRS.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\JoyLuoNieminenCSD.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\FittedInelSM.cuh"
#include "gov\nist\nanoscalemetrology\JMONSEL\NormalMultiPlaneShape.cuh"

#include <fstream>
#include <chrono>

namespace BulkYield
{
   // =========================================================================
   // SIMULATION PARAMETERS
   // =========================================================================

   static const int    N_TRAJECTORIES  = 5000;
   static const double BEAM_SIZE_NM    = 0.5;    // Gaussian beam sigma (nm)
   static const double SE_THRESHOLD_EV = 50.0;   // electrons below this = SE

   // Beam energies to sweep (eV) -- keep below ~25 keV; CSD slow-down at very
   // high energies can make trajectories extremely long in the current framework
   static const double BEAM_ENERGIES_EV[] = { 200., 500., 1000., 2000., 5000., 10000., 15000., 20000. };
   static const int    N_BEAM_ENERGIES    = 8;

   // =========================================================================
   // GAMMA PHASE  (Ni-rich FCC matrix)
   // Representative composition for a single-crystal Ni superalloy gamma phase.
   // Adjust fractions to match your specific alloy (at%).
   // =========================================================================
   static const ElementT* GAMMA_ELEM[] = {
      &Element::Ni, &Element::Cr, &Element::Co,
      &Element::W,  &Element::Re, &Element::Al,
      &Element::Ta
   };
   static const double GAMMA_FRAC[] = {
      63.,          8.,           10.,
      6.,           4.,           6.,
      3.
   };
   static const int    GAMMA_N        = 7;
   static const double GAMMA_DENSITY  = 8700.;   // kg/m^3
   static const double GAMMA_WORKFUN  = 5.15;    // eV  (~Ni workfunction)
   static const double GAMMA_EFERMI   = 8.8;     // eV  (Fermi energy above CB bottom, ~Ni)
   static const double GAMMA_SE_GEN   = 30.;     // eV  per SE generation event (metal estimate)
   static const double GAMMA_BREAK_E  = 45.;     // eV  CSD tracking cutoff

   // =========================================================================
   // GAMMA-PRIME PHASE  (Ni3Al ordered L1-2)
   // Adjust fractions to match your alloy's gamma-prime composition (at%).
   // =========================================================================
   static const ElementT* GPRIME_ELEM[] = {
      &Element::Ni, &Element::Al, &Element::Ti,
      &Element::Ta, &Element::Cr
   };
   static const double GPRIME_FRAC[] = {
      75.,          12.,          5.,
      5.,           3.
   };
   static const int    GPRIME_N        = 5;
   static const double GPRIME_DENSITY  = 8200.;  // kg/m^3 (Ni3Al ~7450; heavier with Ta)
   static const double GPRIME_WORKFUN  = 4.9;    // eV  (slightly lower than Ni matrix)
   static const double GPRIME_EFERMI   = 7.5;    // eV  (intermetallic, less certain)
   static const double GPRIME_SE_GEN   = 30.;    // eV  per SE generation event
   static const double GPRIME_BREAK_E  = 45.;    // eV  CSD tracking cutoff

   // =========================================================================

   static void runPhase(
      const char*      phaseName,
      const ElementT** elems,
      const double*    fracs,
      int                  nElems,
      double               density,
      double               workfun,
      double               efermi,
      double               energySEgen,
      double               breakEeV,
      std::ofstream&       outfile)
   {
      printf("\n--- Phase: %s ---\n", phaseName); fflush(stdout);
      printf("BeamE_eV,BSE_yield,SE_yield,total_yield\n"); fflush(stdout);

      // bandgap = 0 for metals; potU = -(workfun + efermi)
      double potU = -workfun - efermi;

      CompositionT comp;
      comp.defineByMoleFraction(elems, nElems, fracs, nElems);

      SEmaterialT mat(comp, density);
      mat.setWorkfunction(ToSI::eV(workfun));
      mat.setBandgap(0.0);                       // metal
      mat.setEnergyCBbottom(ToSI::eV(potU));

      SelectableElasticSMT elasticSM(mat, NISTMottRS::Factory);
      JoyLuoNieminenCSDT   csd(mat, ToSI::eV(breakEeV));
      FittedInelSMT        inelSM(mat, ToSI::eV(energySEgen), csd);

      ExpQMBarrierSMT             barrier(&mat);
      MONSEL_MaterialScatterModelT msm(&mat, &barrier);
      msm.addScatterMechanism(&elasticSM);
      msm.addScatterMechanism(&inelSM);
      msm.setCSD(&csd);

      // vacuum
      SEmaterialT                  vacMat;
      vacMat.setName("vacuum");
      ExpQMBarrierSMT              vacBarrier(&vacMat);
      MONSEL_MaterialScatterModelT vacMSM(&vacMat, &vacBarrier);

      // geometry: sphere chamber + bulk half-space (z > 0 is inside sample)
      NullMaterialScatterModelT NULL_MSM;
      const double center[] = { 0., 0., 0. };
      SphereT sphere(center, MonteCarloSS::ChamberRadius);

      RegionT chamber(nullptr, &NULL_MSM, &sphere);
      chamber.updateMaterial(*chamber.getScatterModel(), vacMSM);

      const double normalvec[]  = { 0., 0., -1. };
      const double surfacePos[] = { 0., 0.,  0. };
      NormalMultiPlaneShapeT surface;
      PlaneT pl(normalvec, 3, surfacePos, 3);
      surface.addPlane(pl);
      RegionT bulkRegion(&chamber, &msm, (NormalShapeT*)&surface);

      double beamsize = BEAM_SIZE_NM * 1.e-9;

      for (int ei = 0; ei < N_BEAM_ENERGIES; ++ei) {
         double beamEeV = BEAM_ENERGIES_EV[ei];
         double beamE   = ToSI::eV(beamEeV);

         GaussianBeamT eg(beamsize, beamE, center);
         double egCenter[] = { 0., 0., -1.e-9 };  // 1 nm above surface
         eg.setCenter(egCenter);

         MonteCarloSS::MonteCarloSS monte(&eg, &chamber, eg.createElectron());

         int nbins = (int)(beamEeV / 10.);
         if (nbins < 1) nbins = 1;
         BackscatterStatsT back(monte, nbins);
         monte.addActionListener(back);

         monte.runMultipleTrajectories(N_TRAJECTORIES);

         const HistogramT& hist    = back.backscatterEnergyHistogram();
         double ePerBin            = beamEeV / hist.binCount();
         int    maxSEbin           = (int)(SE_THRESHOLD_EV / ePerBin);
         int    totalSE            = 0;
         for (int j = 0; j < maxSEbin && j < (int)hist.binCount(); ++j)
            totalSE += hist.counts(j);

         double SEY   = (double)totalSE / N_TRAJECTORIES;
         double BSEY  = back.backscatterFraction() - SEY;
         double total = back.backscatterFraction();

         printf("%.0f,%.4f,%.4f,%.4f\n", beamEeV, BSEY, SEY, total); fflush(stdout);
         outfile << phaseName << "," << beamEeV << ","
                 << BSEY << "," << SEY << "," << total << "\n";
         outfile.flush();

         monte.removeActionListener(back);
      }
   }

   void run()
   {
      printf("BulkYield: Ni superalloy gamma/gamma-prime, %d trajectories per point\n",
             N_TRAJECTORIES); fflush(stdout);

      std::ofstream outfile("BulkYield_output.csv");
      outfile << "phase,BeamE_eV,BSE_yield,SE_yield,total_yield\n";

      auto wallStart = std::chrono::system_clock::now();

      runPhase("gamma",
               GAMMA_ELEM,  GAMMA_FRAC,  GAMMA_N,
               GAMMA_DENSITY, GAMMA_WORKFUN, GAMMA_EFERMI,
               GAMMA_SE_GEN, GAMMA_BREAK_E,
               outfile);

      runPhase("gamma_prime",
               GPRIME_ELEM, GPRIME_FRAC, GPRIME_N,
               GPRIME_DENSITY, GPRIME_WORKFUN, GPRIME_EFERMI,
               GPRIME_SE_GEN, GPRIME_BREAK_E,
               outfile);

      auto wallEnd = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = wallEnd - wallStart;
      printf("\nBulkYield: done in %.1f s  -->  BulkYield_output.csv\n",
             elapsed.count()); fflush(stdout);
      outfile.close();
   }
}
