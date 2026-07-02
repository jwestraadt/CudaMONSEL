// file: gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage.cu
//
// SEM image simulation for a composite material: one matrix phase with one or
// more spherical precipitates embedded at configurable depths (the legacy
// "precipitate" deck key gives one sphere; the "precipitates" array, e.g. from
// an r65gen microstructure export, gives many — all sharing the precipitate
// phase). The beam is scanned over a 2D (x, y) grid; SE and BSE yields are
// recorded per pixel and written to a CSV file and to PGM image files.
//
// Precipitate center_depth_nm = 0 places the centroid on the surface so the
// sphere is cut exactly in half: the upper hemisphere is in vacuum (no scatter)
// and the lower hemisphere is the precipitate phase. Sphere surfaces must not
// intersect (asserted at parse time). Multi-sphere decks run on the CPU region
// graph; the GPU kernel's geometry is single-sphere in this revision.
//
// Scatter stack: SelectableElasticSM (NISTMott) + FittedInelSM + JoyLuoNieminenCSD
// (same as bulk_yield — no JMONSEL tables required).

#include "gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage.cuh"

#include "gov\nist\microanalysis\EPQLibrary\ToSI.cuh"
#include "gov\nist\microanalysis\EPQLibrary\Element.cuh"
#include "gov\nist\microanalysis\NISTMonte\NullMaterialScatterModel.cuh"
#include "gov\nist\microanalysis\NISTMonte\MonteCarloSS.cuh"
#include "gov\nist\microanalysis\NISTMonte\GaussianBeam.cuh"
#include "gov\nist\microanalysis\NISTMonte\RegionBase.cuh"
#include "gov\nist\microanalysis\NISTMonte\Sphere.cuh"
#include "gov\nist\microanalysis\NISTMonte\BackscatterStats.cuh"
#include "gov\nist\microanalysis\NISTMonte\Electron.cuh"
#include "gov\nist\microanalysis\EPQLibrary\FromSI.cuh"
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
#include "gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage_GPU.cuh"
#include "gov\nist\microanalysis\EPQLibrary\BrowningEmpiricalCrossSection.cuh"
#include "gov\nist\microanalysis\EPQLibrary\MeanIonizationPotential.cuh"
#include "gov\nist\microanalysis\EPQLibrary\NISTMottScatteringAngle.cuh"
#include "gov\nist\microanalysis\EPQLibrary\PhysicalConstants.cuh"

#include <array>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace CompositeImage
{
   struct ScatteringConfig
   {
      std::string elastic;
      std::string inelastic;
      std::string csd;
      std::string barrier;
   };

   struct PhaseConfig
   {
      std::string name;
      std::vector<const ElementT*> elements;
      std::vector<double> fractions;
      double density;
      double workfun;
      double efermi;
      double bandgap;
      double energySEgen;
      double breakEeV;
   };

   struct PrecipitateConfig
   {
      double radiusNm;
      double centerXNm;
      double centerYNm;
      double centerDepthNm;
      bool   isVoid;        // true => vacuum sphere (etched-out particle), no scatter inside
   };

   struct SurfaceLayerConfig
   {
      bool       enabled;
      double     thicknessNm;
      PhaseConfig phase;
   };

   struct ScanConfig
   {
      double centerXNm;    // scan center x (usually = precipitate centerXNm)
      double centerYNm;    // scan center y
      double halfWidthNm;  // half field-of-view (scan covers ±halfWidthNm in x and y)
      int    nxPixels;
      int    nyPixels;
      bool   radial;       // true => 1-D radial line: nx samples at (r,0), r in [0, radialMaxNm]
      double radialMaxNm;  // outer radius of the radial line scan
   };

   // One synthesised detector channel: an (energy, take-off-angle) acceptance box.
   struct DetectorConfig
   {
      std::string name;
      std::string outputPgm;   // optional per-detector PGM map
      double      eMinEv;
      double      eMaxEv;      // a very large value means "no upper bound"
      double      betaMinDeg;  // take-off polar angle from outward optic axis (-z)
      double      betaMaxDeg;
   };

   struct CompositeImageConfig
   {
      std::string name;
      std::string outputCsv;
      std::string outputPgmSE;
      std::string outputPgmSE1;
      std::string outputPgmSE2;
      std::string outputPgmBSE;
      int         trajectoriesPerPixel;
      double      beamEnergyEv;
      double      beamSizeNm;
      double      seThresholdEv;
      double      histogramBinSizeEv;
      bool        trackSecondaries;   // false => BSE-only (skip SE tracking; GPU only)
      std::string backend;
      unsigned long long rngSeed;
      ScatteringConfig   scattering;
      PhaseConfig        matrixPhase;
      PhaseConfig        precipitatePhase;
      // One or more precipitate spheres, all sharing precipitatePhase. The
      // legacy single "precipitate" deck key edits element 0; the
      // "precipitates" array replaces the whole list (may be empty).
      std::vector<PrecipitateConfig> precipitates;
      ScanConfig         scan;
      SurfaceLayerConfig surfaceLayer;

      // Optional per-pixel (escape energy x take-off angle) histogram (GPU backend only).
      bool        escapeHistEnabled;
      int         escapeHistAngleBins;
      std::string escapeHistOutput;   // output base name; writes <base>.bin + <base>.json
      int         escapeHistRadialBins;   // 0 disables the radial escape-distance histogram
      double      escapeHistRadialMaxNm;  // <=0 defaults to the scan half-width

      // Optional inline detector channels (CPU + GPU).
      std::vector<DetectorConfig> detectors;

      // Optional single-probe trajectory capture (CPU backend only): full electron
      // paths for the 2D interaction-volume view + surface-escape records for the
      // 3D escape-electron view. When enabled, forces a 1x1 scan at the scan center
      // and the CPU backend.
      bool        tcEnabled;
      int         tcMaxFullPaths;        // primaries recorded in full (with their secondaries)
      int         tcMaxEscapes;          // cap on surface-escape records
      int         tcStepStride;          // keep every Nth scatter step (paths only)
      bool        tcRecordSecondaries;   // record SE paths in the 2D view
      bool        tcIncludeAbsorbed;     // keep absorbed (non-escaping) paths
      std::string tcOutput;              // base name; writes <base>_traj.csv + <base>_escapes.csv
   };

   class SecondaryCountListener : public ActionListenerT
   {
   public:
      void actionPerformed(const int ae) override
      {
         if (ae == MonteCarloSS::StartSecondaryEvent)
            ++startSecondaryEvents;
      }

      int generatedSecondaries() const
      {
         return startSecondaryEvents / 2;
      }

   private:
      int startSecondaryEvents = 0;
   };

   // CPU-path detector accumulator: mirrors the GPU recordEscape detector loop.
   // On each BackscatterEvent it reads the exiting electron's energy and exit
   // direction and tallies every detector whose (energy, take-off-angle) window
   // accepts it. beta = pi - theta (theta = exit direction polar angle from +z).
   class DetectorCountListener : public ActionListenerT
   {
   public:
      DetectorCountListener(const MonteCarloSS::MonteCarloSS& monte,
                            const std::vector<CompositeImageGPU::DetectorSpec>& specs)
         : mMonte(monte), mSpecs(specs), mCounts(specs.size(), 0) {}

      void actionPerformed(const int ae) override
      {
         if (ae != MonteCarloSS::BackscatterEvent) return;
         const auto&  el      = mMonte.getElectron();
         const double energyJ = el.getEnergy();
         const double beta    = Math2::PI - el.getTheta();
         for (size_t d = 0; d < mSpecs.size(); ++d)
            if (CompositeImageGPU::detectorAccepts(mSpecs[d], energyJ, beta))
               ++mCounts[d];
      }

      const std::vector<int>& counts() const { return mCounts; }

   private:
      const MonteCarloSS::MonteCarloSS&                   mMonte;
      const std::vector<CompositeImageGPU::DetectorSpec>& mSpecs;
      std::vector<int>                                    mCounts;
   };

   // Single-probe trajectory recorder (CPU backend only). Attaches to the same
   // MonteCarloSS event stream as BackscatterStats and captures two products for
   // the visualizations:
   //   * full electron paths (primaries + secondaries) for the 2D
   //     interaction-volume view (<base>_traj.csv), and
   //   * per-electron surface-escape records (position + take-off direction) for
   //     the 3D escape-electron view (<base>_escapes.csv).
   // SE/BSE (energy threshold) and SE1/SE2 (Electron::getType) classification
   // mirrors BackscatterStats exactly. The surface escape point comes from the
   // last ExitMaterialEvent (region change into vacuum); BackscatterEvent fires
   // far out at the chamber boundary and is used only to confirm the escape.
   class TrajectoryRecorder : public ActionListenerT
   {
   public:
      TrajectoryRecorder(const MonteCarloSS::MonteCarloSS& monte, double seThresholdJ,
                         int maxFullPaths, int maxEscapeEvents, int stepStride,
                         bool recordSecondaries, bool includeAbsorbed)
         : mMonte(monte), mSeThreshJ(seThresholdJ),
           mMaxFullPaths(maxFullPaths < 0 ? 0 : maxFullPaths),
           mMaxEscapes(maxEscapeEvents < 0 ? 0 : maxEscapeEvents),
           mStride(stepStride < 1 ? 1 : stepStride),
           mRecordSecondaries(recordSecondaries), mIncludeAbsorbed(includeAbsorbed) {}

      void actionPerformed(const int ae) override
      {
         const Electron::Electron& el = mMonte.getElectron();
         const long id = el.getIdent();

         if (ae == MonteCarloSS::TrajectoryStartEvent) {
            mRecordingThis = (mPrimaryCount < mMaxFullPaths);
            ++mPrimaryCount;
            if (mRecordingThis) openPath(el);
         }
         else if (ae == MonteCarloSS::StartSecondaryEvent) {
            // Fires twice (once as parent, once as the switched-in secondary);
            // lazily open only the not-yet-seen new secondary ident.
            if (mRecordSecondaries && mRecordingThis && mPaths.find(id) == mPaths.end())
               openPath(el);
         }
         else if (ae == MonteCarloSS::ScatterEvent) {
            if (mRecordingThis) appendPt(el, true);
         }
         else if (ae == MonteCarloSS::NonScatterEvent) {
            if (mRecordingThis) appendPt(el, false);
         }
         else if (ae == MonteCarloSS::ExitMaterialEvent) {
            if (mRecordingThis) appendPt(el, false);
            // Candidate surface crossing (also fires on beam entry and on
            // internal matrix/precipitate boundaries; the last one before the
            // terminal BackscatterEvent is the true surface escape).
            const auto p = el.getPosition();
            ExitState& ex = mLastExit[id];
            ex.x = p[0]; ex.y = p[1]; ex.z = p[2];
            ex.energyJ = el.getEnergy();
            ex.theta = el.getTheta(); ex.phi = el.getPhi();
            ex.has = true;
         }
         else if (ae == MonteCarloSS::BackscatterEvent) {
            finalizeEscape(el, id);
         }
         else if (ae == MonteCarloSS::TrajectoryEndEvent) {
            flushTrajectory();
         }
      }

      void writeCsvs(const std::string& base) const
      {
         const std::string trajPath = base + "_traj.csv";
         std::ofstream tf(trajPath.c_str());
         if (tf.good()) {
            tf << "traj_id,parent_id,gen,elec_type,exit_type,seg,x_nm,y_nm,z_nm,energy_ev\n";
            for (const Path& p : mOutPaths)
               for (size_t s = 0; s < p.pts.size(); ++s) {
                  const Pt& pt = p.pts[s];
                  tf << p.ident << "," << p.parentID << "," << p.gen << ","
                     << typeName(p.type) << "," << p.exitType << "," << s << ","
                     << pt.x * 1.e9 << "," << pt.y * 1.e9 << "," << pt.z * 1.e9 << ","
                     << pt.energyEv << "\n";
               }
         }
         const std::string escPath = base + "_escapes.csv";
         std::ofstream ef(escPath.c_str());
         if (ef.good()) {
            ef << "traj_id,elec_type,exit_type,x_nm,y_nm,z_nm,exit_energy_ev,theta_deg,phi_deg\n";
            for (const EscapeRec& e : mOutEscapes)
               ef << e.ident << "," << typeName(e.type) << "," << e.exitType << ","
                  << e.x * 1.e9 << "," << e.y * 1.e9 << "," << e.z * 1.e9 << ","
                  << e.energyEv << "," << e.thetaDeg << "," << e.phiDeg << "\n";
         }
         printf("CompositeImage: trajectories --> %s (%zu paths); escapes --> %s (%zu records)\n",
                trajPath.c_str(), mOutPaths.size(), escPath.c_str(), mOutEscapes.size());
         fflush(stdout);
      }

   private:
      struct Pt { double x, y, z, energyEv; };
      struct Path {
         long        ident = 0;
         long        parentID = 0;
         int         gen = 0;
         int         type = 0;              // 0 PRIMARY, 1 SE1, 2 SE2
         std::string exitType = "ABSORBED"; // BSE / SE1 / SE2 / SE / ABSORBED
         std::vector<Pt> pts;
         long        scatterSeen = 0;       // for step-stride downsampling
      };
      struct ExitState { double x = 0, y = 0, z = 0, energyJ = 0, theta = 0, phi = 0; bool has = false; };
      struct EscapeRec { long ident; int type; std::string exitType;
                         double x, y, z, energyEv, thetaDeg, phiDeg; };

      static const char* typeName(int t) { return t == 1 ? "SE1" : (t == 2 ? "SE2" : "PRIMARY"); }

      std::string classifyExit(double energyJ, int type) const
      {
         if (energyJ >= mSeThreshJ) return "BSE";           // > SE threshold ⇒ BSE (any type)
         if (type == 1) return "SE1";
         if (type == 2) return "SE2";
         return "SE";                                       // low-energy primary escape (rare)
      }

      void openPath(const Electron::Electron& el)
      {
         Path p;
         p.ident    = el.getIdent();
         p.parentID = el.getParentID();
         p.gen      = mMonte.getElectronGeneration();
         p.type     = (int)el.getType();
         const auto pos = el.getPosition();
         p.pts.push_back({ pos[0], pos[1], pos[2], FromSI::eV(el.getEnergy()) });
         mPaths.emplace(p.ident, std::move(p));
      }

      void appendPt(const Electron::Electron& el, bool strided)
      {
         auto it = mPaths.find(el.getIdent());
         if (it == mPaths.end()) return;
         Path& p = it->second;
         if (strided && (p.scatterSeen++ % mStride) != 0) return;
         const auto pos = el.getPosition();
         const double x = pos[0], y = pos[1], z = pos[2];
         if (!p.pts.empty()) {
            const Pt& last = p.pts.back();
            const double dx = x - last.x, dy = y - last.y, dz = z - last.z;
            if (dx * dx + dy * dy + dz * dz < 1.e-26) return;   // dedupe sub-pm nudges
         }
         p.pts.push_back({ x, y, z, FromSI::eV(el.getEnergy()) });
      }

      void finalizeEscape(const Electron::Electron& el, long id)
      {
         auto ex = mLastExit.find(id);
         if (ex == mLastExit.end() || !ex->second.has) return;   // never crossed the surface
         const ExitState& s = ex->second;
         const std::string et = classifyExit(s.energyJ, (int)el.getType());
         auto it = mPaths.find(id);
         if (it != mPaths.end()) it->second.exitType = et;
         if ((int)mOutEscapes.size() < mMaxEscapes)
            mOutEscapes.push_back({ id, (int)el.getType(), et, s.x, s.y, s.z,
                                    FromSI::eV(s.energyJ),
                                    s.theta * 180.0 / Math2::PI, s.phi * 180.0 / Math2::PI });
      }

      void flushTrajectory()
      {
         if (mRecordingThis)
            for (auto& kv : mPaths) {
               Path& p = kv.second;
               if (!mIncludeAbsorbed && p.exitType == "ABSORBED") continue;
               mOutPaths.push_back(std::move(p));
            }
         mPaths.clear();
         mLastExit.clear();
         mRecordingThis = false;
      }

      const MonteCarloSS::MonteCarloSS& mMonte;
      double mSeThreshJ;
      int    mMaxFullPaths, mMaxEscapes, mStride;
      bool   mRecordSecondaries, mIncludeAbsorbed;

      std::map<long, Path>      mPaths;      // full paths for the current trajectory (if recording)
      std::map<long, ExitState> mLastExit;   // last surface crossing per electron (for escapes)
      bool mRecordingThis = false;
      int  mPrimaryCount  = 0;

      std::vector<Path>      mOutPaths;
      std::vector<EscapeRec> mOutEscapes;
   };

   static void addElement(PhaseConfig& phase, const ElementT* element, double fraction)
   {
      phase.elements.push_back(element);
      phase.fractions.push_back(fraction);
   }

   static CompositeImageConfig defaultConfig()
   {
      CompositeImageConfig config;
      config.name                   = "Ni gamma matrix + gamma-prime precipitate image";
      config.outputCsv              = "CompositeImage_output.csv";
      config.outputPgmSE            = "CompositeImage_SE.pgm";
      config.outputPgmSE1           = "";
      config.outputPgmSE2           = "";
      config.outputPgmBSE           = "CompositeImage_BSE.pgm";
      config.trajectoriesPerPixel   = 100;
      config.beamEnergyEv           = 5000.0;
      config.beamSizeNm             = 0.5;
      config.seThresholdEv          = 50.0;
      config.histogramBinSizeEv     = 10.0;
      config.trackSecondaries       = true;
      config.backend                = "auto";
      config.rngSeed                = 0x5eed1234ULL;
      config.escapeHistEnabled      = false;
      config.escapeHistAngleBins    = 18;
      config.escapeHistOutput       = "";
      config.escapeHistRadialBins   = 0;
      config.escapeHistRadialMaxNm  = 0.0;
      config.tcEnabled              = false;
      config.tcMaxFullPaths         = 300;
      config.tcMaxEscapes           = 5000;
      config.tcStepStride           = 1;
      config.tcRecordSecondaries    = true;
      config.tcIncludeAbsorbed      = true;
      config.tcOutput               = "";
      config.scattering.elastic     = "nist_mott";
      config.scattering.inelastic   = "fitted";
      config.scattering.csd         = "joy_luo_nieminen";
      config.scattering.barrier     = "exp_qm";

      PhaseConfig& matrix = config.matrixPhase;
      matrix.name        = "gamma";
      matrix.density     = 8700.;
      matrix.workfun     = 5.15;
      matrix.efermi      = 8.8;
      matrix.bandgap     = 0.0;
      matrix.energySEgen = 30.;
      matrix.breakEeV    = 45.;
      addElement(matrix, &Element::Ni, 63.);
      addElement(matrix, &Element::Cr, 8.);
      addElement(matrix, &Element::Co, 10.);
      addElement(matrix, &Element::W,  6.);
      addElement(matrix, &Element::Re, 4.);
      addElement(matrix, &Element::Al, 6.);
      addElement(matrix, &Element::Ta, 3.);

      PhaseConfig& precip = config.precipitatePhase;
      precip.name        = "gamma_prime";
      precip.density     = 8200.;
      precip.workfun     = 4.9;
      precip.efermi      = 7.5;
      precip.bandgap     = 0.0;
      precip.energySEgen = 30.;
      precip.breakEeV    = 45.;
      addElement(precip, &Element::Ni, 75.);
      addElement(precip, &Element::Al, 12.);
      addElement(precip, &Element::Ti, 5.);
      addElement(precip, &Element::Ta, 5.);
      addElement(precip, &Element::Cr, 3.);

      PrecipitateConfig sphere;
      sphere.radiusNm      = 30.0;
      sphere.centerXNm     = 0.0;
      sphere.centerYNm     = 0.0;
      sphere.centerDepthNm = 0.0;   // centroid on surface
      sphere.isVoid        = false;
      config.precipitates.assign(1, sphere);

      config.scan.centerXNm  = 0.0;
      config.scan.centerYNm  = 0.0;
      config.scan.halfWidthNm = 90.0;           // 180 nm field of view
      config.scan.nxPixels   = 64;
      config.scan.nyPixels   = 64;
      config.scan.radial     = false;
      config.scan.radialMaxNm = 0.0;

      config.surfaceLayer.enabled     = false;
      config.surfaceLayer.thicknessNm = 0.0;

      return config;
   }

   static std::string normalizeModelName(const std::string& value)
   {
      std::string result = value;
      for (size_t i = 0; i < result.size(); ++i) {
         if (result[i] >= 'A' && result[i] <= 'Z')
            result[i] = (char)(result[i] - 'A' + 'a');
         else if (result[i] == '-' || result[i] == ' ')
            result[i] = '_';
      }
      return result;
   }

   static void requireModel(const std::string& fieldName, const std::string& modelName, const std::string& supported)
   {
      if (normalizeModelName(modelName) != supported)
         throw std::runtime_error("Unsupported composite_image scattering model for " + fieldName + ": " + modelName);
   }

   static void validateScatteringConfig(const ScatteringConfig& s)
   {
      requireModel("elastic",   s.elastic,   "nist_mott");
      requireModel("inelastic", s.inelastic, "fitted");
      requireModel("csd",       s.csd,       "joy_luo_nieminen");
      requireModel("barrier",   s.barrier,   "exp_qm");
   }

   static const RuntimeInput::JsonValue* findAny(
      const RuntimeInput::JsonValue& obj,
      const char* k1, const char* k2 = nullptr, const char* k3 = nullptr)
   {
      const RuntimeInput::JsonValue* v = k1 == nullptr ? nullptr : obj.find(k1);
      if (v) return v;
      v = k2 == nullptr ? nullptr : obj.find(k2);
      if (v) return v;
      return k3 == nullptr ? nullptr : obj.find(k3);
   }

   static std::string requireString(const RuntimeInput::JsonValue& obj, const char* k1, const char* k2 = nullptr)
   {
      const RuntimeInput::JsonValue* v = findAny(obj, k1, k2);
      if (!v) throw std::runtime_error(std::string("Missing required string field: ") + k1);
      if (!v->isString()) throw std::runtime_error(std::string("Expected string field: ") + k1);
      return v->stringValue;
   }

   static double requireNumber(const RuntimeInput::JsonValue& obj, const char* k1, const char* k2 = nullptr)
   {
      const RuntimeInput::JsonValue* v = findAny(obj, k1, k2);
      if (!v) throw std::runtime_error(std::string("Missing required number field: ") + k1);
      if (!v->isNumber()) throw std::runtime_error(std::string("Expected number field: ") + k1);
      return v->numberValue;
   }

   static double numberOr(const RuntimeInput::JsonValue& obj, double def,
                          const char* k1, const char* k2 = nullptr, const char* k3 = nullptr)
   {
      const RuntimeInput::JsonValue* v = findAny(obj, k1, k2, k3);
      if (!v) return def;
      if (!v->isNumber()) throw std::runtime_error(std::string("Expected number field: ") + k1);
      return v->numberValue;
   }

   static std::string stringOr(const RuntimeInput::JsonValue& obj, const std::string& def,
                               const char* k1, const char* k2 = nullptr, const char* k3 = nullptr)
   {
      const RuntimeInput::JsonValue* v = findAny(obj, k1, k2, k3);
      if (!v) return def;
      if (!v->isString()) throw std::runtime_error(std::string("Expected string field: ") + k1);
      return v->stringValue;
   }

   static const ElementT* elementByName(const std::string& symbol)
   {
      const ElementT& el = Element::byName(symbol.c_str());
      if (!el.isValid()) throw std::runtime_error("Unknown element in composite_image composition: " + symbol);
      return &el;
   }

   static void readCompositionObject(PhaseConfig& phase, const RuntimeInput::JsonValue& comp)
   {
      if (!comp.isObject()) throw std::runtime_error("composite_image phase composition must be an object");
      for (auto it = comp.objectValue.begin(); it != comp.objectValue.end(); ++it) {
         if (!it->second.isNumber()) throw std::runtime_error("composite_image composition values must be numbers");
         addElement(phase, elementByName(it->first), it->second.numberValue);
      }
   }

   static void readElementArray(PhaseConfig& phase, const RuntimeInput::JsonValue& elems)
   {
      if (!elems.isArray()) throw std::runtime_error("composite_image phase elements must be an array");
      for (size_t i = 0; i < elems.arrayValue.size(); ++i) {
         const RuntimeInput::JsonValue& item = elems.arrayValue[i];
         if (!item.isObject()) throw std::runtime_error("composite_image elements entries must be objects");
         addElement(phase, elementByName(requireString(item, "symbol", "element")),
                    requireNumber(item, "fraction", "mole_fraction"));
      }
   }

   static PhaseConfig readPhaseConfig(const RuntimeInput::JsonValue& json, const char* fieldName)
   {
      if (!json.isObject()) throw std::runtime_error(std::string(fieldName) + " must be a JSON object");
      PhaseConfig phase;
      phase.name        = requireString(json, "name");
      phase.density     = requireNumber(json, "density_kg_m3",                  "density");
      phase.workfun     = requireNumber(json, "work_function_ev",               "workfunction_ev");
      phase.efermi      = requireNumber(json, "fermi_energy_ev",                "efermi_ev");
      phase.bandgap     = numberOr     (json, 0.0, "bandgap_ev",               "band_gap_ev");
      phase.energySEgen = requireNumber(json, "secondary_generation_energy_ev", "energy_se_gen_ev");
      phase.breakEeV    = requireNumber(json, "break_energy_ev",               "csd_break_energy_ev");

      const RuntimeInput::JsonValue* elems = json.find("elements");
      const RuntimeInput::JsonValue* comp  = json.find("composition");
      if (elems)      readElementArray(phase, *elems);
      else if (comp)  readCompositionObject(phase, *comp);
      else throw std::runtime_error(std::string(fieldName) + " requires elements[] or composition{}");

      if (phase.elements.empty()) throw std::runtime_error(std::string(fieldName) + " has no elements: " + phase.name);
      return phase;
   }

   static ScanConfig readScanConfig(const RuntimeInput::JsonValue& json, const ScanConfig& defaults)
   {
      if (!json.isObject()) throw std::runtime_error("composite_image scan must be an object");
      ScanConfig sc = defaults;
      sc.centerXNm   = numberOr(json, sc.centerXNm,   "center_x_nm",   "cx_nm");
      sc.centerYNm   = numberOr(json, sc.centerYNm,   "center_y_nm",   "cy_nm");
      sc.halfWidthNm = numberOr(json, sc.halfWidthNm,
                                "half_width_nm", "fov_half_nm", "half_fov_nm");
      sc.nxPixels    = (int)numberOr(json, (double)sc.nxPixels, "nx_pixels", "nx");
      sc.nyPixels    = (int)numberOr(json, (double)sc.nyPixels, "ny_pixels", "ny");
      sc.radial      = json.boolOr("radial", sc.radial);
      sc.radialMaxNm = numberOr(json, sc.radialMaxNm, "radial_max_nm");
      if (sc.radial) {
         // 1-D radial line: nx samples along (r,0); one row.
         sc.nyPixels = 1;
         if (sc.radialMaxNm <= 0.0)
            throw std::runtime_error("composite_image radial scan requires positive radial_max_nm");
      }
      return sc;
   }

   // Parse one detector entry. A preset seeds the (energy, take-off-angle) box;
   // explicit fields or a working-distance/radii geometry then override it.
   static DetectorConfig readDetectorConfig(const RuntimeInput::JsonValue& json,
                                            double seThresholdEv)
   {
      if (!json.isObject()) throw std::runtime_error("composite_image detector must be an object");
      const double NO_MAX = 1.0e12;
      DetectorConfig d;
      d.name      = stringOr(json, "detector", "name");
      d.outputPgm = stringOr(json, "", "output_pgm", "pgm");
      d.eMinEv = 0.0; d.eMaxEv = NO_MAX; d.betaMinDeg = 0.0; d.betaMaxDeg = 90.0;

      std::string preset = normalizeModelName(stringOr(json, "", "preset", "type"));
      if (preset == "inlens_se" || preset == "tld_se") {
         d.eMaxEv = seThresholdEv; d.betaMinDeg = 0.0;  d.betaMaxDeg = 15.0;
      } else if (preset == "annular_bse" || preset == "bse") {
         d.eMinEv = seThresholdEv; d.betaMinDeg = 30.0; d.betaMaxDeg = 60.0;
      } else if (preset == "etd_se" || preset == "chamber_se") {
         d.eMaxEv = seThresholdEv; d.betaMinDeg = 30.0; d.betaMaxDeg = 70.0;
      } else if (!preset.empty()) {
         throw std::runtime_error("composite_image detector unknown preset: " + preset);
      }

      d.eMinEv = numberOr(json, d.eMinEv, "energy_min_ev", "e_min_ev");
      d.eMaxEv = numberOr(json, d.eMaxEv, "energy_max_ev", "e_max_ev");

      // Geometry: a detector of radius [r_in, r_out] at working distance WD maps,
      // in the field-free approximation, to take-off angles atan(r/WD).
      const RuntimeInput::JsonValue* wd = findAny(json, "working_distance_mm", "wd_mm");
      if (wd && wd->isNumber()) {
         double wdmm = wd->numberValue;
         double rin  = numberOr(json, 0.0, "inner_radius_mm", "r_inner_mm");
         double rout = numberOr(json, 0.0, "outer_radius_mm", "r_outer_mm");
         if (wdmm > 0.0 && rout > rin) {
            d.betaMinDeg = ::atan(rin  / wdmm) * 180.0 / Math2::PI;
            d.betaMaxDeg = ::atan(rout / wdmm) * 180.0 / Math2::PI;
         }
      }
      d.betaMinDeg = numberOr(json, d.betaMinDeg, "polar_min_deg", "beta_min_deg");
      d.betaMaxDeg = numberOr(json, d.betaMaxDeg, "polar_max_deg", "beta_max_deg");

      if (d.eMaxEv <= d.eMinEv)
         throw std::runtime_error("composite_image detector energy window invalid: " + d.name);
      if (d.betaMaxDeg <= d.betaMinDeg)
         throw std::runtime_error("composite_image detector angle window invalid: " + d.name);
      return d;
   }

   static std::vector<CompositeImageGPU::DetectorSpec>
   makeDetectorSpecs(const std::vector<DetectorConfig>& dets)
   {
      const double radPerDeg = Math2::PI / 180.0;
      std::vector<CompositeImageGPU::DetectorSpec> specs;
      specs.reserve(dets.size());
      for (size_t i = 0; i < dets.size(); ++i) {
         CompositeImageGPU::DetectorSpec s;
         s.eMinJ      = ToSI::eV(dets[i].eMinEv);
         s.eMaxJ      = ToSI::eV(dets[i].eMaxEv);
         s.betaMinRad = dets[i].betaMinDeg * radPerDeg;
         s.betaMaxRad = dets[i].betaMaxDeg * radPerDeg;
         specs.push_back(s);
      }
      return specs;
   }

   static PrecipitateConfig readPrecipitateConfig(const RuntimeInput::JsonValue& json,
                                                  const PrecipitateConfig& defaults)
   {
      PrecipitateConfig p = defaults;
      p.radiusNm      = numberOr(json, p.radiusNm,      "radius_nm",       "radius");
      p.centerXNm     = numberOr(json, p.centerXNm,     "center_x_nm",     "x_nm");
      p.centerYNm     = numberOr(json, p.centerYNm,     "center_y_nm",     "y_nm");
      p.centerDepthNm = numberOr(json, p.centerDepthNm, "center_depth_nm", "depth_nm");
      p.isVoid        = json.boolOr("void", p.isVoid);
      return p;
   }

   // Sphere surfaces must never intersect: the GPU transport exits a precipitate
   // through the root of its own sphere only, so overlapping spheres would
   // corrupt region bookkeeping. r65gen decks guarantee >= 1 nm gaps by RSA
   // construction; this uniform-grid check (O(N) expected) catches bad decks at
   // parse time instead.
   static void assertNoSphereOverlaps(const std::vector<PrecipitateConfig>& spheres)
   {
      if (spheres.size() < 2) return;
      double rMax = 0.0;
      for (size_t i = 0; i < spheres.size(); ++i) rMax = std::max(rMax, spheres[i].radiusNm);
      const double cell = std::max(2.0 * rMax, 1.0);
      std::map<std::tuple<long, long, long>, std::vector<size_t>> grid;
      for (size_t i = 0; i < spheres.size(); ++i) {
         const PrecipitateConfig& a = spheres[i];
         const long ix = (long)std::floor(a.centerXNm / cell);
         const long iy = (long)std::floor(a.centerYNm / cell);
         const long iz = (long)std::floor(a.centerDepthNm / cell);
         for (long dx = -1; dx <= 1; ++dx)
            for (long dy = -1; dy <= 1; ++dy)
               for (long dz = -1; dz <= 1; ++dz) {
                  auto it = grid.find(std::make_tuple(ix + dx, iy + dy, iz + dz));
                  if (it == grid.end()) continue;
                  for (size_t k = 0; k < it->second.size(); ++k) {
                     const PrecipitateConfig& b = spheres[it->second[k]];
                     const double ddx = a.centerXNm - b.centerXNm;
                     const double ddy = a.centerYNm - b.centerYNm;
                     const double ddz = a.centerDepthNm - b.centerDepthNm;
                     const double rsum = a.radiusNm + b.radiusNm;
                     if (ddx * ddx + ddy * ddy + ddz * ddz < rsum * rsum)
                        throw std::runtime_error(
                           "composite_image precipitates overlap (entries " +
                           std::to_string(it->second[k]) + " and " + std::to_string(i) +
                           "): sphere surfaces must not intersect");
                  }
               }
         grid[std::make_tuple(ix, iy, iz)].push_back(i);
      }
   }

   static CompositeImageConfig readConfig(const RuntimeInput::JsonValue& json)
   {
      if (!json.isObject()) throw std::runtime_error("composite_image config must be a JSON object");

      CompositeImageConfig config = defaultConfig();
      const RuntimeInput::JsonValue* params = json.find("parameters");
      const RuntimeInput::JsonValue& src = (params && params->isObject()) ? *params : json;

      config.name                 = stringOr(src, config.name,               "name");
      config.outputCsv            = stringOr(src, config.outputCsv,          "output_csv",    "output");
      config.outputPgmSE          = stringOr(src, config.outputPgmSE,        "output_pgm_se", "output_pgm");
      config.outputPgmSE1         = stringOr(src, config.outputPgmSE1,       "output_pgm_se1");
      config.outputPgmSE2         = stringOr(src, config.outputPgmSE2,       "output_pgm_se2");
      config.outputPgmBSE         = stringOr(src, config.outputPgmBSE,       "output_pgm_bse");
      config.trajectoriesPerPixel = (int)numberOr(src, (double)config.trajectoriesPerPixel,
                                                  "trajectories_per_pixel", "trajectories");
      config.beamEnergyEv         = numberOr(src, config.beamEnergyEv,       "beam_energy_ev", "beam_energy");
      config.beamSizeNm           = numberOr(src, config.beamSizeNm,         "beam_size_nm",   "beam_sigma_nm");
      config.seThresholdEv        = numberOr(src, config.seThresholdEv,
                                             "secondary_electron_threshold_ev", "se_threshold_ev");
      config.histogramBinSizeEv   = numberOr(src, config.histogramBinSizeEv,
                                             "histogram_bin_size_ev", "bin_size_ev");
      config.backend              = normalizeModelName(stringOr(src, config.backend, "backend", "execution_backend"));
      config.trackSecondaries     = src.boolOr("track_secondaries", config.trackSecondaries);
      config.rngSeed              = (unsigned long long)numberOr(src, (double)config.rngSeed, "rng_seed", "seed");

      const RuntimeInput::JsonValue* scattering = src.find("scattering");
      if (scattering) {
         if (!scattering->isObject()) throw std::runtime_error("composite_image scattering must be an object");
         config.scattering.elastic   = normalizeModelName(stringOr(*scattering, config.scattering.elastic,   "elastic"));
         config.scattering.inelastic = normalizeModelName(stringOr(*scattering, config.scattering.inelastic, "inelastic"));
         config.scattering.csd       = normalizeModelName(stringOr(*scattering, config.scattering.csd,       "csd"));
         config.scattering.barrier   = normalizeModelName(stringOr(*scattering, config.scattering.barrier,   "barrier"));
      }
      validateScatteringConfig(config.scattering);

      const RuntimeInput::JsonValue* matrixJson = findAny(src, "matrix_phase", "matrix");
      if (matrixJson) config.matrixPhase = readPhaseConfig(*matrixJson, "matrix_phase");

      const RuntimeInput::JsonValue* precipPhaseJson = findAny(src, "precipitate_phase", "precipitate_material");
      if (precipPhaseJson) config.precipitatePhase = readPhaseConfig(*precipPhaseJson, "precipitate_phase");

      const RuntimeInput::JsonValue* precipJson = findAny(src, "precipitate", "sphere");
      if (precipJson && precipJson->isObject())
         config.precipitates.assign(1, readPrecipitateConfig(*precipJson, config.precipitates[0]));

      // Multi-sphere form (schema v1, r65gen export): replaces the whole list.
      const RuntimeInput::JsonValue* precipsJson = src.find("precipitates");
      if (precipsJson) {
         if (!precipsJson->isArray())
            throw std::runtime_error("composite_image precipitates must be an array");
         PrecipitateConfig defaults;
         defaults.radiusNm = 0.0;   // every entry must carry its own radius
         defaults.centerXNm = defaults.centerYNm = defaults.centerDepthNm = 0.0;
         defaults.isVoid = false;
         config.precipitates.clear();
         for (size_t i = 0; i < precipsJson->arrayValue.size(); ++i) {
            const RuntimeInput::JsonValue& entry = precipsJson->arrayValue[i];
            if (!entry.isObject())
               throw std::runtime_error("composite_image precipitates[" + std::to_string(i) +
                                        "] must be an object");
            config.precipitates.push_back(readPrecipitateConfig(entry, defaults));
         }
      }

      const RuntimeInput::JsonValue* scanJson = src.find("scan");
      if (scanJson) config.scan = readScanConfig(*scanJson, config.scan);

      const RuntimeInput::JsonValue* slJson = findAny(src, "surface_layer", "surface_film", "contamination_layer");
      if (slJson && slJson->isObject()) {
         config.surfaceLayer.enabled     = true;
         config.surfaceLayer.thicknessNm = requireNumber(*slJson, "thickness_nm");
         config.surfaceLayer.phase       = readPhaseConfig(*slJson, "surface_layer");
      }

      const RuntimeInput::JsonValue* histJson = findAny(src, "escape_histogram", "detector_histogram");
      if (histJson && histJson->isObject()) {
         config.escapeHistEnabled   = histJson->boolOr("enabled", true);
         config.escapeHistAngleBins = (int)numberOr(*histJson, (double)config.escapeHistAngleBins,
                                                     "angle_bins", "polar_bins");
         config.escapeHistOutput    = stringOr(*histJson, config.escapeHistOutput, "output", "output_base");
         config.escapeHistRadialBins   = (int)numberOr(*histJson, (double)config.escapeHistRadialBins, "radial_bins");
         config.escapeHistRadialMaxNm  = numberOr(*histJson, config.escapeHistRadialMaxNm, "radial_max_nm");
      }

      const RuntimeInput::JsonValue* detsJson = src.find("detectors");
      if (detsJson) {
         if (!detsJson->isArray()) throw std::runtime_error("composite_image detectors must be an array");
         for (size_t i = 0; i < detsJson->arrayValue.size(); ++i)
            config.detectors.push_back(readDetectorConfig(detsJson->arrayValue[i], config.seThresholdEv));
      }

      const RuntimeInput::JsonValue* tcJson = findAny(src, "trajectory_capture", "trajectory_viz");
      if (tcJson && tcJson->isObject()) {
         config.tcEnabled           = tcJson->boolOr("enabled", true);
         config.tcMaxFullPaths      = (int)numberOr(*tcJson, (double)config.tcMaxFullPaths, "max_full_paths", "max_paths");
         config.tcMaxEscapes        = (int)numberOr(*tcJson, (double)config.tcMaxEscapes, "max_escape_events", "max_escapes");
         config.tcStepStride        = (int)numberOr(*tcJson, (double)config.tcStepStride, "step_stride");
         config.tcRecordSecondaries = tcJson->boolOr("record_secondaries", config.tcRecordSecondaries);
         config.tcIncludeAbsorbed   = tcJson->boolOr("include_absorbed", config.tcIncludeAbsorbed);
         config.tcOutput            = stringOr(*tcJson, config.tcOutput, "output", "output_base");
      }

      if (config.trajectoriesPerPixel <= 0) throw std::runtime_error("composite_image trajectories_per_pixel must be positive");
      if (config.beamEnergyEv <= 0.0)       throw std::runtime_error("composite_image beam_energy_ev must be positive");
      if (config.scan.nxPixels <= 0 || config.scan.nyPixels <= 0)
         throw std::runtime_error("composite_image scan nx_pixels and ny_pixels must be positive");
      if (!config.scan.radial && config.scan.halfWidthNm <= 0.0)
         throw std::runtime_error("composite_image scan half_width_nm must be positive");
      for (size_t i = 0; i < config.precipitates.size(); ++i) {
         const PrecipitateConfig& p = config.precipitates[i];
         if (p.radiusNm <= 0.0)
            throw std::runtime_error("composite_image precipitates[" + std::to_string(i) +
                                     "] radius_nm must be positive");
         if (p.centerDepthNm <= -p.radiusNm)
            throw std::runtime_error("composite_image precipitates[" + std::to_string(i) +
                                     "] center_depth_nm must be > -radius_nm "
                                     "(sphere is entirely above the surface)");
      }
      assertNoSphereOverlaps(config.precipitates);
      if (config.surfaceLayer.enabled && config.surfaceLayer.thicknessNm <= 0.0)
         throw std::runtime_error("composite_image surface_layer thickness_nm must be positive");
      if (config.backend != "auto" && config.backend != "cpu" && config.backend != "gpu")
         throw std::runtime_error("composite_image backend must be auto, cpu, or gpu");
      if (config.escapeHistEnabled && config.escapeHistAngleBins < 1)
         throw std::runtime_error("composite_image escape_histogram angle_bins must be positive");

      // Trajectory capture is a single-probe, CPU-only diagnostic. Normalize the
      // config so it runs one trajectory batch at the scan center on the CPU.
      if (config.tcEnabled) {
         if (config.tcStepStride < 1)   config.tcStepStride = 1;
         if (config.tcMaxFullPaths < 0) config.tcMaxFullPaths = 0;
         if (config.tcMaxEscapes < 0)   config.tcMaxEscapes = 0;
         if (config.tcOutput.empty()) {
            std::string base = config.outputCsv;
            size_t dot = base.find_last_of('.');
            if (dot != std::string::npos) base = base.substr(0, dot);
            config.tcOutput = base;
         }
         if (config.backend == "gpu")
            printf("  NOTE: trajectory_capture forces the CPU backend (the GPU kernel does not stream per-step paths)\n");
         config.backend = "cpu";
         if (config.scan.nxPixels != 1 || config.scan.nyPixels != 1) {
            printf("  NOTE: trajectory_capture uses a single probe at scan center (%.1f, %.1f) nm; forcing a 1x1 scan\n",
                   config.scan.centerXNm, config.scan.centerYNm);
            config.scan.nxPixels = 1;
            config.scan.nyPixels = 1;
         }
      }

      return config;
   }

   // Write a P2 (ASCII grayscale) PGM file.  values is row-major [row][col] = 0..maxval.
   static void writePGM(const std::string& filename,
                        const std::vector<std::vector<double>>& values,
                        int nyRows, int nxCols)
   {
      double minV = values[0][0], maxV = values[0][0];
      for (int row = 0; row < nyRows; ++row)
         for (int col = 0; col < nxCols; ++col) {
            if (values[row][col] < minV) minV = values[row][col];
            if (values[row][col] > maxV) maxV = values[row][col];
         }

      std::ofstream f(filename.c_str());
      if (!f.good()) {
         printf("WARNING: could not open PGM output file: %s\n", filename.c_str());
         return;
      }
      f << "P2\n# CudaMONSEL composite_image\n" << nxCols << " " << nyRows << "\n255\n";
      double range = (maxV > minV) ? (maxV - minV) : 1.0;
      for (int row = 0; row < nyRows; ++row) {
         for (int col = 0; col < nxCols; ++col) {
            int pixel = (int)(255.0 * (values[row][col] - minV) / range + 0.5);
            if (pixel < 0) pixel = 0;
            if (pixel > 255) pixel = 255;
            f << pixel;
            if (col < nxCols - 1) f << ' ';
         }
         f << '\n';
      }
   }

   static void computeWeightFractions(const PhaseConfig& phase, std::vector<double>& weightFractions)
   {
      weightFractions.assign(phase.elements.size(), 0.0);
      double totalWeight = 0.0;
      for (size_t i = 0; i < phase.elements.size(); ++i)
         totalWeight += phase.fractions[i] * phase.elements[i]->getAtomicWeight();

      if (totalWeight <= 0.0)
         throw std::runtime_error("composite_image GPU material has non-positive total atomic weight: " + phase.name);

      for (size_t i = 0; i < phase.elements.size(); ++i)
         weightFractions[i] = phase.fractions[i] * phase.elements[i]->getAtomicWeight() / totalWeight;
   }

   static int ensureGpuElement(
      std::map<int, int>& elementIndex,
      std::vector<CompositeImageGPU::ElemTableGPU>& elems,
      const ElementT* element)
   {
      int z = element->getAtomicNumber();
      std::map<int, int>::const_iterator found = elementIndex.find(z);
      if (found != elementIndex.end())
         return found->second;

      CompositeImageGPU::ElemTableGPU table;
      std::memset(&table, 0, sizeof(table));
      table.Z = z;
      table.Zp17 = std::pow((double)z, 1.7);
      table.Zp2 = std::pow((double)z, 2.0);
      table.Zp3 = std::pow((double)z, 3.0);
      table.extraBelowE = ToSI::eV(50.0);

      const NISTMottScatteringAngle::NISTMottScatteringAngle& msa =
         NISTMottScatteringAngle::getNISTMSA(z);
      const VectorXd& spwem = msa.getSpwem();
      const MatrixXd& x1 = msa.getX1();

      if ((int)spwem.size() != CompositeImageGPU::SPWEM_LEN || (int)x1.size() != CompositeImageGPU::SPWEM_LEN)
         throw std::runtime_error("Unexpected NIST Mott table shape for Z=" + std::to_string(z));

      for (int i = 0; i < CompositeImageGPU::SPWEM_LEN; ++i) {
         table.spwem[i] = spwem[i];
         if ((int)x1[i].size() != CompositeImageGPU::X1_LEN)
            throw std::runtime_error("Unexpected NIST Mott x1 table shape for Z=" + std::to_string(z));
         for (int j = 0; j < CompositeImageGPU::X1_LEN; ++j)
            table.x1[i * CompositeImageGPU::X1_LEN + j] = x1[i][j];
      }

      const RandomizedScatterT& nist = NISTMottRS::Factory.get(*element);
      const BrowningEmpiricalCrossSection::BrowningEmpiricalCrossSection& browning =
         BrowningEmpiricalCrossSection::getBECS(z);
      table.MottXSatMin = nist.totalCrossSection(table.extraBelowE);
      table.sfBrowning = table.MottXSatMin / browning.totalCrossSection(table.extraBelowE);

      int index = (int)elems.size();
      elems.push_back(table);
      elementIndex[z] = index;
      return index;
   }

   static double gpuMeanIonizationPotential(const ElementT* element)
   {
      double z = (double)element->getAtomicNumber();
      if (z < 13.0)
         return ToSI::eV(11.5 * z);
      return ToSI::eV(9.76 * z + 58.8 * std::pow(z, -0.19));
   }

   static CompositeImageGPU::MatGPU makeVacuumGpuMaterial()
   {
      CompositeImageGPU::MatGPU mat;
      std::memset(&mat, 0, sizeof(mat));
      mat.minEtrack = -INFINITY;
      mat.isVacuum = true;
      return mat;
   }

   static CompositeImageGPU::MatGPU makeGpuMaterial(
      const PhaseConfig& phase,
      std::map<int, int>& elementIndex,
      std::vector<CompositeImageGPU::ElemTableGPU>& elems)
   {
      if ((int)phase.elements.size() > CompositeImageGPU::MAX_MAT_ELEM)
         throw std::runtime_error("composite_image GPU supports at most 8 elements per material: " + phase.name);

      CompositeImageGPU::MatGPU mat;
      std::memset(&mat, 0, sizeof(mat));
      mat.isVacuum = false;
      mat.nElems = (int)phase.elements.size();
      mat.nCSD = mat.nElems;
      mat.densityNa = phase.density * PhysicalConstants::AvagadroNumber;
      mat.energySEgen = ToSI::eV(phase.energySEgen);
      mat.eFermi = ToSI::eV(phase.efermi);
      mat.energyCBbottom = ToSI::eV(-phase.workfun - phase.efermi);
      mat.minEtrack = std::max(-mat.energyCBbottom, 0.0);
      mat.breakE = ToSI::eV(phase.breakEeV);
      mat.bhplus1eV = mat.minEtrack + ToSI::eV(1.0);
      if (mat.breakE < mat.bhplus1eV)
         mat.breakE = mat.bhplus1eV;

      std::vector<double> wf;
      computeWeightFractions(phase, wf);
      for (int i = 0; i < mat.nElems; ++i) {
         const ElementT* element = phase.elements[i];
         mat.elemIdx[i] = ensureGpuElement(elementIndex, elems, element);
         mat.scalefactor[i] = 1000.0 * wf[i] / element->getAtomicWeight();
         mat.recipJ[i] = 1.166 / gpuMeanIonizationPotential(element);
         mat.betaJL[i] = 1.0 - (mat.recipJ[i] * mat.bhplus1eV);
         mat.coefJL[i] = 2.01507E-28 * phase.density * wf[i] *
            element->getAtomicNumber() / element->getAtomicWeight();
      }

      mat.gammaN = 0.0;
      for (int i = 0; i < mat.nCSD; ++i)
         mat.gammaN += mat.coefJL[i] * std::log((mat.recipJ[i] * mat.breakE) + mat.betaJL[i]);
      mat.gammaN /= std::pow(mat.breakE, 3.5);
      return mat;
   }

   // Build the uniform grid over the sphere-populated AABB (CSR cell lists).
   // Cell size follows the median radius (registration in every overlapped
   // cell keeps this correct even when radii span decades); total cell count
   // is capped so degenerate inputs cannot exhaust memory.
   static void buildSphereGrid(CompositeImageGPU::GPURunConfig& gpu)
   {
      auto& g = gpu.geom;
      const size_t n = gpu.spheres.size();
      g.nSpheres = (int)n;
      g.spheres = nullptr;   // device pointers patched inside CompositeImageGPU::run
      g.cellStart = nullptr;
      g.cellItems = nullptr;
      if (n == 0) {
         g.gridOx = g.gridOy = g.gridOz = 0.0;
         g.cellInv = 1.0;
         g.ncx = g.ncy = g.ncz = 0;
         gpu.cellStart.assign(1, 0);
         gpu.cellItems.clear();
         return;
      }

      double lo[3] = { 1.0e300, 1.0e300, 1.0e300 };
      double hi[3] = { -1.0e300, -1.0e300, -1.0e300 };
      std::vector<double> radii(n);
      for (size_t i = 0; i < n; ++i) {
         const CompositeImageGPU::SphereGPU& s = gpu.spheres[i];
         const double c[3] = { s.x, s.y, s.z };
         for (int ax = 0; ax < 3; ++ax) {
            lo[ax] = std::min(lo[ax], c[ax] - s.r);
            hi[ax] = std::max(hi[ax], c[ax] + s.r);
         }
         radii[i] = s.r;
      }
      const double pad = 1.0e-9;
      for (int ax = 0; ax < 3; ++ax) { lo[ax] -= pad; hi[ax] += pad; }

      std::nth_element(radii.begin(), radii.begin() + n / 2, radii.end());
      double cell = std::max(4.0 * radii[n / 2], 1.0e-9);
      long long ncx, ncy, ncz;
      for (;;) {
         ncx = std::max(1LL, (long long)std::ceil((hi[0] - lo[0]) / cell));
         ncy = std::max(1LL, (long long)std::ceil((hi[1] - lo[1]) / cell));
         ncz = std::max(1LL, (long long)std::ceil((hi[2] - lo[2]) / cell));
         if (ncx * ncy * ncz <= 2000000LL) break;
         cell *= 1.26;   // ~2x cell volume per iteration
      }
      g.gridOx = lo[0]; g.gridOy = lo[1]; g.gridOz = lo[2];
      g.cellInv = 1.0 / cell;
      g.ncx = (int)ncx; g.ncy = (int)ncy; g.ncz = (int)ncz;

      // CSR: count per cell, prefix-sum, fill.
      const size_t nCells = (size_t)(ncx * ncy * ncz);
      std::vector<int> counts(nCells, 0);
      auto cellRange = [&](const CompositeImageGPU::SphereGPU& s, int r0[3], int r1[3]) {
         const double c[3] = { s.x, s.y, s.z };
         const double o[3] = { g.gridOx, g.gridOy, g.gridOz };
         const int    nc[3] = { g.ncx, g.ncy, g.ncz };
         for (int ax = 0; ax < 3; ++ax) {
            r0[ax] = std::max(0, (int)std::floor((c[ax] - s.r - o[ax]) * g.cellInv));
            r1[ax] = std::min(nc[ax] - 1, (int)std::floor((c[ax] + s.r - o[ax]) * g.cellInv));
         }
      };
      for (size_t i = 0; i < n; ++i) {
         int r0[3], r1[3];
         cellRange(gpu.spheres[i], r0, r1);
         for (int iz = r0[2]; iz <= r1[2]; ++iz)
            for (int iy = r0[1]; iy <= r1[1]; ++iy)
               for (int ix = r0[0]; ix <= r1[0]; ++ix)
                  ++counts[((size_t)iz * g.ncy + iy) * g.ncx + ix];
      }
      gpu.cellStart.assign(nCells + 1, 0);
      for (size_t c = 0; c < nCells; ++c)
         gpu.cellStart[c + 1] = gpu.cellStart[c] + counts[c];
      gpu.cellItems.assign(gpu.cellStart[nCells], 0);
      std::vector<int> cursor(gpu.cellStart.begin(), gpu.cellStart.end() - 1);
      for (size_t i = 0; i < n; ++i) {
         int r0[3], r1[3];
         cellRange(gpu.spheres[i], r0, r1);
         for (int iz = r0[2]; iz <= r1[2]; ++iz)
            for (int iy = r0[1]; iy <= r1[1]; ++iy)
               for (int ix = r0[0]; ix <= r1[0]; ++ix)
                  gpu.cellItems[cursor[((size_t)iz * g.ncy + iy) * g.ncx + ix]++] = (int)i;
      }
   }

   static CompositeImageGPU::GPURunConfig makeGpuRunConfig(
      const CompositeImageConfig& config,
      const PhaseConfig& matrix,
      const PhaseConfig& precip,
      bool hasSL,
      double slThicknessM,
      const std::vector<std::array<double, 3>>& sphereCenters,
      const std::vector<double>& sphereRadii,
      double beamE,
      double beamsize,
      double beamStartZ,
      int nx,
      int ny,
      double cx,
      double cy,
      double hw,
      double dxm,
      double dym)
   {
      CompositeImageGPU::GPURunConfig gpu;
      std::map<int, int> elementIndex;

      // All spheres share one material slot; mixed void/solid decks are routed
      // to the CPU backend by the caller (runImage dispatch).
      bool allVoid = !config.precipitates.empty();
      for (size_t i = 0; i < config.precipitates.size(); ++i)
         allVoid = allVoid && config.precipitates[i].isVoid;

      gpu.elems.clear();
      gpu.mats[0] = makeVacuumGpuMaterial();
      gpu.mats[1] = hasSL ? makeGpuMaterial(config.surfaceLayer.phase, elementIndex, gpu.elems) : makeVacuumGpuMaterial();
      gpu.mats[2] = makeGpuMaterial(matrix, elementIndex, gpu.elems);
      gpu.mats[3] = allVoid
         ? makeVacuumGpuMaterial()                                  // etched-out voids: no scatter inside
         : makeGpuMaterial(precip, elementIndex, gpu.elems);

      gpu.spheres.resize(sphereCenters.size());
      for (size_t i = 0; i < sphereCenters.size(); ++i) {
         gpu.spheres[i].x = sphereCenters[i][0];
         gpu.spheres[i].y = sphereCenters[i][1];
         gpu.spheres[i].z = sphereCenters[i][2];
         gpu.spheres[i].r = sphereRadii[i];
      }
      buildSphereGrid(gpu);
      gpu.geom.spheresAreVoid = allVoid;
      gpu.geom.slThick = slThicknessM;
      gpu.geom.hasSL = hasSL;
      if (gpu.geom.nSpheres > 1)
         printf("  GPU sphere grid: %d spheres, %dx%dx%d cells, %zu references\n",
                gpu.geom.nSpheres, gpu.geom.ncx, gpu.geom.ncy, gpu.geom.ncz,
                gpu.cellItems.size());

      gpu.nx = nx;
      gpu.ny = ny;
      gpu.pixelX.resize(nx * ny);
      gpu.pixelY.resize(nx * ny);
      if (config.scan.radial) {
         // 1-D radial line: nx beam positions at (r, 0), r in [0, radialMaxNm].
         const double dr = (nx > 1) ? (config.scan.radialMaxNm * 1.e-9 / (nx - 1)) : 0.0;
         for (int col = 0; col < nx; ++col) {
            gpu.pixelX[col] = col * dr;
            gpu.pixelY[col] = 0.0;
         }
      }
      else {
         for (int row = 0; row < ny; ++row) {
            for (int col = 0; col < nx; ++col) {
               int idx = row * nx + col;
               gpu.pixelX[idx] = cx - hw + col * dxm;
               gpu.pixelY[idx] = cy + hw - row * dym;
            }
         }
      }

      gpu.beamE = beamE;
      gpu.beamSizeM = beamsize;
      gpu.beamStartZ = beamStartZ;
      gpu.trajPerPixel = config.trajectoriesPerPixel;
      gpu.seThresholdJ = ToSI::eV(config.seThresholdEv);
      gpu.seed = config.rngSeed;
      gpu.trackSecondaries = config.trackSecondaries;

      gpu.histEnabled    = config.escapeHistEnabled;
      gpu.histNBbins     = config.escapeHistAngleBins;
      gpu.histEbinWidthJ = ToSI::eV(config.histogramBinSizeEv);
      gpu.histNEbins     = (gpu.histEbinWidthJ > 0.0) ? (int)(beamE / gpu.histEbinWidthJ) : 0;
      if (gpu.histEnabled && gpu.histNEbins < 1) gpu.histNEbins = 1;

      gpu.detectors = makeDetectorSpecs(config.detectors);

      gpu.radialNBins = config.escapeHistRadialBins;
      gpu.radialMaxM  = (config.escapeHistRadialMaxNm > 0.0)
         ? config.escapeHistRadialMaxNm * 1.e-9 : hw;   // default: scan half-width
      return gpu;
   }

   // Write the per-pixel (escape energy x take-off angle) histogram as a raw
   // int32 .bin plus a .json sidecar describing dimensions and bin edges.
   // Layout: counts[((row*nx + col) * nEbins + ie) * nBbins + ib], row 0 = +half_width.
   static void writeEscapeHistogram(const std::string& base,
                                    const CompositeImageGPU::GPUOutput& out,
                                    const CompositeImageConfig& config,
                                    int nx, int ny, double cx, double cy, double hw)
   {
      const std::string binPath  = base + ".bin";
      const std::string jsonPath = base + ".json";

      std::ofstream bin(binPath.c_str(), std::ios::binary);
      if (!bin.good()) {
         printf("WARNING: could not open escape-histogram bin file: %s\n", binPath.c_str());
         return;
      }
      bin.write(reinterpret_cast<const char*>(out.escapeHist.data()),
                (std::streamsize)(out.escapeHist.size() * sizeof(int)));
      bin.close();

      const double ebinEv     = config.histogramBinSizeEv;
      const double betaBinDeg = (out.histNBbins > 0) ? (90.0 / out.histNBbins) : 0.0;
      std::ofstream js(jsonPath.c_str());
      if (!js.good()) {
         printf("WARNING: could not open escape-histogram json file: %s\n", jsonPath.c_str());
         return;
      }
      js << "{\n"
         << "  \"format\": \"cudamonsel_escape_histogram_v2\",\n"
         << "  \"dtype\": \"int32\",\n"
         << "  \"order\": \"[pixel][type][energy_bin][angle_bin]; pixel = row*nx + col; row 0 = +half_width (top)\",\n"
         << "  \"type_bins\": " << out.histNTypes << ",\n"
         << "  \"type_order\": [\"SE1\", \"SE2\", \"other\"],\n"
         << "  \"nx\": " << nx << ",\n"
         << "  \"ny\": " << ny << ",\n"
         << "  \"energy_bins\": " << out.histNEbins << ",\n"
         << "  \"angle_bins\": " << out.histNBbins << ",\n"
         << "  \"energy_bin_width_ev\": " << ebinEv << ",\n"
         << "  \"energy_max_ev\": " << (out.histNEbins * ebinEv) << ",\n"
         << "  \"beam_energy_ev\": " << config.beamEnergyEv << ",\n"
         << "  \"se_threshold_ev\": " << config.seThresholdEv << ",\n"
         << "  \"angle_bin_width_deg\": " << betaBinDeg << ",\n"
         << "  \"angle_max_deg\": 90.0,\n"
         << "  \"angle_convention\": \"beta = take-off polar angle from outward optic axis (-z); 0 deg = up the column, 90 deg = grazing\",\n"
         << "  \"trajectories_per_pixel\": " << config.trajectoriesPerPixel << ",\n"
         << "  \"half_width_nm\": " << (hw * 1.e9) << ",\n"
         << "  \"center_x_nm\": " << (cx * 1.e9) << ",\n"
         << "  \"center_y_nm\": " << (cy * 1.e9) << ",\n"
         << "  \"bin_file\": \"" << binPath << "\"\n"
         << "}\n";
      js.close();
      printf("CompositeImage: escape histogram --> %s (+ .json)\n", binPath.c_str()); fflush(stdout);
   }

   // Write the per-pixel radial escape-distance histogram (by type) as a raw
   // int32 .bin + .json sidecar. Layout: counts[(pixel*type_bins + t)*radial_bins + ir].
   static void writeRadialHistogram(const std::string& base,
                                    const CompositeImageGPU::GPUOutput& out,
                                    const CompositeImageConfig& config,
                                    int nx, int ny)
   {
      const std::string binPath  = base + "_radial.bin";
      const std::string jsonPath = base + "_radial.json";

      std::ofstream bin(binPath.c_str(), std::ios::binary);
      if (!bin.good()) {
         printf("WARNING: could not open radial-histogram bin file: %s\n", binPath.c_str());
         return;
      }
      bin.write(reinterpret_cast<const char*>(out.radialHist.data()),
                (std::streamsize)(out.radialHist.size() * sizeof(int)));
      bin.close();

      const double binNm = (out.radialNBins > 0) ? (out.radialMaxM * 1.e9 / out.radialNBins) : 0.0;
      std::ofstream js(jsonPath.c_str());
      if (!js.good()) {
         printf("WARNING: could not open radial-histogram json file: %s\n", jsonPath.c_str());
         return;
      }
      js << "{\n"
         << "  \"format\": \"cudamonsel_radial_histogram_v1\",\n"
         << "  \"dtype\": \"int32\",\n"
         << "  \"order\": \"[pixel][type][radial_bin]; pixel = row*nx + col; radius = |escape_xy - beam_center_xy|\",\n"
         << "  \"type_bins\": " << out.radialNTypes << ",\n"
         << "  \"type_order\": [\"SE1\", \"SE2\", \"other\"],\n"
         << "  \"nx\": " << nx << ",\n"
         << "  \"ny\": " << ny << ",\n"
         << "  \"radial_bins\": " << out.radialNBins << ",\n"
         << "  \"radial_max_nm\": " << (out.radialMaxM * 1.e9) << ",\n"
         << "  \"radial_bin_width_nm\": " << binNm << ",\n"
         << "  \"trajectories_per_pixel\": " << config.trajectoriesPerPixel << ",\n"
         << "  \"half_width_nm\": " << config.scan.halfWidthNm << ",\n"
         << "  \"bin_file\": \"" << binPath << "\"\n"
         << "}\n";
      js.close();
      printf("CompositeImage: radial histogram --> %s (+ .json)\n", binPath.c_str()); fflush(stdout);
   }

   static void runImage(const CompositeImageConfig& config)
   {
      PhaseConfig matrix = config.matrixPhase;
      PhaseConfig precip = config.precipitatePhase;
      bool   hasSL        = config.surfaceLayer.enabled;
      double slThicknessM = hasSL ? config.surfaceLayer.thicknessNm * 1.e-9 : 0.0;

      printf("\nCompositeImage: %s\n", config.name.c_str()); fflush(stdout);
      printf("  Matrix: %s  /  Precipitate: %s\n", matrix.name.c_str(), precip.name.c_str()); fflush(stdout);
      if (config.precipitates.size() == 1) {
         printf("  Precipitate: sphere r=%.1f nm, center depth=%.1f nm (x=%.1f, y=%.1f nm)\n",
                config.precipitates[0].radiusNm, config.precipitates[0].centerDepthNm,
                config.precipitates[0].centerXNm, config.precipitates[0].centerYNm);
      }
      else {
         double rMin = 1.0e300, rMax = 0.0, dMin = 1.0e300, dMax = -1.0e300;
         for (size_t i = 0; i < config.precipitates.size(); ++i) {
            const PrecipitateConfig& p = config.precipitates[i];
            rMin = std::min(rMin, p.radiusNm);  rMax = std::max(rMax, p.radiusNm);
            dMin = std::min(dMin, p.centerDepthNm);  dMax = std::max(dMax, p.centerDepthNm);
         }
         printf("  Precipitates: %zu spheres, r=%.1f..%.1f nm, center depth=%.1f..%.1f nm\n",
                config.precipitates.size(), rMin, rMax, dMin, dMax);
      }
      fflush(stdout);
      printf("  Scan: %dx%d pixels, ±%.1f nm FOV, %.0f eV beam, %d traj/pixel\n",
             config.scan.nxPixels, config.scan.nyPixels, config.scan.halfWidthNm,
             config.beamEnergyEv, config.trajectoriesPerPixel); fflush(stdout);
      if (hasSL)
         printf("  Surface layer: %s  %.3f nm  density=%.0f kg/m3  phi=%.2f eV\n",
                config.surfaceLayer.phase.name.c_str(), config.surfaceLayer.thicknessNm,
                config.surfaceLayer.phase.density, config.surfaceLayer.phase.workfun);
      fflush(stdout);

      // === Matrix material ===
      double matPotU = -matrix.workfun - matrix.efermi;
      CompositionT matComp;
      matComp.defineByMoleFraction(matrix.elements.data(), (int)matrix.elements.size(),
                                   matrix.fractions.data(), (int)matrix.fractions.size());
      SEmaterialT matMat(matComp, matrix.density);
      matMat.setWorkfunction(ToSI::eV(matrix.workfun));
      matMat.setBandgap(ToSI::eV(matrix.bandgap));
      matMat.setEnergyCBbottom(ToSI::eV(matPotU));

      // === Precipitate material ===
      double precPotU = -precip.workfun - precip.efermi;
      CompositionT precComp;
      precComp.defineByMoleFraction(precip.elements.data(), (int)precip.elements.size(),
                                    precip.fractions.data(), (int)precip.fractions.size());
      SEmaterialT precMat(precComp, precip.density);
      precMat.setWorkfunction(ToSI::eV(precip.workfun));
      precMat.setBandgap(ToSI::eV(precip.bandgap));
      precMat.setEnergyCBbottom(ToSI::eV(precPotU));

      // === Vacuum ===
      SEmaterialT vacMat;
      vacMat.setName("vacuum");

      // === Surface layer material (optional — e.g. carbon contamination) ===
      // Keep slCompPtr and slMatPtr alive for the whole parallel loop (shared read-only).
      std::unique_ptr<CompositionT> slCompPtr;
      std::unique_ptr<SEmaterialT>  slMatPtr;
      if (hasSL) {
         PhaseConfig sl = config.surfaceLayer.phase;   // copy so .data() is non-const
         slCompPtr = std::make_unique<CompositionT>();
         slCompPtr->defineByMoleFraction(sl.elements.data(), (int)sl.elements.size(),
                                         sl.fractions.data(), (int)sl.fractions.size());
         slMatPtr = std::make_unique<SEmaterialT>(*slCompPtr, sl.density);
         slMatPtr->setWorkfunction(ToSI::eV(sl.workfun));
         slMatPtr->setBandgap(ToSI::eV(sl.bandgap));
         slMatPtr->setEnergyCBbottom(ToSI::eV(-sl.workfun - sl.efermi));
      }

      // === Geometry constants (used to build per-thread regions in parallel loop) ===
      // NormalMultiPlaneShape stores its last-computed normal as mutable state, so it
      // cannot be shared across threads.
      const double origin[]     = { 0., 0., 0. };
      const double normalvec[]  = { 0., 0., -1. };
      const double surfacePos[] = { 0., 0.,  0. };
      // Sphere list in SI (meters); nSpheres == 1 is the legacy single-sphere
      // path (GPU-capable in this revision), > 1 runs on the CPU region graph.
      const size_t nSpheres = config.precipitates.size();
      std::vector<std::array<double, 3>> sphereCenters(nSpheres);
      std::vector<double>                sphereRadii(nSpheres);
      for (size_t s = 0; s < nSpheres; ++s) {
         sphereCenters[s] = { config.precipitates[s].centerXNm     * 1.e-9,
                              config.precipitates[s].centerYNm     * 1.e-9,
                              config.precipitates[s].centerDepthNm * 1.e-9 };
         sphereRadii[s]   = config.precipitates[s].radiusNm * 1.e-9;
      }
      // The GPU material model has one precipitate slot, so a deck mixing void
      // and solid spheres cannot run there (CPU assigns materials per sphere).
      bool anyVoid = false, allVoid = (nSpheres > 0);
      for (size_t s = 0; s < nSpheres; ++s) {
         anyVoid = anyVoid || config.precipitates[s].isVoid;
         allVoid = allVoid && config.precipitates[s].isVoid;
      }
      const bool mixedVoid = anyVoid && !allVoid;

      // === Beam parameters ===
      double beamE    = ToSI::eV(config.beamEnergyEv);
      double beamsize = config.beamSizeNm * 1.e-9;

      int    nbins   = (int)(config.beamEnergyEv / config.histogramBinSizeEv);
      if (nbins < 1) nbins = 1;

      // === Build scan grid ===
      int nx = config.scan.nxPixels;
      int ny = config.scan.nyPixels;
      double hw   = config.scan.halfWidthNm * 1.e-9;
      double cx   = config.scan.centerXNm  * 1.e-9;
      double cy   = config.scan.centerYNm  * 1.e-9;
      double dxm  = (nx > 1) ? (2.0 * hw / (nx - 1)) : 0.0;
      double dym  = (ny > 1) ? (2.0 * hw / (ny - 1)) : 0.0;
      double minBeamZ   = hasSL ? (-slThicknessM - 1.e-9) : -1.e-9;
      double beamStartZ = minBeamZ;
      for (size_t s = 0; s < nSpheres; ++s)
         beamStartZ = std::min(beamStartZ, (sphereCenters[s][2] - sphereRadii[s]) - 5.e-9);

      // Results: row 0 = +halfWidthNm (top of image), row ny-1 = -halfWidthNm (bottom).
      std::vector<std::vector<double>> seMap  (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> se1Map (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> se2Map (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> bseMap (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> genSeMap(ny, std::vector<double>(nx, 0.0));

      // Optional detector channels (same specs used by GPU and CPU paths).
      const int nDet = (int)config.detectors.size();
      std::vector<CompositeImageGPU::DetectorSpec> detSpecs = makeDetectorSpecs(config.detectors);
      std::vector<std::vector<std::vector<double>>> detMap(
         nDet, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));

      std::ofstream csvFile(config.outputCsv.c_str());
      if (!csvFile.good()) throw std::runtime_error("Unable to open composite_image output file: " + config.outputCsv);
      csvFile << "x_nm,y_nm,SE_yield,SE1_yield,SE2_yield,BSE_yield,total_yield";
      for (int d = 0; d < nDet; ++d) csvFile << ",det_" << config.detectors[d].name;
      csvFile << "\n";

      auto wallStart  = std::chrono::system_clock::now();
      int  totalPixels = nx * ny;
      int  pixelsDone  = 0;
      bool gpuUsed = false;

      if (mixedVoid && config.backend == "gpu")
         throw std::runtime_error("composite_image backend=gpu cannot mix void and solid "
                                  "precipitates (one GPU material slot) - use backend=cpu");
      if (mixedVoid && config.backend != "cpu") {
         printf("  NOTE: mixed void/solid precipitates -> CPU backend\n"); fflush(stdout);
      }

      if (config.backend != "cpu" && !mixedVoid) {
         if (CompositeImageGPU::isAvailable()) {
            printf("  Trying CUDA GPU backend\n"); fflush(stdout);
            try {
               CompositeImageGPU::GPURunConfig gpuCfg = makeGpuRunConfig(
                  config, matrix, precip, hasSL, slThicknessM, sphereCenters, sphereRadii,
                  beamE, beamsize, beamStartZ, nx, ny, cx, cy, hw, dxm, dym);
               CompositeImageGPU::GPUOutput gpuOut;
               if (CompositeImageGPU::run(gpuCfg, gpuOut)) {
                  for (int row = 0; row < ny; ++row) {
                     for (int col = 0; col < nx; ++col) {
                        int idx = row * nx + col;
                        seMap[row][col]  = gpuOut.seYield[idx];
                        se1Map[row][col] = gpuOut.se1Yield[idx];
                        se2Map[row][col] = gpuOut.se2Yield[idx];
                        bseMap[row][col] = gpuOut.bseYield[idx];
                        for (int d = 0; d < nDet && gpuOut.nDet == nDet; ++d)
                           detMap[d][row][col] = gpuOut.detYield[(size_t)idx * nDet + d];
                     }
                  }
                  gpuUsed = true;

                  // Diagnostic: mean SE generation rate and escape fraction
                  double meanGenSE = 0.0, meanSE = 0.0, meanBSE = 0.0;
                  for (int i = 0; i < totalPixels; ++i) {
                     meanGenSE += gpuOut.genSeYield[i];
                     meanSE    += gpuOut.seYield[i];
                     meanBSE   += gpuOut.bseYield[i];
                  }
                  meanGenSE /= totalPixels;
                  meanSE    /= totalPixels;
                  meanBSE   /= totalPixels;
                  printf("  CUDA GPU backend complete\n"
                         "    GPU mean SE yield=%.4f  BSE yield=%.4f  total=%.4f\n"
                         "    GPU mean genSE/traj=%.4f  SE escape ratio=%.4f\n",
                         meanSE, meanBSE, meanSE + meanBSE,
                         meanGenSE, meanGenSE > 0.0 ? meanSE / meanGenSE : 0.0);
                  fflush(stdout);

                  if ((config.escapeHistEnabled && !gpuOut.escapeHist.empty()) || !gpuOut.radialHist.empty()) {
                     std::string base = config.escapeHistOutput;
                     if (base.empty()) {
                        base = config.outputCsv;
                        size_t dot = base.find_last_of('.');
                        if (dot != std::string::npos) base = base.substr(0, dot);
                        base += "_eahist";
                     }
                     if (!gpuOut.escapeHist.empty())
                        writeEscapeHistogram(base, gpuOut, config, nx, ny, cx, cy, hw);
                     if (!gpuOut.radialHist.empty())
                        writeRadialHistogram(base, gpuOut, config, nx, ny);
                  }
                  else if (config.escapeHistEnabled) {
                     printf("  WARNING: escape_histogram enabled but GPU returned no histogram\n"); fflush(stdout);
                  }
               }
               else if (config.backend == "gpu") {
                  throw std::runtime_error("composite_image CUDA backend failed");
               }
               else {
                  printf("  CUDA GPU backend failed; falling back to CPU\n"); fflush(stdout);
               }
            }
            catch (const std::exception& ex) {
               if (config.backend == "gpu")
                  throw;
               printf("  CUDA GPU backend unavailable (%s); falling back to CPU\n", ex.what()); fflush(stdout);
            }
         }
         else if (config.backend == "gpu") {
            throw std::runtime_error("composite_image backend=gpu requested but no CUDA device is available");
         }
      }

      if (!gpuUsed) {
      printf("  Using %d OpenMP thread(s)\n", omp_get_max_threads()); fflush(stdout);

      #pragma omp parallel for schedule(dynamic) shared(seMap, bseMap, pixelsDone)
      for (int px = 0; px < totalPixels; ++px) {
         int    row = px / nx;
         int    col = px % nx;
         // Single-probe trajectory capture parks the beam exactly at the scan
         // center; the raster formula would otherwise land the 1x1 pixel at the
         // (cx-hw, cy+hw) corner.
         double x, y;
         if (config.scan.radial) {                     // 1-D radial line: (r, 0)
            double dr = (nx > 1) ? (config.scan.radialMaxNm * 1.e-9 / (nx - 1)) : 0.0;
            x = col * dr;
            y = 0.0;
         }
         else if (config.tcEnabled) { x = cx; y = cy; } // single probe at scan center
         else {
            x = cx - hw + col * dxm;
            y = cy + hw - row * dym;
         }

         // Per-worker scatter models.
         // MONSEL_MaterialScatterModel caches scatter rates in mutable member
         // state (cached_eK, cached_cumulativeScatterRate) and SelectableElasticSM
         // also caches cached_kE; neither is safe to share across threads.
         // Material property objects (matMat, precMat, vacMat) and the static
         // cross-section tables are shared read-only and are safe.
         SelectableElasticSMT         matElastic_t(matMat, NISTMottRS::Factory);
         JoyLuoNieminenCSDT           matCSD_t(matMat, ToSI::eV(matrix.breakEeV));
         FittedInelSMT                matInel_t(matMat, ToSI::eV(matrix.energySEgen), matCSD_t);
         ExpQMBarrierSMT              matBarrier_t(&matMat);
         MONSEL_MaterialScatterModelT matMSM_t(&matMat, &matBarrier_t);
         matMSM_t.addScatterMechanism(&matElastic_t);
         matMSM_t.addScatterMechanism(&matInel_t);
         matMSM_t.setCSD(&matCSD_t);

         SelectableElasticSMT         precElastic_t(precMat, NISTMottRS::Factory);
         JoyLuoNieminenCSDT           precCSD_t(precMat, ToSI::eV(precip.breakEeV));
         FittedInelSMT                precInel_t(precMat, ToSI::eV(precip.energySEgen), precCSD_t);
         ExpQMBarrierSMT              precBarrier_t(&precMat);
         MONSEL_MaterialScatterModelT precMSM_t(&precMat, &precBarrier_t);
         precMSM_t.addScatterMechanism(&precElastic_t);
         precMSM_t.addScatterMechanism(&precInel_t);
         precMSM_t.setCSD(&precCSD_t);

         ExpQMBarrierSMT              vacBarrier_t(&vacMat);
         MONSEL_MaterialScatterModelT vacMSM_t(&vacMat, &vacBarrier_t);

         // Per-thread surface layer scatter models (only constructed when hasSL).
         std::unique_ptr<SelectableElasticSMT>         slElastic_up;
         std::unique_ptr<JoyLuoNieminenCSDT>           slCSD_up;
         std::unique_ptr<FittedInelSMT>                slInel_up;
         std::unique_ptr<ExpQMBarrierSMT>              slBarrier_up;
         std::unique_ptr<MONSEL_MaterialScatterModelT> slMSM_up;
         if (hasSL) {
            const PhaseConfig& sl = config.surfaceLayer.phase;
            slElastic_up = std::make_unique<SelectableElasticSMT>(*slMatPtr, NISTMottRS::Factory);
            slCSD_up     = std::make_unique<JoyLuoNieminenCSDT>(*slMatPtr, ToSI::eV(sl.breakEeV));
            slInel_up    = std::make_unique<FittedInelSMT>(*slMatPtr, ToSI::eV(sl.energySEgen), *slCSD_up);
            slBarrier_up = std::make_unique<ExpQMBarrierSMT>(slMatPtr.get());
            slMSM_up     = std::make_unique<MONSEL_MaterialScatterModelT>(slMatPtr.get(), slBarrier_up.get());
            slMSM_up->addScatterMechanism(slElastic_up.get());
            slMSM_up->addScatterMechanism(slInel_up.get());
            slMSM_up->setCSD(slCSD_up.get());
         }

         // Per-thread geometry; NormalMultiPlaneShape has mutable stored-normal state.
         SphereT                chamberSphere_t(origin, MonteCarloSS::ChamberRadius);
         NullMaterialScatterModelT nullMSM_t;
         RegionT                chamber_t(nullptr, &nullMSM_t, &chamberSphere_t);
         chamber_t.updateMaterial(*chamber_t.getScatterModel(), vacMSM_t);

         // Optional surface layer: occupies the half-space z > -slThicknessM.
         // bulkRegion becomes a child of the surface layer rather than of chamber.
         double                 slSurfPos[] = { 0.0, 0.0, -slThicknessM };
         NormalMultiPlaneShapeT slSurface_t;
         // slPl_t must outlive slSurface_t: NormalMultiPlaneShape::contains(pos)
         // dereferences the stored plane pointer, so a plane scoped to the if-block
         // would dangle once the surface layer is actually used during tracing.
         PlaneT                 slPl_t(normalvec, 3, slSurfPos, 3);
         std::unique_ptr<RegionT> slRegion_up;
         if (hasSL) {
            slSurface_t.addPlane(slPl_t);
            slRegion_up = std::make_unique<RegionT>(&chamber_t, slMSM_up.get(), (NormalShapeT*)&slSurface_t);
         }

         NormalMultiPlaneShapeT surface_t;
         PlaneT                 pl_t(normalvec, 3, surfacePos, 3);
         surface_t.addPlane(pl_t);
         RegionT*               bulkParent = hasSL ? slRegion_up.get() : &chamber_t;
         RegionT                bulkRegion_t(bulkParent, &matMSM_t, (NormalShapeT*)&surface_t);

         // One subregion per precipitate sphere (all share the precipitate
         // material; void entries scatter as vacuum). The general region graph
         // handles any count; cost is O(nSpheres) per step on this backend.
         std::vector<std::unique_ptr<SphereT>> precipSpheres_t;
         std::vector<std::unique_ptr<RegionT>> precipRegions_t;
         precipSpheres_t.reserve(nSpheres);
         precipRegions_t.reserve(nSpheres);
         for (size_t s = 0; s < nSpheres; ++s) {
            precipSpheres_t.push_back(
               std::make_unique<SphereT>(sphereCenters[s].data(), sphereRadii[s]));
            precipRegions_t.push_back(std::make_unique<RegionT>(
               &bulkRegion_t,
               config.precipitates[s].isVoid ? &vacMSM_t : &precMSM_t,
               precipSpheres_t.back().get()));
         }

         // Start beam above: the sphere apex, and (when a surface layer is present)
         // above the top of that layer, so the initial region lookup is always vacuum.
         double        egCenter[] = { x, y, beamStartZ };
         GaussianBeamT eg_t(beamsize, beamE, origin);
         eg_t.setCenter(egCenter);

         MonteCarloSS::MonteCarloSS monte_t(&eg_t, &chamber_t, eg_t.createElectron());
         BackscatterStatsT          back_t(monte_t, nbins);
         SecondaryCountListener     secondary_t;
         DetectorCountListener      detector_t(monte_t, detSpecs);
         monte_t.addActionListener(back_t);
         monte_t.addActionListener(secondary_t);
         if (nDet > 0) monte_t.addActionListener(detector_t);

         // Single-probe trajectory/escape capture (CPU only; totalPixels == 1 here).
         std::unique_ptr<TrajectoryRecorder> traj_up;
         if (config.tcEnabled) {
            traj_up = std::make_unique<TrajectoryRecorder>(
               monte_t, ToSI::eV(config.seThresholdEv),
               config.tcMaxFullPaths, config.tcMaxEscapes, config.tcStepStride,
               config.tcRecordSecondaries, config.tcIncludeAbsorbed);
            monte_t.addActionListener(*traj_up);
         }

         monte_t.runMultipleTrajectories(config.trajectoriesPerPixel);

         if (config.tcEnabled) {
            monte_t.removeActionListener(*traj_up);
            traj_up->writeCsvs(config.tcOutput);
         }
         if (nDet > 0) monte_t.removeActionListener(detector_t);
         monte_t.removeActionListener(secondary_t);
         monte_t.removeActionListener(back_t);

         const HistogramT& hist    = back_t.backscatterEnergyHistogram();
         const HistogramT& se1hist = back_t.se1EnergyHistogram();
         const HistogramT& se2hist = back_t.se2EnergyHistogram();
         double ePerBin  = config.beamEnergyEv / hist.binCount();
         int    maxSEbin = (int)(config.seThresholdEv / ePerBin);
         int    totalSE = 0, totalSE1 = 0, totalSE2 = 0;
         for (int j = 0; j < maxSEbin && j < (int)hist.binCount(); ++j) {
            totalSE  += hist.counts(j);
            totalSE1 += se1hist.counts(j);
            totalSE2 += se2hist.counts(j);
         }

         double SEY  = (double)totalSE  / config.trajectoriesPerPixel;
         double SE1Y = (double)totalSE1 / config.trajectoriesPerPixel;
         double SE2Y = (double)totalSE2 / config.trajectoriesPerPixel;
         double BSEY = back_t.backscatterFraction() - SEY;
         double GenSEY = (double)secondary_t.generatedSecondaries() / config.trajectoriesPerPixel;

         seMap [row][col] = SEY;   // unique cell per thread, no race
         se1Map[row][col] = SE1Y;
         se2Map[row][col] = SE2Y;
         bseMap[row][col] = BSEY;
         genSeMap[row][col] = GenSEY;
         for (int d = 0; d < nDet; ++d)
            detMap[d][row][col] = (double)detector_t.counts()[d] / config.trajectoriesPerPixel;

         #pragma omp critical(progress)
         {
            ++pixelsDone;
            if (pixelsDone % 10 == 0 || pixelsDone == totalPixels) {
               printf("  pixel %d/%d  (x=%.1f nm, y=%.1f nm)  SE=%.4f  SE1=%.4f  SE2=%.4f  BSE=%.4f\n",
                      pixelsDone, totalPixels, x * 1.e9, y * 1.e9, SEY, SE1Y, SE2Y, BSEY);
               fflush(stdout);
            }
         }
      }
      }

      if (!gpuUsed) {
         double meanGenSE = 0.0, meanSE = 0.0;
         for (int row = 0; row < ny; ++row) {
            for (int col = 0; col < nx; ++col) {
               meanGenSE += genSeMap[row][col];
               meanSE    += seMap[row][col];
            }
         }
         meanGenSE /= totalPixels;
         meanSE    /= totalPixels;
         printf("  CPU mean genSE/traj=%.4f  SE escape ratio=%.4f\n",
                meanGenSE, meanGenSE > 0.0 ? meanSE / meanGenSE : 0.0);
         fflush(stdout);
      }

      auto wallEnd = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = wallEnd - wallStart;
      printf("\nCompositeImage: scan complete in %.1f s\n", elapsed.count()); fflush(stdout);

      // Write CSV in scan order (row-major) after the parallel section.
      const double radialDr = (config.scan.radial && nx > 1)
         ? (config.scan.radialMaxNm * 1.e-9 / (nx - 1)) : 0.0;
      for (int row = 0; row < ny; ++row) {
         for (int col = 0; col < nx; ++col) {
            double x     = config.scan.radial ? (col * radialDr) : (cx - hw + col * dxm);
            double y     = config.scan.radial ? 0.0 : (cy + hw - row * dym);
            double total = seMap[row][col] + bseMap[row][col];
            csvFile << x * 1.e9 << "," << y * 1.e9 << ","
                    << seMap[row][col]  << "," << se1Map[row][col] << ","
                    << se2Map[row][col] << "," << bseMap[row][col] << "," << total;
            for (int d = 0; d < nDet; ++d) csvFile << "," << detMap[d][row][col];
            csvFile << "\n";
         }
      }
      csvFile.close();

      if (!config.outputPgmSE.empty()) {
         writePGM(config.outputPgmSE, seMap, ny, nx);
         printf("CompositeImage: SE image  --> %s\n", config.outputPgmSE.c_str()); fflush(stdout);
      }
      if (!config.outputPgmSE1.empty()) {
         writePGM(config.outputPgmSE1, se1Map, ny, nx);
         printf("CompositeImage: SE1 image --> %s\n", config.outputPgmSE1.c_str()); fflush(stdout);
      }
      if (!config.outputPgmSE2.empty()) {
         writePGM(config.outputPgmSE2, se2Map, ny, nx);
         printf("CompositeImage: SE2 image --> %s\n", config.outputPgmSE2.c_str()); fflush(stdout);
      }
      if (!config.outputPgmBSE.empty()) {
         writePGM(config.outputPgmBSE, bseMap, ny, nx);
         printf("CompositeImage: BSE image --> %s\n", config.outputPgmBSE.c_str()); fflush(stdout);
      }
      for (int d = 0; d < nDet; ++d) {
         if (!config.detectors[d].outputPgm.empty()) {
            writePGM(config.detectors[d].outputPgm, detMap[d], ny, nx);
            printf("CompositeImage: detector '%s' image --> %s\n",
                   config.detectors[d].name.c_str(), config.detectors[d].outputPgm.c_str()); fflush(stdout);
         }
      }
      printf("CompositeImage: CSV data  --> %s\n", config.outputCsv.c_str()); fflush(stdout);
   }

   void run()
   {
      runImage(defaultConfig());
   }

   void run(const RuntimeInput::JsonValue& config)
   {
      runImage(readConfig(config));
   }
}
