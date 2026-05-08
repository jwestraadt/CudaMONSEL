// file: gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage.cu
//
// SEM image simulation for a composite material: one matrix phase with a single
// spherical precipitate embedded at a configurable depth.  The beam is scanned
// over a 2D (x, y) grid; SE and BSE yields are recorded per pixel and written
// to a CSV file and to PGM image files (one each for SE and BSE yield).
//
// Precipitate center_depth_nm = 0 places the centroid on the surface so the
// sphere is cut exactly in half: the upper hemisphere is in vacuum (no scatter)
// and the lower hemisphere is the precipitate phase.
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
#include <stdexcept>
#include <string>
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
   };

   struct ScanConfig
   {
      double centerXNm;    // scan center x (usually = precipitate centerXNm)
      double centerYNm;    // scan center y
      double halfWidthNm;  // half field-of-view (scan covers ±halfWidthNm in x and y)
      int    nxPixels;
      int    nyPixels;
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
      ScatteringConfig  scattering;
      PhaseConfig       matrixPhase;
      PhaseConfig       precipitatePhase;
      PrecipitateConfig precipitate;
      ScanConfig        scan;
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

      config.precipitate.radiusNm      = 30.0;
      config.precipitate.centerXNm     = 0.0;
      config.precipitate.centerYNm     = 0.0;
      config.precipitate.centerDepthNm = 0.0;   // centroid on surface

      config.scan.centerXNm  = 0.0;
      config.scan.centerYNm  = 0.0;
      config.scan.halfWidthNm = 90.0;           // 180 nm field of view
      config.scan.nxPixels   = 64;
      config.scan.nyPixels   = 64;

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
      return sc;
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
      if (precipJson && precipJson->isObject()) {
         config.precipitate.radiusNm      = numberOr(*precipJson, config.precipitate.radiusNm,      "radius_nm",       "radius");
         config.precipitate.centerXNm     = numberOr(*precipJson, config.precipitate.centerXNm,     "center_x_nm",     "x_nm");
         config.precipitate.centerYNm     = numberOr(*precipJson, config.precipitate.centerYNm,     "center_y_nm",     "y_nm");
         config.precipitate.centerDepthNm = numberOr(*precipJson, config.precipitate.centerDepthNm, "center_depth_nm", "depth_nm");
      }

      const RuntimeInput::JsonValue* scanJson = src.find("scan");
      if (scanJson) config.scan = readScanConfig(*scanJson, config.scan);

      if (config.trajectoriesPerPixel <= 0) throw std::runtime_error("composite_image trajectories_per_pixel must be positive");
      if (config.beamEnergyEv <= 0.0)       throw std::runtime_error("composite_image beam_energy_ev must be positive");
      if (config.scan.nxPixels <= 0 || config.scan.nyPixels <= 0)
         throw std::runtime_error("composite_image scan nx_pixels and ny_pixels must be positive");
      if (config.scan.halfWidthNm <= 0.0)   throw std::runtime_error("composite_image scan half_width_nm must be positive");
      if (config.precipitate.radiusNm <= 0.0) throw std::runtime_error("composite_image precipitate radius_nm must be positive");
      if (config.precipitate.centerDepthNm <= -config.precipitate.radiusNm)
         throw std::runtime_error("composite_image precipitate center_depth_nm must be > -radius_nm (sphere is entirely above the surface)");

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

   static void runImage(const CompositeImageConfig& config)
   {
      PhaseConfig matrix = config.matrixPhase;
      PhaseConfig precip = config.precipitatePhase;

      printf("\nCompositeImage: %s\n", config.name.c_str()); fflush(stdout);
      printf("  Matrix: %s  /  Precipitate: %s\n", matrix.name.c_str(), precip.name.c_str()); fflush(stdout);
      printf("  Precipitate: sphere r=%.1f nm, center depth=%.1f nm (x=%.1f, y=%.1f nm)\n",
             config.precipitate.radiusNm, config.precipitate.centerDepthNm,
             config.precipitate.centerXNm, config.precipitate.centerYNm); fflush(stdout);
      printf("  Scan: %dx%d pixels, ±%.1f nm FOV, %.0f eV beam, %d traj/pixel\n",
             config.scan.nxPixels, config.scan.nyPixels, config.scan.halfWidthNm,
             config.beamEnergyEv, config.trajectoriesPerPixel); fflush(stdout);

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

      // === Geometry constants (used to build per-thread regions in parallel loop) ===
      // NormalMultiPlaneShape stores its last-computed normal as mutable state, so it
      // cannot be shared across threads.
      const double origin[]     = { 0., 0., 0. };
      const double normalvec[]  = { 0., 0., -1. };
      const double surfacePos[] = { 0., 0.,  0. };
      const double precipCenter[] = {
         config.precipitate.centerXNm     * 1.e-9,
         config.precipitate.centerYNm     * 1.e-9,
         config.precipitate.centerDepthNm * 1.e-9
      };
      double precipRadius = config.precipitate.radiusNm * 1.e-9;

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

      // Results: row 0 = +halfWidthNm (top of image), row ny-1 = -halfWidthNm (bottom).
      std::vector<std::vector<double>> seMap  (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> se1Map (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> se2Map (ny, std::vector<double>(nx, 0.0));
      std::vector<std::vector<double>> bseMap (ny, std::vector<double>(nx, 0.0));

      std::ofstream csvFile(config.outputCsv.c_str());
      if (!csvFile.good()) throw std::runtime_error("Unable to open composite_image output file: " + config.outputCsv);
      csvFile << "x_nm,y_nm,SE_yield,SE1_yield,SE2_yield,BSE_yield,total_yield\n";

      auto wallStart  = std::chrono::system_clock::now();
      int  totalPixels = nx * ny;
      int  pixelsDone  = 0;

      printf("  Using %d OpenMP thread(s)\n", omp_get_max_threads()); fflush(stdout);

      #pragma omp parallel for schedule(dynamic) shared(seMap, bseMap, pixelsDone)
      for (int px = 0; px < totalPixels; ++px) {
         int    row = px / nx;
         int    col = px % nx;
         double x   = cx - hw + col * dxm;
         double y   = cy + hw - row * dym;

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

         // Per-thread geometry; NormalMultiPlaneShape has mutable stored-normal state.
         SphereT                chamberSphere_t(origin, MonteCarloSS::ChamberRadius);
         NullMaterialScatterModelT nullMSM_t;
         RegionT                chamber_t(nullptr, &nullMSM_t, &chamberSphere_t);
         chamber_t.updateMaterial(*chamber_t.getScatterModel(), vacMSM_t);

         NormalMultiPlaneShapeT surface_t;
         PlaneT                 pl_t(normalvec, 3, surfacePos, 3);
         surface_t.addPlane(pl_t);
         RegionT                bulkRegion_t(&chamber_t, &matMSM_t, (NormalShapeT*)&surface_t);

         SphereT                precipSphere_t(precipCenter, precipRadius);
         RegionT                precipRegion_t(&bulkRegion_t, &precMSM_t, &precipSphere_t);

         double        egCenter[] = { x, y, -1.e-9 };
         GaussianBeamT eg_t(beamsize, beamE, origin);
         eg_t.setCenter(egCenter);

         MonteCarloSS::MonteCarloSS monte_t(&eg_t, &chamber_t, eg_t.createElectron());
         BackscatterStatsT          back_t(monte_t, nbins);
         monte_t.addActionListener(back_t);
         monte_t.runMultipleTrajectories(config.trajectoriesPerPixel);
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

         seMap [row][col] = SEY;   // unique cell per thread, no race
         se1Map[row][col] = SE1Y;
         se2Map[row][col] = SE2Y;
         bseMap[row][col] = BSEY;

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

      auto wallEnd = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = wallEnd - wallStart;
      printf("\nCompositeImage: scan complete in %.1f s\n", elapsed.count()); fflush(stdout);

      // Write CSV in scan order (row-major) after the parallel section.
      for (int row = 0; row < ny; ++row) {
         for (int col = 0; col < nx; ++col) {
            double x     = cx - hw + col * dxm;
            double y     = cy + hw - row * dym;
            double total = seMap[row][col] + bseMap[row][col];
            csvFile << x * 1.e9 << "," << y * 1.e9 << ","
                    << seMap[row][col]  << "," << se1Map[row][col] << ","
                    << se2Map[row][col] << "," << bseMap[row][col] << "," << total << "\n";
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
