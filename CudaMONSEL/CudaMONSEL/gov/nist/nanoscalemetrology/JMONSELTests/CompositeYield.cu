// file: gov\nist\nanoscalemetrology\JMONSELTests\CompositeYield.cu
//
// SE and BSE yield simulation for a composite material: one matrix phase with a
// single spherical precipitate embedded at a configurable depth.
//
// Geometry: semi-infinite matrix half-space (z > 0) with a sphere sub-region
// whose scatter model switches to the precipitate phase when the electron is
// inside the sphere.  MonteCarloSS handles boundary crossings automatically
// via RegionBase::findEndOfStep / containingSubRegion.
//
// Scatter stack: SelectableElasticSM (NISTMott) + FittedInelSM + JoyLuoNieminenCSD
// (same as bulk_yield — no JMONSEL tables required).

#include "gov\nist\nanoscalemetrology\JMONSELTests\CompositeYield.cuh"

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

namespace CompositeYield
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

   struct CompositeYieldConfig
   {
      std::string name;
      std::string outputCsv;
      std::string outputHistogramCsv;
      int trajectories;
      double beamSizeNm;
      double seThresholdEv;
      double histogramBinSizeEv;
      ScatteringConfig    scattering;
      std::vector<double> beamEnergiesEv;
      PhaseConfig         matrixPhase;
      PhaseConfig         precipitatePhase;
      PrecipitateConfig   precipitate;
   };

   static void addElement(PhaseConfig& phase, const ElementT* element, double fraction)
   {
      phase.elements.push_back(element);
      phase.fractions.push_back(fraction);
   }

   static CompositeYieldConfig defaultConfig()
   {
      CompositeYieldConfig config;
      config.name                = "Ni gamma matrix + gamma-prime precipitate";
      config.outputCsv           = "CompositeYield_output.csv";
      config.outputHistogramCsv  = "";
      config.trajectories        = 5000;
      config.beamSizeNm          = 0.5;
      config.seThresholdEv       = 50.0;
      config.histogramBinSizeEv  = 10.0;
      config.scattering.elastic  = "nist_mott";
      config.scattering.inelastic = "fitted";
      config.scattering.csd      = "joy_luo_nieminen";
      config.scattering.barrier  = "exp_qm";

      const double beamEnergies[] = { 200., 500., 1000., 2000., 5000., 10000., 15000., 20000. };
      for (int i = 0; i < 8; ++i) config.beamEnergiesEv.push_back(beamEnergies[i]);

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

      config.precipitate.radiusNm      = 25.0;
      config.precipitate.centerXNm     = 0.0;
      config.precipitate.centerYNm     = 0.0;
      config.precipitate.centerDepthNm = 25.0;

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

   static void requireModel(const std::string& fieldName, const std::string& modelName, const std::string& supportedName)
   {
      if (normalizeModelName(modelName) != supportedName)
         throw std::runtime_error("Unsupported composite_yield scattering model for " + fieldName + ": " + modelName);
   }

   static void validateScatteringConfig(const ScatteringConfig& scattering)
   {
      requireModel("elastic",   scattering.elastic,   "nist_mott");
      requireModel("inelastic", scattering.inelastic, "fitted");
      requireModel("csd",       scattering.csd,       "joy_luo_nieminen");
      requireModel("barrier",   scattering.barrier,   "exp_qm");
   }

   static const RuntimeInput::JsonValue* findAny(
      const RuntimeInput::JsonValue& object,
      const char* key1,
      const char* key2 = nullptr,
      const char* key3 = nullptr)
   {
      const RuntimeInput::JsonValue* value = key1 == nullptr ? nullptr : object.find(key1);
      if (value != nullptr) return value;
      value = key2 == nullptr ? nullptr : object.find(key2);
      if (value != nullptr) return value;
      return key3 == nullptr ? nullptr : object.find(key3);
   }

   static std::string requireString(
      const RuntimeInput::JsonValue& object,
      const char* key1,
      const char* key2 = nullptr,
      const char* key3 = nullptr)
   {
      const RuntimeInput::JsonValue* value = findAny(object, key1, key2, key3);
      if (value == nullptr) throw std::runtime_error(std::string("Missing required string field: ") + key1);
      if (!value->isString()) throw std::runtime_error(std::string("Expected string field: ") + key1);
      return value->stringValue;
   }

   static double requireNumber(
      const RuntimeInput::JsonValue& object,
      const char* key1,
      const char* key2 = nullptr,
      const char* key3 = nullptr)
   {
      const RuntimeInput::JsonValue* value = findAny(object, key1, key2, key3);
      if (value == nullptr) throw std::runtime_error(std::string("Missing required number field: ") + key1);
      if (!value->isNumber()) throw std::runtime_error(std::string("Expected number field: ") + key1);
      return value->numberValue;
   }

   static double numberOr(
      const RuntimeInput::JsonValue& object,
      double defaultValue,
      const char* key1,
      const char* key2 = nullptr,
      const char* key3 = nullptr)
   {
      const RuntimeInput::JsonValue* value = findAny(object, key1, key2, key3);
      if (value == nullptr) return defaultValue;
      if (!value->isNumber()) throw std::runtime_error(std::string("Expected number field: ") + key1);
      return value->numberValue;
   }

   static std::string stringOr(
      const RuntimeInput::JsonValue& object,
      const std::string& defaultValue,
      const char* key1,
      const char* key2 = nullptr,
      const char* key3 = nullptr)
   {
      const RuntimeInput::JsonValue* value = findAny(object, key1, key2, key3);
      if (value == nullptr) return defaultValue;
      if (!value->isString()) throw std::runtime_error(std::string("Expected string field: ") + key1);
      return value->stringValue;
   }

   static ScatteringConfig readScatteringConfig(
      const RuntimeInput::JsonValue& json,
      const ScatteringConfig& defaults)
   {
      if (!json.isObject()) throw std::runtime_error("composite_yield scattering must be an object");
      ScatteringConfig scattering = defaults;
      scattering.elastic   = normalizeModelName(stringOr(json, scattering.elastic,   "elastic"));
      scattering.inelastic = normalizeModelName(stringOr(json, scattering.inelastic, "inelastic"));
      scattering.csd       = normalizeModelName(stringOr(json, scattering.csd,       "csd"));
      scattering.barrier   = normalizeModelName(stringOr(json, scattering.barrier,   "barrier"));
      validateScatteringConfig(scattering);
      return scattering;
   }

   static const ElementT* elementByName(const std::string& symbol)
   {
      const ElementT& element = Element::byName(symbol.c_str());
      if (!element.isValid()) throw std::runtime_error("Unknown element in composite_yield composition: " + symbol);
      return &element;
   }

   static void readCompositionObject(PhaseConfig& phase, const RuntimeInput::JsonValue& composition)
   {
      if (!composition.isObject()) throw std::runtime_error("composite_yield phase composition must be an object");
      for (std::map<std::string, RuntimeInput::JsonValue>::const_iterator it = composition.objectValue.begin();
           it != composition.objectValue.end(); ++it) {
         if (!it->second.isNumber()) throw std::runtime_error("composite_yield composition values must be numbers");
         addElement(phase, elementByName(it->first), it->second.numberValue);
      }
   }

   static void readElementArray(PhaseConfig& phase, const RuntimeInput::JsonValue& elements)
   {
      if (!elements.isArray()) throw std::runtime_error("composite_yield phase elements must be an array");
      for (size_t i = 0; i < elements.arrayValue.size(); ++i) {
         const RuntimeInput::JsonValue& item = elements.arrayValue[i];
         if (!item.isObject()) throw std::runtime_error("composite_yield elements entries must be objects");
         std::string symbol   = requireString(item, "symbol", "element", "name");
         double      fraction = requireNumber(item, "fraction", "mole_fraction", "atomic_percent");
         addElement(phase, elementByName(symbol), fraction);
      }
   }

   static PhaseConfig readPhaseConfig(const RuntimeInput::JsonValue& json, const char* fieldName)
   {
      if (!json.isObject()) throw std::runtime_error(std::string(fieldName) + " must be a JSON object");
      PhaseConfig phase;
      phase.name        = requireString(json, "name");
      phase.density     = requireNumber(json, "density_kg_m3",                "density");
      phase.workfun     = requireNumber(json, "work_function_ev",             "workfunction_ev");
      phase.efermi      = requireNumber(json, "fermi_energy_ev",              "efermi_ev");
      phase.bandgap     = numberOr    (json, 0.0, "bandgap_ev",              "band_gap_ev");
      phase.energySEgen = requireNumber(json, "secondary_generation_energy_ev", "energy_se_gen_ev");
      phase.breakEeV    = requireNumber(json, "break_energy_ev",             "csd_break_energy_ev");

      const RuntimeInput::JsonValue* elements    = json.find("elements");
      const RuntimeInput::JsonValue* composition = json.find("composition");
      if (elements != nullptr)         readElementArray(phase, *elements);
      else if (composition != nullptr) readCompositionObject(phase, *composition);
      else throw std::runtime_error(std::string(fieldName) + " requires elements[] or composition{}");

      if (phase.elements.empty()) throw std::runtime_error(std::string(fieldName) + " has no elements: " + phase.name);
      return phase;
   }

   static PrecipitateConfig readPrecipitateConfig(
      const RuntimeInput::JsonValue& json,
      const PrecipitateConfig& defaults)
   {
      if (!json.isObject()) throw std::runtime_error("composite_yield precipitate must be an object");
      PrecipitateConfig pc = defaults;
      pc.radiusNm      = numberOr(json, pc.radiusNm,      "radius_nm",       "radius");
      pc.centerXNm     = numberOr(json, pc.centerXNm,     "center_x_nm",     "x_nm");
      pc.centerYNm     = numberOr(json, pc.centerYNm,     "center_y_nm",     "y_nm");
      pc.centerDepthNm = numberOr(json, pc.centerDepthNm, "center_depth_nm", "depth_nm");
      return pc;
   }

   static CompositeYieldConfig readConfig(const RuntimeInput::JsonValue& json)
   {
      if (!json.isObject()) throw std::runtime_error("composite_yield config must be a JSON object");

      CompositeYieldConfig config = defaultConfig();
      const RuntimeInput::JsonValue* parameters = json.find("parameters");
      const RuntimeInput::JsonValue& source =
         (parameters != nullptr && parameters->isObject()) ? *parameters : json;

      config.name               = stringOr(source, config.name,               "name");
      config.outputCsv          = stringOr(source, config.outputCsv,          "output_csv", "output", "outputCsv");
      config.outputHistogramCsv = stringOr(source, config.outputHistogramCsv, "output_histogram_csv", "histogram_csv");
      config.trajectories       = (int)numberOr(source, (double)config.trajectories, "trajectories", "n_trajectories");
      config.beamSizeNm         = numberOr(source, config.beamSizeNm,         "beam_size_nm",                  "beam_sigma_nm");
      config.seThresholdEv      = numberOr(source, config.seThresholdEv,      "secondary_electron_threshold_ev", "se_threshold_ev");
      config.histogramBinSizeEv = numberOr(source, config.histogramBinSizeEv, "histogram_bin_size_ev",          "bin_size_ev");

      const RuntimeInput::JsonValue* scattering = source.find("scattering");
      if (scattering != nullptr)
         config.scattering = readScatteringConfig(*scattering, config.scattering);
      else
         validateScatteringConfig(config.scattering);

      const RuntimeInput::JsonValue* energies = findAny(source, "beam_energies_ev", "beamEnergiesEv");
      if (energies != nullptr) {
         if (!energies->isArray()) throw std::runtime_error("composite_yield beam_energies_ev must be an array");
         config.beamEnergiesEv.clear();
         for (size_t i = 0; i < energies->arrayValue.size(); ++i) {
            if (!energies->arrayValue[i].isNumber()) throw std::runtime_error("composite_yield beam_energies_ev entries must be numbers");
            config.beamEnergiesEv.push_back(energies->arrayValue[i].numberValue);
         }
      }

      const RuntimeInput::JsonValue* matrixJson = findAny(source, "matrix_phase", "matrix");
      if (matrixJson != nullptr) config.matrixPhase = readPhaseConfig(*matrixJson, "matrix_phase");

      const RuntimeInput::JsonValue* precipPhaseJson = findAny(source, "precipitate_phase", "precipitate_material");
      if (precipPhaseJson != nullptr) config.precipitatePhase = readPhaseConfig(*precipPhaseJson, "precipitate_phase");

      const RuntimeInput::JsonValue* precipJson = findAny(source, "precipitate", "sphere");
      if (precipJson != nullptr) config.precipitate = readPrecipitateConfig(*precipJson, config.precipitate);

      if (config.trajectories <= 0) throw std::runtime_error("composite_yield trajectories must be positive");
      if (config.beamEnergiesEv.empty()) throw std::runtime_error("composite_yield beam_energies_ev must not be empty");
      if (config.histogramBinSizeEv <= 0.0) throw std::runtime_error("composite_yield histogram_bin_size_ev must be positive");
      if (config.precipitate.radiusNm <= 0.0) throw std::runtime_error("composite_yield precipitate radius_nm must be positive");
      if (config.precipitate.centerDepthNm <= -config.precipitate.radiusNm)
         throw std::runtime_error("composite_yield precipitate center_depth_nm must be > -radius_nm (sphere is entirely above the surface)");

      return config;
   }

   static void runComposite(
      const CompositeYieldConfig& config,
      std::ofstream&              outfile,
      std::ofstream*              histfile)
   {
      PhaseConfig matrix = config.matrixPhase;
      PhaseConfig precip = config.precipitatePhase;

      printf("\n--- Matrix: %s  /  Precipitate: %s ---\n",
             matrix.name.c_str(), precip.name.c_str()); fflush(stdout);
      printf("Precipitate: sphere r=%.1f nm at depth=%.1f nm (x=%.1f, y=%.1f nm)\n",
             config.precipitate.radiusNm,
             config.precipitate.centerDepthNm,
             config.precipitate.centerXNm,
             config.precipitate.centerYNm); fflush(stdout);
      printf("BeamE_eV,BSE_yield,SE_yield,total_yield\n"); fflush(stdout);

      // === Matrix scatter model ===
      double matPotU = -matrix.workfun - matrix.efermi;
      CompositionT matComp;
      matComp.defineByMoleFraction(matrix.elements.data(), (int)matrix.elements.size(),
                                   matrix.fractions.data(), (int)matrix.fractions.size());
      SEmaterialT matMat(matComp, matrix.density);
      matMat.setWorkfunction(ToSI::eV(matrix.workfun));
      matMat.setBandgap(ToSI::eV(matrix.bandgap));
      matMat.setEnergyCBbottom(ToSI::eV(matPotU));

      SelectableElasticSMT matElastic(matMat, NISTMottRS::Factory);
      JoyLuoNieminenCSDT   matCSD(matMat, ToSI::eV(matrix.breakEeV));
      FittedInelSMT        matInel(matMat, ToSI::eV(matrix.energySEgen), matCSD);
      ExpQMBarrierSMT      matBarrier(&matMat);
      MONSEL_MaterialScatterModelT matMSM(&matMat, &matBarrier);
      matMSM.addScatterMechanism(&matElastic);
      matMSM.addScatterMechanism(&matInel);
      matMSM.setCSD(&matCSD);

      // === Precipitate scatter model ===
      double precPotU = -precip.workfun - precip.efermi;
      CompositionT precComp;
      precComp.defineByMoleFraction(precip.elements.data(), (int)precip.elements.size(),
                                    precip.fractions.data(), (int)precip.fractions.size());
      SEmaterialT precMat(precComp, precip.density);
      precMat.setWorkfunction(ToSI::eV(precip.workfun));
      precMat.setBandgap(ToSI::eV(precip.bandgap));
      precMat.setEnergyCBbottom(ToSI::eV(precPotU));

      SelectableElasticSMT precElastic(precMat, NISTMottRS::Factory);
      JoyLuoNieminenCSDT   precCSD(precMat, ToSI::eV(precip.breakEeV));
      FittedInelSMT        precInel(precMat, ToSI::eV(precip.energySEgen), precCSD);
      ExpQMBarrierSMT      precBarrier(&precMat);
      MONSEL_MaterialScatterModelT precMSM(&precMat, &precBarrier);
      precMSM.addScatterMechanism(&precElastic);
      precMSM.addScatterMechanism(&precInel);
      precMSM.setCSD(&precCSD);

      // === Vacuum ===
      SEmaterialT vacMat;
      vacMat.setName("vacuum");
      ExpQMBarrierSMT              vacBarrier(&vacMat);
      MONSEL_MaterialScatterModelT vacMSM(&vacMat, &vacBarrier);

      // === Geometry ===
      NullMaterialScatterModelT NULL_MSM;
      const double origin[] = { 0., 0., 0. };
      SphereT chamberSphere(origin, MonteCarloSS::ChamberRadius);
      RegionT chamber(nullptr, &NULL_MSM, &chamberSphere);
      chamber.updateMaterial(*chamber.getScatterModel(), vacMSM);

      const double normalvec[]  = { 0., 0., -1. };
      const double surfacePos[] = { 0., 0.,  0. };
      NormalMultiPlaneShapeT surface;
      PlaneT pl(normalvec, 3, surfacePos, 3);
      surface.addPlane(pl);
      RegionT bulkRegion(&chamber, &matMSM, (NormalShapeT*)&surface);

      // Precipitate sphere — self-registers into bulkRegion.mSubRegions on construction
      const double precipCenter[] = {
         config.precipitate.centerXNm    * 1.e-9,
         config.precipitate.centerYNm    * 1.e-9,
         config.precipitate.centerDepthNm * 1.e-9
      };
      double precipRadius = config.precipitate.radiusNm * 1.e-9;
      SphereT precipSphere(precipCenter, precipRadius);
      RegionT precipRegion(&bulkRegion, &precMSM, &precipSphere);

      double beamsize = config.beamSizeNm * 1.e-9;

      for (size_t ei = 0; ei < config.beamEnergiesEv.size(); ++ei) {
         double beamEeV = config.beamEnergiesEv[ei];
         double beamE   = ToSI::eV(beamEeV);

         GaussianBeamT eg(beamsize, beamE, origin);
         double egCenter[] = { 0., 0., -1.e-9 };
         eg.setCenter(egCenter);

         MonteCarloSS::MonteCarloSS monte(&eg, &chamber, eg.createElectron());

         int nbins = (int)(beamEeV / config.histogramBinSizeEv);
         if (nbins < 1) nbins = 1;
         BackscatterStatsT back(monte, nbins);
         monte.addActionListener(back);

         monte.runMultipleTrajectories(config.trajectories);

         const HistogramT& hist = back.backscatterEnergyHistogram();
         double ePerBin         = beamEeV / hist.binCount();
         int    maxSEbin        = (int)(config.seThresholdEv / ePerBin);
         int    totalSE         = 0;
         for (int j = 0; j < maxSEbin && j < (int)hist.binCount(); ++j)
            totalSE += hist.counts(j);

         double SEY   = (double)totalSE / config.trajectories;
         double BSEY  = back.backscatterFraction() - SEY;
         double total = back.backscatterFraction();

         printf("%.0f,%.4f,%.4f,%.4f\n", beamEeV, BSEY, SEY, total); fflush(stdout);
         outfile << config.name << "," << beamEeV << ","
                 << BSEY << "," << SEY << "," << total << "\n";
         outfile.flush();

         if (histfile != nullptr) {
            for (int j = 0; j < (int)hist.binCount(); ++j) {
               double binMin = hist.minValue(j);
               double binMax = hist.maxValue(j);
               *histfile << config.name << "," << beamEeV << ","
                         << binMin << "," << binMax << ","
                         << hist.counts(j) << ","
                         << (double)hist.counts(j) / config.trajectories << "\n";
            }
            histfile->flush();
         }

         monte.removeActionListener(back);
      }
   }

   static void runConfig(CompositeYieldConfig config)
   {
      printf("CompositeYield: %s, %d trajectories per point\n",
             config.name.c_str(), config.trajectories); fflush(stdout);
      printf("CompositeYield scattering: elastic=%s, inelastic=%s, csd=%s, barrier=%s\n",
             config.scattering.elastic.c_str(),
             config.scattering.inelastic.c_str(),
             config.scattering.csd.c_str(),
             config.scattering.barrier.c_str()); fflush(stdout);

      std::ofstream outfile(config.outputCsv.c_str());
      if (!outfile.good()) throw std::runtime_error("Unable to open composite_yield output file: " + config.outputCsv);
      outfile << "simulation,BeamE_eV,BSE_yield,SE_yield,total_yield\n";

      std::ofstream histfile;
      if (!config.outputHistogramCsv.empty()) {
         histfile.open(config.outputHistogramCsv.c_str());
         if (!histfile.good()) throw std::runtime_error("Unable to open histogram output file: " + config.outputHistogramCsv);
         histfile << "simulation,beam_energy_ev,bin_min_ev,bin_max_ev,counts,yield\n";
      }

      auto wallStart = std::chrono::system_clock::now();

      runComposite(config, outfile, histfile.is_open() ? &histfile : nullptr);

      auto wallEnd = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = wallEnd - wallStart;
      printf("\nCompositeYield: done in %.1f s  -->  %s\n",
             elapsed.count(), config.outputCsv.c_str()); fflush(stdout);
      outfile.close();
   }

   void run()
   {
      runConfig(defaultConfig());
   }

   void run(const RuntimeInput::JsonValue& config)
   {
      runConfig(readConfig(config));
   }
}
