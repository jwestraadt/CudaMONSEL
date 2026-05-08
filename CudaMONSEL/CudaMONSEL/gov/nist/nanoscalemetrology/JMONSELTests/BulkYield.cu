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
#include <stdexcept>
#include <string>
#include <vector>

namespace BulkYield
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

   struct BulkYieldConfig
   {
      std::string name;
      std::string outputCsv;
      std::string outputHistogramCsv;
      int trajectories;
      double beamSizeNm;
      double seThresholdEv;
      double histogramBinSizeEv;
      ScatteringConfig scattering;
      std::vector<double> beamEnergiesEv;
      std::vector<PhaseConfig> phases;
   };

   static void addElement(PhaseConfig& phase, const ElementT* element, double fraction)
   {
      phase.elements.push_back(element);
      phase.fractions.push_back(fraction);
   }

   static BulkYieldConfig defaultConfig()
   {
      BulkYieldConfig config;
      config.name = "Ni superalloy gamma/gamma-prime";
      config.outputCsv = "BulkYield_output.csv";
      config.outputHistogramCsv = "";
      config.trajectories = 5000;
      config.beamSizeNm = 0.5;
      config.seThresholdEv = 50.0;
      config.histogramBinSizeEv = 10.0;
      config.scattering.elastic = "nist_mott";
      config.scattering.inelastic = "fitted";
      config.scattering.csd = "joy_luo_nieminen";
      config.scattering.barrier = "exp_qm";

      const double beamEnergies[] = { 200., 500., 1000., 2000., 5000., 10000., 15000., 20000. };
      for (int i = 0; i < 8; ++i) config.beamEnergiesEv.push_back(beamEnergies[i]);

      PhaseConfig gamma;
      gamma.name = "gamma";
      gamma.density = 8700.;
      gamma.workfun = 5.15;
      gamma.efermi = 8.8;
      gamma.bandgap = 0.0;
      gamma.energySEgen = 30.;
      gamma.breakEeV = 45.;
      addElement(gamma, &Element::Ni, 63.);
      addElement(gamma, &Element::Cr, 8.);
      addElement(gamma, &Element::Co, 10.);
      addElement(gamma, &Element::W, 6.);
      addElement(gamma, &Element::Re, 4.);
      addElement(gamma, &Element::Al, 6.);
      addElement(gamma, &Element::Ta, 3.);
      config.phases.push_back(gamma);

      PhaseConfig gammaPrime;
      gammaPrime.name = "gamma_prime";
      gammaPrime.density = 8200.;
      gammaPrime.workfun = 4.9;
      gammaPrime.efermi = 7.5;
      gammaPrime.bandgap = 0.0;
      gammaPrime.energySEgen = 30.;
      gammaPrime.breakEeV = 45.;
      addElement(gammaPrime, &Element::Ni, 75.);
      addElement(gammaPrime, &Element::Al, 12.);
      addElement(gammaPrime, &Element::Ti, 5.);
      addElement(gammaPrime, &Element::Ta, 5.);
      addElement(gammaPrime, &Element::Cr, 3.);
      config.phases.push_back(gammaPrime);

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
         throw std::runtime_error("Unsupported bulk_yield scattering model for " + fieldName + ": " + modelName);
   }

   static void validateScatteringConfig(const ScatteringConfig& scattering)
   {
      requireModel("elastic", scattering.elastic, "nist_mott");
      requireModel("inelastic", scattering.inelastic, "fitted");
      requireModel("csd", scattering.csd, "joy_luo_nieminen");
      requireModel("barrier", scattering.barrier, "exp_qm");
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
      if (!json.isObject()) throw std::runtime_error("bulk_yield scattering must be an object");

      ScatteringConfig scattering = defaults;
      scattering.elastic = normalizeModelName(stringOr(json, scattering.elastic, "elastic"));
      scattering.inelastic = normalizeModelName(stringOr(json, scattering.inelastic, "inelastic"));
      scattering.csd = normalizeModelName(stringOr(json, scattering.csd, "csd"));
      scattering.barrier = normalizeModelName(stringOr(json, scattering.barrier, "barrier"));
      validateScatteringConfig(scattering);
      return scattering;
   }

   static const ElementT* elementByName(const std::string& symbol)
   {
      const ElementT& element = Element::byName(symbol.c_str());
      if (!element.isValid()) throw std::runtime_error("Unknown element in bulk_yield composition: " + symbol);
      return &element;
   }

   static void readCompositionObject(PhaseConfig& phase, const RuntimeInput::JsonValue& composition)
   {
      if (!composition.isObject()) throw std::runtime_error("bulk_yield phase composition must be an object");
      for (std::map<std::string, RuntimeInput::JsonValue>::const_iterator it = composition.objectValue.begin();
           it != composition.objectValue.end(); ++it) {
         if (!it->second.isNumber()) throw std::runtime_error("bulk_yield composition values must be numbers");
         addElement(phase, elementByName(it->first), it->second.numberValue);
      }
   }

   static void readElementArray(PhaseConfig& phase, const RuntimeInput::JsonValue& elements)
   {
      if (!elements.isArray()) throw std::runtime_error("bulk_yield phase elements must be an array");
      for (size_t i = 0; i < elements.arrayValue.size(); ++i) {
         const RuntimeInput::JsonValue& item = elements.arrayValue[i];
         if (!item.isObject()) throw std::runtime_error("bulk_yield elements entries must be objects");

         std::string symbol = requireString(item, "symbol", "element", "name");
         double fraction = requireNumber(item, "fraction", "mole_fraction", "atomic_percent");
         addElement(phase, elementByName(symbol), fraction);
      }
   }

   static PhaseConfig readPhaseConfig(const RuntimeInput::JsonValue& json)
   {
      if (!json.isObject()) throw std::runtime_error("bulk_yield phases must be objects");

      PhaseConfig phase;
      phase.name = requireString(json, "name");
      phase.density = requireNumber(json, "density_kg_m3", "density");
      phase.workfun = requireNumber(json, "work_function_ev", "workfunction_ev");
      phase.efermi = requireNumber(json, "fermi_energy_ev", "efermi_ev");
      phase.bandgap = numberOr(json, 0.0, "bandgap_ev", "band_gap_ev");
      phase.energySEgen = requireNumber(json, "secondary_generation_energy_ev", "energy_se_gen_ev");
      phase.breakEeV = requireNumber(json, "break_energy_ev", "csd_break_energy_ev");

      const RuntimeInput::JsonValue* elements = json.find("elements");
      const RuntimeInput::JsonValue* composition = json.find("composition");
      if (elements != nullptr) readElementArray(phase, *elements);
      else if (composition != nullptr) readCompositionObject(phase, *composition);
      else throw std::runtime_error("bulk_yield phase requires elements[] or composition{}");

      if (phase.elements.empty()) throw std::runtime_error("bulk_yield phase has no elements: " + phase.name);
      return phase;
   }

   static BulkYieldConfig readConfig(const RuntimeInput::JsonValue& json)
   {
      if (!json.isObject()) throw std::runtime_error("bulk_yield config must be a JSON object");

      BulkYieldConfig config = defaultConfig();
      const RuntimeInput::JsonValue* parameters = json.find("parameters");
      const RuntimeInput::JsonValue& source =
         (parameters != nullptr && parameters->isObject()) ? *parameters : json;

      config.name = stringOr(source, stringOr(json, config.name, "name"), "name");
      config.outputCsv = stringOr(source, config.outputCsv, "output_csv", "output", "outputCsv");
      config.outputHistogramCsv = stringOr(source, config.outputHistogramCsv, "output_histogram_csv", "histogram_csv");
      config.trajectories = (int)numberOr(source, (double)config.trajectories, "trajectories", "n_trajectories");
      config.beamSizeNm = numberOr(source, config.beamSizeNm, "beam_size_nm", "beam_sigma_nm");
      config.seThresholdEv = numberOr(source, config.seThresholdEv, "secondary_electron_threshold_ev", "se_threshold_ev");
      config.histogramBinSizeEv = numberOr(source, config.histogramBinSizeEv, "histogram_bin_size_ev", "bin_size_ev");

      const RuntimeInput::JsonValue* scattering = source.find("scattering");
      if (scattering != nullptr)
         config.scattering = readScatteringConfig(*scattering, config.scattering);
      else
         validateScatteringConfig(config.scattering);

      const RuntimeInput::JsonValue* energies = findAny(source, "beam_energies_ev", "beamEnergiesEv");
      if (energies != nullptr) {
         if (!energies->isArray()) throw std::runtime_error("bulk_yield beam_energies_ev must be an array");
         config.beamEnergiesEv.clear();
         for (size_t i = 0; i < energies->arrayValue.size(); ++i) {
            if (!energies->arrayValue[i].isNumber()) throw std::runtime_error("bulk_yield beam_energies_ev entries must be numbers");
            config.beamEnergiesEv.push_back(energies->arrayValue[i].numberValue);
         }
      }

      const RuntimeInput::JsonValue* phases = source.find("phases");
      if (phases != nullptr) {
         if (!phases->isArray()) throw std::runtime_error("bulk_yield phases must be an array");
         config.phases.clear();
         for (size_t i = 0; i < phases->arrayValue.size(); ++i)
            config.phases.push_back(readPhaseConfig(phases->arrayValue[i]));
      }

      if (config.trajectories <= 0) throw std::runtime_error("bulk_yield trajectories must be positive");
      if (config.beamEnergiesEv.empty()) throw std::runtime_error("bulk_yield beam_energies_ev must not be empty");
      if (config.phases.empty()) throw std::runtime_error("bulk_yield phases must not be empty");
      if (config.histogramBinSizeEv <= 0.0) throw std::runtime_error("bulk_yield histogram_bin_size_ev must be positive");
      return config;
   }

   static void runPhase(
      const BulkYieldConfig& config,
      PhaseConfig&          phase,
      std::ofstream&        outfile,
      std::ofstream*        histfile)
   {
      printf("\n--- Phase: %s ---\n", phase.name.c_str()); fflush(stdout);
      printf("BeamE_eV,BSE_yield,SE_yield,total_yield\n"); fflush(stdout);

      double potU = -phase.workfun - phase.efermi;

      CompositionT comp;
      comp.defineByMoleFraction(phase.elements.data(), (int)phase.elements.size(),
                                phase.fractions.data(), (int)phase.fractions.size());

      SEmaterialT mat(comp, phase.density);
      mat.setWorkfunction(ToSI::eV(phase.workfun));
      mat.setBandgap(ToSI::eV(phase.bandgap));
      mat.setEnergyCBbottom(ToSI::eV(potU));

      // Currently supported JSON-selected stack:
      // nist_mott + fitted + joy_luo_nieminen + exp_qm.
      SelectableElasticSMT elasticSM(mat, NISTMottRS::Factory);
      JoyLuoNieminenCSDT   csd(mat, ToSI::eV(phase.breakEeV));
      FittedInelSMT        inelSM(mat, ToSI::eV(phase.energySEgen), csd);

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

      double beamsize = config.beamSizeNm * 1.e-9;

      for (size_t ei = 0; ei < config.beamEnergiesEv.size(); ++ei) {
         double beamEeV = config.beamEnergiesEv[ei];
         double beamE   = ToSI::eV(beamEeV);

         GaussianBeamT eg(beamsize, beamE, center);
         double egCenter[] = { 0., 0., -1.e-9 };  // 1 nm above surface
         eg.setCenter(egCenter);

         MonteCarloSS::MonteCarloSS monte(&eg, &chamber, eg.createElectron());

         int nbins = (int)(beamEeV / config.histogramBinSizeEv);
         if (nbins < 1) nbins = 1;
         BackscatterStatsT back(monte, nbins);
         monte.addActionListener(back);

         monte.runMultipleTrajectories(config.trajectories);

         const HistogramT& hist    = back.backscatterEnergyHistogram();
         double ePerBin            = beamEeV / hist.binCount();
         int    maxSEbin           = (int)(config.seThresholdEv / ePerBin);
         int    totalSE            = 0;
         for (int j = 0; j < maxSEbin && j < (int)hist.binCount(); ++j)
            totalSE += hist.counts(j);

         double SEY   = (double)totalSE / config.trajectories;
         double BSEY  = back.backscatterFraction() - SEY;
         double total = back.backscatterFraction();

         printf("%.0f,%.4f,%.4f,%.4f\n", beamEeV, BSEY, SEY, total); fflush(stdout);
         outfile << phase.name << "," << beamEeV << ","
                 << BSEY << "," << SEY << "," << total << "\n";
         outfile.flush();

         if (histfile != nullptr) {
            for (int j = 0; j < (int)hist.binCount(); ++j) {
               double binMin = hist.minValue(j);
               double binMax = hist.maxValue(j);
               *histfile << phase.name << "," << beamEeV << ","
                         << binMin << "," << binMax << ","
                         << hist.counts(j) << ","
                         << (double)hist.counts(j) / config.trajectories << "\n";
            }
            histfile->flush();
         }

         monte.removeActionListener(back);
      }
   }

   static void runConfig(BulkYieldConfig config)
   {
      printf("BulkYield: %s, %d trajectories per point\n",
             config.name.c_str(), config.trajectories); fflush(stdout);
      printf("BulkYield scattering: elastic=%s, inelastic=%s, csd=%s, barrier=%s\n",
             config.scattering.elastic.c_str(),
             config.scattering.inelastic.c_str(),
             config.scattering.csd.c_str(),
             config.scattering.barrier.c_str()); fflush(stdout);

      std::ofstream outfile(config.outputCsv.c_str());
      if (!outfile.good()) throw std::runtime_error("Unable to open bulk_yield output file: " + config.outputCsv);
      outfile << "phase,BeamE_eV,BSE_yield,SE_yield,total_yield\n";

      std::ofstream histfile;
      if (!config.outputHistogramCsv.empty()) {
         histfile.open(config.outputHistogramCsv.c_str());
         if (!histfile.good()) throw std::runtime_error("Unable to open histogram output file: " + config.outputHistogramCsv);
         histfile << "phase,beam_energy_ev,bin_min_ev,bin_max_ev,counts,yield\n";
      }

      auto wallStart = std::chrono::system_clock::now();

      for (size_t i = 0; i < config.phases.size(); ++i)
         runPhase(config, config.phases[i], outfile, histfile.is_open() ? &histfile : nullptr);

      auto wallEnd = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = wallEnd - wallStart;
      printf("\nBulkYield: done in %.1f s  -->  %s\n",
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
