/*
* - without length, the array parameter (eg const double a[]) is always 3 dimensional
* - use "auto" keyword for pointers ONLY
*/

#include <stdio.h>

#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "gov\nist\microanalysis\Utility\UncertainValue2.cuh"
#include "gov\nist\microanalysis\EPQLibrary\Element.cuh"
#include "gov\nist\microanalysis\EPQLibrary\Material.cuh"

#include "gov\nist\microanalysis\EPQLibrary\CzyzewskiMottScatteringAngle.cuh"
#include "gov\nist\microanalysis\EPQLibrary\GasScatteringCrossSection.cuh"
#include "gov\nist\microanalysis\EPQLibrary\NISTMottScatteringAngle.cuh"
#include "gov\nist\microanalysis\EPQLibrary\MeanIonizationPotential.cuh"

#include "gov\nist\nanoscalemetrology\JMONSEL\NISTMottRS.cuh"

#include "gov\nist\microanalysis\EPQTests\UncertainValue2Test.cuh"
#include "gov\nist\microanalysis\EPQTests\ElementTest.cuh"
#include "gov\nist\microanalysis\EPQTests\MaterialTest.cuh"
#include "gov\nist\microanalysis\EPQTests\AtomicShellTest.cuh"
#include "gov\nist\microanalysis\EPQTests\EdgeEnergyTest.cuh"
#include "gov\nist\microanalysis\EPQTests\MeanIonizationPotentialTest.cuh"
#include "gov\nist\microanalysis\EPQTests\SphereTest.cuh"
#include "gov\nist\microanalysis\EPQTests\Math2Test.cuh"
#include "gov\nist\microanalysis\EPQTests\CylindricalShapeTest.cuh"
#include "gov\nist\microanalysis\EPQTests\SumShapeTest.cuh"
#include "gov\nist\microanalysis\EPQTests\BetheElectronEnergyLossTest.cuh"
#include "gov\nist\microanalysis\EPQTests\MonteCarloSSTest.cuh"

#include "gov\nist\nanoscalemetrology\JMONSELTests\LinesOnLayers.cuh"
#include "gov\nist\nanoscalemetrology\JMONSELTests\BulkYield.cuh"
#include "gov\nist\nanoscalemetrology\JMONSELTests\CompositeYield.cuh"
#include "gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage.cuh"

#include "ImageUtil.h"
#include "RuntimeInput.cuh"

static void initializeRuntime()
{
   printf("init: CzyzewskiMott\n"); fflush(stdout);
   CzyzewskiMottScatteringAngle::init();
   printf("init: NISTMott\n"); fflush(stdout);
   NISTMottScatteringAngle::init();
   printf("init: Gas\n"); fflush(stdout);
   GasScatteringCrossSection::init();
   printf("init: NISTMottRS\n"); fflush(stdout);
   NISTMottRS::init();
   printf("init: Berger64\n"); fflush(stdout);
   MeanIonizationPotential::Berger64MeanIonizationPotential::readTabulatedValues();
   printf("init: Berger83\n"); fflush(stdout);
   MeanIonizationPotential::Berger83MeanIonizationPotential::readTabulatedValues();
   printf("init: done\n"); fflush(stdout);
}

static void runSelfTests()
{
#define RUN(label, stmt) printf("TEST: %s\n", label); fflush(stdout); stmt; printf("PASS: %s\n", label); fflush(stdout);

   RUN("Math2Test::testRandom1", Math2Test::testRandom1())
   RUN("Math2Test::testRandom2", Math2Test::testRandom2())
   RUN("UncertainValue2Test::testSpecialValues", { UncertainValue2Test::UncertainValue2Test uvTest; uvTest.testSpecialValues(); uvTest.testA(); uvTest.testB(); uvTest.testC(); uvTest.testAB(); uvTest.testAdd1(); uvTest.testAdd2(); uvTest.testAdd3(); uvTest.testMultiply(); uvTest.testDivide(); uvTest.testFunctions(); })
   RUN("ElementTest", { ElementTest::ElementTest t; t.testZero(); t.testOne(); })
   RUN("MaterialTest", { MaterialTest::MaterialTest t; t.testOne(); })
   RUN("AtomicShellTest", AtomicShellTest::testOne())
   RUN("EdgeEnergyTest", EdgeEnergyTest::testOne())
   RUN("MeanIonizationPotentialTest", MeanIonizationPotentialTest::testOne())
   RUN("SphereTest::testContains", SphereTest::testContains())
   RUN("SphereTest::testGetFirstIntersection", SphereTest::testGetFirstIntersection())
   RUN("CylindricalShapeTest", { CylindricalShapeTest::testZero(); CylindricalShapeTest::testOne(); CylindricalShapeTest::testTwo(); CylindricalShapeTest::testThree(); CylindricalShapeTest::testFour(); CylindricalShapeTest::testFive(); CylindricalShapeTest::testSix(); CylindricalShapeTest::testSeven(); CylindricalShapeTest::testEight(); CylindricalShapeTest::testNine(); CylindricalShapeTest::testTen(); CylindricalShapeTest::testEleven(); CylindricalShapeTest::testTwelve(); })
   RUN("BetheElectronEnergyLossTest", BetheElectronEnergyLossTest::testOne())
   RUN("MonteCarloSSTest", MonteCarloSSTest::testOne())
   RUN("SumShapeTest", SumShapeTest::testGetFirstIntersection())

#undef RUN
}

static void runDefaultSuite()
{
#define RUN(label, stmt) printf("TEST: %s\n", label); fflush(stdout); stmt; printf("PASS: %s\n", label); fflush(stdout);

   runSelfTests();
   RUN("LinesOnLayers", LinesOnLayers::run())
   RUN("BulkYield", BulkYield::run())

#undef RUN
}

static std::string lowerCopy(const std::string& value)
{
   std::string result = value;
   for (size_t i = 0; i < result.size(); ++i) {
      if (result[i] >= 'A' && result[i] <= 'Z')
         result[i] = (char)(result[i] - 'A' + 'a');
   }
   return result;
}

static void collectSimulations(const RuntimeInput::JsonValue& root, std::vector<const RuntimeInput::JsonValue*>& simulations)
{
   const RuntimeInput::JsonValue* run = root.find("run");
   const RuntimeInput::JsonValue* list = root.find("simulations");
   if (list == nullptr && run != nullptr && run->isObject())
      list = run->find("simulations");

   if (list != nullptr) {
      if (!list->isArray()) throw std::runtime_error("JSON simulations field must be an array");
      for (size_t i = 0; i < list->arrayValue.size(); ++i)
         simulations.push_back(&list->arrayValue[i]);
      return;
   }

   if (root.find("type") != nullptr) {
      simulations.push_back(&root);
      return;
   }

   if (run != nullptr && run->isObject() && run->find("type") != nullptr)
      simulations.push_back(run);
}

static bool boolFromRootOrRun(const RuntimeInput::JsonValue& root, const char* key, bool defaultValue)
{
   const RuntimeInput::JsonValue* run = root.find("run");
   if (run != nullptr && run->isObject()) {
      const RuntimeInput::JsonValue* runValue = run->find(key);
      if (runValue != nullptr) {
         if (!runValue->isBool()) throw std::runtime_error(std::string("JSON field must be boolean: ") + key);
         return runValue->boolValue;
      }
   }
   return root.boolOr(key, defaultValue);
}

static void runConfiguredSimulation(const RuntimeInput::JsonValue& simulation)
{
   if (!simulation.isObject()) throw std::runtime_error("Each simulation entry must be an object");
   if (!simulation.boolOr("enabled", true)) return;

   std::string type = lowerCopy(simulation.stringOr("type", ""));
   if (type == "bulk_yield" || type == "bulk-yield" || type == "bulkyield") {
      printf("SIMULATION: bulk_yield\n"); fflush(stdout);
      BulkYield::run(simulation);
   }
   else if (type == "composite_yield" || type == "composite-yield" || type == "compositeyield") {
      printf("SIMULATION: composite_yield\n"); fflush(stdout);
      CompositeYield::run(simulation);
   }
   else if (type == "composite_image" || type == "composite-image" || type == "compositeimage") {
      printf("SIMULATION: composite_image\n"); fflush(stdout);
      CompositeImage::run(simulation);
   }
   else if (type == "lines_on_layers" || type == "lines-on-layers" || type == "linesonlayers") {
      printf("SIMULATION: lines_on_layers\n"); fflush(stdout);
      LinesOnLayers::run();
   }
   else if (type == "self_tests" || type == "tests" || type == "test_suite") {
      printf("SIMULATION: self_tests\n"); fflush(stdout);
      runSelfTests();
   }
   else {
      throw std::runtime_error("Unknown simulation type in JSON input: " + type);
   }
}

int main(int argc, char** argv)
{
   try {
      initializeRuntime();

      if (argc <= 1) {
         runDefaultSuite();
         return 0;
      }

      RuntimeInput::JsonValue root = RuntimeInput::parseFile(argv[1]);
      if (boolFromRootOrRun(root, "run_tests", false))
         runSelfTests();

      std::vector<const RuntimeInput::JsonValue*> simulations;
      collectSimulations(root, simulations);
      if (simulations.empty())
         throw std::runtime_error("JSON input must define simulations[] or a single object with a type field");

      for (size_t i = 0; i < simulations.size(); ++i)
         runConfiguredSimulation(*simulations[i]);
   }
   catch (const std::exception& ex) {
      printf("CudaMONSEL input error: %s\n", ex.what()); fflush(stdout);
      return 1;
   }

   return 0;
}
