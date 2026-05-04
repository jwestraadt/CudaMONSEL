/*
* - without length, the array parameter (eg const double a[]) is always 3 dimensional
* - use "auto" keyword for pointers ONLY
*/

#include <stdio.h>

#include <cuda_runtime.h>

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

#include "ImageUtil.h"

int main()
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
   RUN("LinesOnLayers", LinesOnLayers::run())

   return 0;
}
