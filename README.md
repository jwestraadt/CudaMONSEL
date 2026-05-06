# CudaMONSEL
Switch to branch double-to-float for latest updates.

<!--
![alt text](https://raw.githubusercontent.com/zengrz/CudaMONSEL/double-to-float/CudaMONSEL/outputs/14.png)
-->
<p align="center"> 
<img src="https://raw.githubusercontent.com/zengrz/CudaMONSEL/double-to-float/CudaMONSEL/BSE0.png"> 
</p>
<p align="center"> 
<img src="https://raw.githubusercontent.com/zengrz/CudaMONSEL/double-to-float/CudaMONSEL/gt0.png"> 
</p>

CudaMONSEL is a full-fledged electron tracker based on first physical principles. Its primary application is to carry out Monte Carlo simulation of SEM Signals. It can be ran on CPU using a thread pool, as well as GPU using the CUDA framework.

CudaMONSEL is a direct port of JMONSEL, a Java version the software built by J.S. Villarrubia and Nicholas Ritchie of NIST. CudaMONSEL aims to speed up the original simulation to mass produce SEM images for ML training. It also added extra functionalities to describe the geometry of the setup. For example, 2D projection of outline of shapes on an arbitrary plane.

## Building

### Requirements

- Windows with Visual Studio 2022 and the C++ desktop workload.
- NVIDIA CUDA Toolkit 13.1. The checked-in Visual Studio project imports `CUDA 13.1.props` and `CUDA 13.1.targets`.
- An NVIDIA GPU compatible with the configured CUDA architecture. The Visual Studio project and CMake build currently default to `sm_89` / `compute_89`; change this to match your GPU when needed.

### Build the executable with Visual Studio or MSBuild

Open `CudaMONSEL/CudaMONSEL.sln` in Visual Studio and build the `Release|x64` configuration.

From a Visual Studio Developer PowerShell or Developer Command Prompt, the same build is:

```powershell
msbuild CudaMONSEL\CudaMONSEL.sln /m /p:Configuration=Release /p:Platform=x64
```

The executable is written under `CudaMONSEL/x64/Release/CudaMONSEL.exe`. Debug builds are written under `CudaMONSEL/x64/Debug/`.

### CMake compile check

The CMake files currently compile the microanalysis utility target and are useful for checking CUDA/CMake configuration:

```powershell
cmake -S CudaMONSEL -B build-cuda13 -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda13 --config Release
```

Use `-DCMAKE_CUDA_ARCHITECTURES=<arch>` for a different GPU architecture, for example `75`, `86`, or `89`.

### Run

Run the built executable from the project directory so relative output paths are predictable:

```powershell
cd CudaMONSEL\CudaMONSEL
..\x64\Release\CudaMONSEL.exe
```

`Main.cu` initializes the scattering data and then runs the enabled tests and simulations. Generated run logs, build directories, binaries, and local CSV outputs are ignored by git.

## Setting Up A New Simulation

1. Create a new pair of files under `CudaMONSEL/CudaMONSEL/gov/nist/nanoscalemetrology/JMONSELTests/`, for example `MySimulation.cuh` and `MySimulation.cu`.
2. Follow the existing namespace pattern:

```cpp
namespace MySimulation
{
   void run();
}
```

3. In the `.cu` file, define the simulation constants first: trajectory count, beam energies, beam size, material density, work function, band gap, Fermi energy, and output file name.
4. Build the material stack with `CompositionT`, `SEmaterialT`, `SelectableElasticSMT`, a continuous slowing-down model such as `JoyLuoNieminenCSDT`, and either `FittedInelSMT` or `TabulatedInelasticSMT`.
5. Build the chamber and sample geometry with the `NISTMonte` and `JMONSEL` shape types. `BulkYield.cu` is the simplest bulk-material example; `LinesOnLayers.cu` is a more complex layered-geometry example.
6. Add the new `.cu` file to `CudaMONSEL/CudaMONSEL/CudaMONSEL.vcxproj` as a `CudaCompile` item and the `.cuh` file as a `ClInclude` item.
7. Include the header in `CudaMONSEL/CudaMONSEL/Main.cu` and add a `RUN("MySimulation", MySimulation::run())` line.
8. Start with a small trajectory count to verify geometry, scattering setup, and output format, then increase trajectories for production runs.

If the simulation uses tabulated JMONSEL inelastic scattering data, install the tables under `C:\Program Files\NIST\JMONSEL\ScatteringTables` or update the hard-coded table paths in the simulation source.

## Citing CudaMONSEL:

If you use CudaMONSEL in your research, please cite with:
```
@misc{villarrubia2015jmonsel,
  title={Scanning electron microscope measurement of width and shape of 10 nm patterned lines using a JMONSEL-modeled library},
  author={J.S. Villarrubia et. al.},
  howpublished={\url{https://ws680.nist.gov/publication/get_pdf.cfm?pub_id=916512}},
  year={2015}
}

@misc{zeng2019cudamonsel,
  title={CudaMONSEL},
  author={Ruizi, Zeng},
  howpublished={\url{https://github.com/zengrz/CudaMONSEL/}},
  year={2019}
}
```
## Citing Amphibian:
Amphibian is an initiative to build a library of data structures and algorithms that can be used on both host and device. Can be useful when transitioning from CPU to CUDA code.

```
@misc{zeng2019amphibian,
  title={Amphibian},
  author={Zeng, Ruizi},
  howpublished={\url{https://github.com/zengrz/Amphibian/}},
  year={2019}
}
```
