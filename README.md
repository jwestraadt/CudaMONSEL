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

With no arguments, `Main.cu` initializes the scattering data and then runs the built-in test and simulation sequence.

To run from JSON input, pass a config file:

```powershell
cd CudaMONSEL\CudaMONSEL
..\x64\Release\CudaMONSEL.exe input.json
```

The runtime JSON format uses a `simulations` array. Each entry has a `type` field. Supported runtime types are:

- `bulk_yield`: fully configurable from JSON, including trajectory count, beam energies, output CSV, and phase material properties.
- `lines_on_layers`: dispatches the existing hard-coded lines-on-layers simulation.
- `self_tests`: runs the built-in test suite.

`CudaMONSEL/CudaMONSEL/input.json` contains a Ni superalloy gamma/gamma-prime `bulk_yield` example. Generated run logs, build directories, binaries, and local CSV outputs are ignored by git.

## BulkYield Example

`BulkYield` simulates secondary electron (SE) and backscattered electron (BSE) yields for bulk homogeneous material phases as a function of beam energy. The included `input.json` models a Ni-based superalloy with two phases (gamma and gamma-prime) and is the recommended starting point for new material studies.

### Running the included example

```powershell
cd CudaMONSEL\CudaMONSEL
..\x64\Release\CudaMONSEL.exe input.json
```

Progress is printed to stdout per beam energy point. When finished, results are written to the CSV named in `output_csv` (default: `BulkYield_output.csv`):

```
phase,BeamE_eV,BSE_yield,SE_yield,total_yield
gamma,200,0.0312,0.1847,0.2159
gamma,500,0.0421,0.1523,0.1944
...
```

### input.json reference

#### Top-level fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | integer | Must be `1` |
| `run_tests` | boolean | When `true`, runs the built-in test suite before simulations |
| `simulations` | array | List of simulation entries executed in order |

#### bulk_yield simulation fields

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | — | Must be `"bulk_yield"` |
| `name` | string | `"Ni superalloy gamma/gamma-prime"` | Label used in console output |
| `enabled` | boolean | `true` | Set to `false` to skip this entry without removing it |
| `output_csv` | string | `"BulkYield_output.csv"` | Output file path (relative to the working directory) |
| `trajectories` | integer | `5000` | Number of electron trajectories per beam energy point per phase |
| `beam_size_nm` | number | `0.5` | Gaussian beam 1-sigma radius in nm |
| `secondary_electron_threshold_ev` | number | `50.0` | Electrons exiting below this energy (eV) count as SE; above as BSE |
| `histogram_bin_size_ev` | number | `10.0` | Energy bin width (eV) used for the exit-energy histogram |

#### beam_energies_ev

An array of incident beam energies in eV. A separate yield point is computed for each energy for every phase.

```json
"beam_energies_ev": [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0]
```

#### scattering models

The `scattering` object selects the physical models used in the Monte Carlo scatter stack. All four models must be specified; only the combination shown below is currently supported.

```json
"scattering": {
  "elastic":   "nist_mott",
  "inelastic": "fitted",
  "csd":       "joy_luo_nieminen",
  "barrier":   "exp_qm"
}
```

| Key | Supported value | Model description |
|---|---|---|
| `elastic` | `nist_mott` | NIST Mott elastic cross-sections (`SelectableElasticSM` + `NISTMottRS`) |
| `inelastic` | `fitted` | `FittedInelSM` — parameterized SE generation; calibrated for organics, semi-quantitative for metals |
| `csd` | `joy_luo_nieminen` | Joy–Luo–Nieminen continuous slowing-down model (`JoyLuoNieminenCSD`) |
| `barrier` | `exp_qm` | Exponential quantum-mechanical surface barrier (`ExpQMBarrierSM`) |

> **Note:** `FittedInelSM` gives a realistic SE yield trend for metals but absolute values should be treated as semi-quantitative. Replace with a tabulated inelastic model and JMONSEL scattering tables for calibrated results.

#### phases — material parameters

Each entry in the `phases` array defines one bulk material phase. All energy values are in eV and density in kg/m³.

| Field | Required | Description |
|---|---|---|
| `name` | yes | Phase label used in CSV output |
| `density_kg_m3` | yes | Mass density in kg/m³ |
| `work_function_ev` | yes | Surface work function in eV |
| `fermi_energy_ev` | yes | Fermi energy in eV (set to actual value for metals) |
| `bandgap_ev` | no (default `0`) | Band gap in eV; use `0` for metals |
| `secondary_generation_energy_ev` | yes | Mean energy per SE generation event in eV (~30 eV for metals, ~65 eV for organics) |
| `break_energy_ev` | yes | CSD break energy in eV — transition point in the Joy–Luo–Nieminen model |
| `composition` | yes (or `elements`) | Object mapping element symbol to mole fraction (unnormalized weights are normalised internally) |

The `composition` object uses standard element symbols as keys:

```json
"composition": {
  "Ni": 63.0,
  "Cr":  8.0,
  "Co": 10.0,
  "W":   6.0,
  "Re":  4.0,
  "Al":  6.0,
  "Ta":  3.0
}
```

Alternatively, an `elements` array may be used with explicit `symbol` and `fraction` entries:

```json
"elements": [
  { "symbol": "Ni", "fraction": 63.0 },
  { "symbol": "Al", "fraction": 12.0 }
]
```

#### Full example — Ni superalloy gamma/gamma-prime

```json
{
  "schema_version": 1,
  "run_tests": false,
  "simulations": [
    {
      "type": "bulk_yield",
      "name": "Ni superalloy gamma/gamma-prime",
      "enabled": true,
      "output_csv": "BulkYield_output.csv",
      "trajectories": 5000,
      "beam_size_nm": 0.5,
      "secondary_electron_threshold_ev": 50.0,
      "histogram_bin_size_ev": 10.0,
      "scattering": {
        "elastic":   "nist_mott",
        "inelastic": "fitted",
        "csd":       "joy_luo_nieminen",
        "barrier":   "exp_qm"
      },
      "beam_energies_ev": [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0],
      "phases": [
        {
          "name": "gamma",
          "density_kg_m3": 8700.0,
          "work_function_ev": 5.15,
          "fermi_energy_ev": 8.8,
          "bandgap_ev": 0.0,
          "secondary_generation_energy_ev": 30.0,
          "break_energy_ev": 45.0,
          "composition": { "Ni": 63.0, "Cr": 8.0, "Co": 10.0, "W": 6.0, "Re": 4.0, "Al": 6.0, "Ta": 3.0 }
        },
        {
          "name": "gamma_prime",
          "density_kg_m3": 8200.0,
          "work_function_ev": 4.9,
          "fermi_energy_ev": 7.5,
          "bandgap_ev": 0.0,
          "secondary_generation_energy_ev": 30.0,
          "break_energy_ev": 45.0,
          "composition": { "Ni": 75.0, "Al": 12.0, "Ti": 5.0, "Ta": 5.0, "Cr": 3.0 }
        }
      ]
    }
  ]
}
```

### Adapting to a new material

1. Copy `input.json` and change `name` and `output_csv`.
2. Update `composition` with your element symbols and mole fractions (values are normalised internally).
3. Set `density_kg_m3`, `work_function_ev`, and `fermi_energy_ev` from literature or DFT.
4. For metals keep `bandgap_ev` at `0` and `secondary_generation_energy_ev` at `~30`; for insulators/organics use `~65`.
5. Start with a low `trajectories` count (500–1000) to verify setup, then increase to 5000+ for production statistics.
6. Adjust `beam_energies_ev` to cover the energy range relevant to your SEM operating conditions.

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
