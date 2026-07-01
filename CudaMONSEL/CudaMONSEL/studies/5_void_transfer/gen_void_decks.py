"""
Generate CudaMONSEL decks recreating Payton & Mills (2011) Fig. 5:
BSE-coefficient radial profiles of spherical VOIDS (etched-out particles) in a
Rene 104 matrix, over radius R x depth x x accelerating voltage.

Writes studies/5_void_transfer/void_{5,12,20}kV.json (60 sims each = 6 R x 10 x).
Each sim is a 1-D radial eta(r) line scan (r in [0,4R], 306 samples/micron),
7500-equivalent statistics (we use 5000, well past the paper's sigma<0.007
convergence, since the render adds +/-0.024 noise anyway). BSE-only (no SE
tracking) since eta is unaffected. Run each deck, then void_transfer_plot.py.

    python studies/5_void_transfer/gen_void_decks.py     # from the project dir
"""
import json, os

OUT = "studies/5_void_transfer"
DATA = OUT + "/data"

RENE104 = {  # at% (nominal wt% -> at%), <=8 elements for the GPU backend
    "name": "Rene104", "density_kg_m3": 8250.0,
    "work_function_ev": 4.8, "fermi_energy_ev": 8.0, "bandgap_ev": 0.0,
    "secondary_generation_energy_ev": 30.0, "break_energy_ev": 45.0,
    "composition": {"Ni": 49.3, "Co": 20.1, "Cr": 14.5, "Al": 7.2,
                    "Ti": 4.7, "Mo": 2.3, "Ta": 0.8, "W": 0.65},
}
RADII = [25, 75, 125, 175, 225, 275]                 # nm
XFRACS = [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0]  # centroid height / R
KVS = [5, 12, 20]
STEPS_PER_MICRON = 306
BEAM_FWHM_NM = 3.2
SIGMA_NM = BEAM_FWHM_NM / 2.3548                      # Gaussian 1-sigma
TRAJ = 5000


def xtag(xf):
    s = ("p" if xf >= 0 else "m") + ("%.1f" % abs(xf)).replace(".", "")
    return s


def sim(kv, R, xf):
    N = max(2, round(4 * R * STEPS_PER_MICRON / 1000.0))
    center_depth = -xf * R                            # paper x above surface -> depth below
    base = "void_%dkV_R%d_x%s" % (kv, R, xtag(xf))
    return {
        "type": "composite_image", "name": base, "enabled": True,
        "output_csv": "%s/%s.csv" % (DATA, base),
        "beam_energy_ev": kv * 1000.0, "beam_size_nm": round(SIGMA_NM, 4),
        "trajectories_per_pixel": TRAJ,
        "secondary_electron_threshold_ev": 50.0, "histogram_bin_size_ev": 10.0,
        "backend": "gpu", "track_secondaries": False,
        "scattering": {"elastic": "nist_mott", "inelastic": "fitted",
                       "csd": "joy_luo_nieminen", "barrier": "exp_qm"},
        "scan": {"radial": True, "radial_max_nm": 4.0 * R, "nx_pixels": N, "ny_pixels": 1},
        "matrix_phase": RENE104,
        "precipitate": {"shape": "sphere", "radius_nm": float(R),
                        "center_depth_nm": center_depth,
                        "center_x_nm": 0.0, "center_y_nm": 0.0, "void": True},
    }


def main():
    os.makedirs(DATA, exist_ok=True)
    for kv in KVS:
        deck = {"schema_version": 1, "run_tests": False,
                "simulations": [sim(kv, R, xf) for R in RADII for xf in XFRACS]}
        path = "%s/void_%dkV.json" % (OUT, kv)
        with open(path, "w") as f:
            json.dump(deck, f, indent=1)
        print("wrote %s  (%d sims)" % (path, len(deck["simulations"])))


if __name__ == "__main__":
    main()
