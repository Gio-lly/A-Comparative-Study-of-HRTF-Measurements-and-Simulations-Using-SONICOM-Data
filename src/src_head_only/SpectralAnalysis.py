import sofar as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import csv
import math
import re

# ============================================================
# CONFIGURATION
# ============================================================

files = {
    "sim_nostra": "/home/capstone/Downloads/P067/SOFA_P067_Right_merged/HRIR_SONICOM_grid_48000.sofa",
    "sim_sonicom": "/home/capstone/Downloads/P067/HRIR_SONICOM_48000.sofa",
    "mis_sonicom": "/home/capstone/Downloads/P067/P0067_FreeFieldComp_48kHz.sofa",
}

# Angular convention used in the SOFA/SONICOM files:
# azimuth 0°    = front
# azimuth +90°  = listener's left side
# azimuth -90°  = listener's right side
# azimuth ±180° = back
AZIMUTH_CONVENTION_NOTE = "SOFA/SONICOM: +90 deg = left, -90 deg = right"

# Pairs to compare in the difference maps and CSVs.
comparison_pairs = [
    ("sim_nostra", "mis_sonicom"),
    ("sim_sonicom", "mis_sonicom"),
    ("sim_nostra", "sim_sonicom"),
]

# FFT and display limits.
NFFT = 4096
FREQ_MIN_PLOT = 100.0
FREQ_MAX_PLOT = 20000.0

# ============================================================
# COMMON COLOR SCALES FOR THE MAPS
# ============================================================
# Goal:
# use the same dB scale for all absolute HRTF maps
# and the same symmetric dB scale for all difference maps.
#
# If HRTF_COLOR_LIMITS_DB = None, the common limits are estimated
# automatically across all generated HRTF maps.
# If you want to fix them manually, use for example:
# HRTF_COLOR_LIMITS_DB = (-40.0, 20.0)
HRTF_COLOR_LIMITS_DB = None

# Percentiles used to estimate the common scale for the absolute HRTF maps.
# Using percentiles avoids a few outliers forcing an overly wide colorbar.
COMMON_HRTF_COLOR_PERCENTILES = (1.0, 99.0)

# If DIFFERENCE_COLOR_LIMIT_DB = None, the common symmetric limit for the
# difference maps is estimated automatically across all differences.
# If you want to fix it manually, use for example:
# DIFFERENCE_COLOR_LIMIT_DB = 30.0
DIFFERENCE_COLOR_LIMIT_DB = None

# Percentile used to estimate the common limit for the difference maps.
# The scale will be symmetric: [-vmax, +vmax].
COMMON_DIFFERENCE_COLOR_PERCENTILE = 99.0

# Enable/disable the common scales.
USE_COMMON_HRTF_COLOR_SCALE = True
USE_COMMON_DIFFERENCE_COLOR_SCALE = True

# Internal variables computed automatically before generating the maps.
COMMON_HRTF_COLOR_LIMITS_DB = None
COMMON_DIFFERENCE_COLOR_LIMIT_DB = None

# Frequencies used for the preliminary descriptive metrics.
# These do NOT replace LSD/ITD/ILD computed with SAM.
FREQ_MIN_METRIC = 100.0
FREQ_MAX_METRIC = 20000.0

# ============================================================
# GLOBAL BROADBAND NORMALIZATION
# ============================================================

# If True, applies a global offset in dB to some datasets before
# generating HRTF plots, difference maps and spectral metric CSVs.
# The goal is to remove a broadband gain offset between measurements and simulations,
# so as to compare mainly the direction-dependent spectral shape.
APPLY_BROADBAND_GAIN_NORMALIZATION = True

# Dataset used as the level reference.
# Here it's best to use the official SONICOM simulation as the reference.
GAIN_NORMALIZATION_REFERENCE_DATASET = "sim_sonicom"

# Dataset(s) to which the estimated offset is applied.
# By default we only normalize the measured HRTF relative to the simulated SONICOM one.
GAIN_NORMALIZATION_TARGET_DATASETS = ["mis_sonicom"]

# Band used to estimate the gain offset.
# Avoid frequencies that are too low or too high, where the measurement/simulation may
# be less stable or have different limitations.
GAIN_NORMALIZATION_FREQ_LOW = 500.0
GAIN_NORMALIZATION_FREQ_HIGH = 16000.0

# Bands used to summarize the average spectral difference.
frequency_bands = [
    ("100_500_Hz", 100.0, 500.0),
    ("500_1500_Hz", 500.0, 1500.0),
    ("1500_3000_Hz", 1500.0, 3000.0),
    ("3000_6000_Hz", 3000.0, 6000.0),
    ("6000_10000_Hz", 6000.0, 10000.0),
    ("10000_16000_Hz", 10000.0, 16000.0),
    ("16000_20000_Hz", 16000.0, 20000.0),
]

# Output controls.
SAVE_LOCAL_HRIR_PLOTS = True
SAVE_LOCAL_HRTF_PLOTS = True
SAVE_LOCAL_DIFFERENCE_PLOTS = True
SAVE_GRID_HRTF_PLOTS = True
SAVE_HRTF_MAPS = True
SAVE_DIFFERENCE_MAPS = True
SAVE_SANITY_CHECK_CSV = True

# Print a quick check on the left/right convention using the HRIR energy.
PRINT_AZIMUTH_SANITY_CHECK = True

# To avoid generating too many images, the creation of local plots can be limited
# to only some target sets.
# Set to None to generate them for all.
LOCAL_PLOTS_ONLY_FOR = None
# Example:
# LOCAL_PLOTS_ONLY_FOR = {"principal", "horizontal_30deg"}

# ============================================================
# OUTPUT FOLDER
# ============================================================

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

output_root = script_dir / "plots_compare_multi_directions_broadband_norm"
output_root.mkdir(parents=True, exist_ok=True)

print(f"Output principale: {output_root}")
print(f"Convenzione azimuth: {AZIMUTH_CONVENTION_NOTE}")

# ============================================================
# TARGET SET DEFINITION
# ============================================================

def make_horizontal_targets(step_deg=30):
    """
    Horizontal plane: elevation = 0°, azimuth from -180° to 180°.

    SONICOM/SOFA convention:
    +90° = left, -90° = right.
    """
    azimuths = list(range(-180, 181, step_deg))
    return [
        (f"az_{az:+04d}_el_000", float(az), 0.0)
        for az in azimuths
    ]


def make_elevation_targets(name_prefix, azimuth, elevations):
    """
    Vertical plane at a fixed azimuth.
    """
    return [
        (
            f"{name_prefix}_az_{int(azimuth):+04d}_el_{int(el):+04d}",
            float(azimuth),
            float(el),
        )
        for el in elevations
    ]


# SONICOM/SOFA convention correction:
# +90° is the left side, -90° is the right side.
principal_targets = [
    ("frontale", 0.0, 0.0),
    ("laterale_sinistra", 90.0, 0.0),
    ("laterale_destra", -90.0, 0.0),
    ("posteriore", 180.0, 0.0),
]

horizontal_targets = make_horizontal_targets(step_deg=30)

# Actual elevations of the released SONICOM grid.
# We avoid ±15°, which do not belong to the standard grid and could
# introduce ambiguous nearest-neighbour matching between ±10° and ±20°.
SONICOM_ELEVATIONS = [-45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 75, 90]


median_front_targets = make_elevation_targets(
    "median_front",
    azimuth=0.0,
    elevations=SONICOM_ELEVATIONS,
)

median_back_targets = make_elevation_targets(
    "median_back",
    azimuth=180.0,
    elevations=SONICOM_ELEVATIONS,
)

# Convention correction:
# lateral_left  -> azimuth +90°
# lateral_right -> azimuth -90°
lateral_left_targets = make_elevation_targets(
    "lateral_left",
    azimuth=90.0,
    elevations=SONICOM_ELEVATIONS,
)

lateral_right_targets = make_elevation_targets(
    "lateral_right",
    azimuth=-90.0,
    elevations=SONICOM_ELEVATIONS,
)

target_sets = {
    "principal": principal_targets,
    "horizontal_30deg": horizontal_targets,
    "median_front": median_front_targets,
    "median_back": median_back_targets,
    "lateral_left": lateral_left_targets,
    "lateral_right": lateral_right_targets,
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_filename(text):
    """
    Makes a string usable as a file name.
    """
    text = str(text)
    text = text.replace("+", "p")
    text = text.replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_\.]+", "_", text)
    return text


def normalize_azimuth_deg(az):
    """
    Normalizes the azimuth to the range [-180, 180).
    180° and -180° represent the same direction.
    """
    return ((az + 180.0) % 360.0) - 180.0


def az_el_to_unit_vector(az_deg, el_deg):
    """
    Converts spherical coordinates to a unit cartesian vector.

    Convention consistent with SOFA/SONICOM:
    - x forward;
    - y towards the left;
    - z upward;
    - positive azimuth towards the left.
    """
    az = np.deg2rad(normalize_azimuth_deg(az_deg))
    el = np.deg2rad(el_deg)

    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    return np.array([x, y, z], dtype=float)


def angular_distance_spherical_deg(source_positions, az_target, el_target):
    """
    Computes the actual angular distance on the sphere between all SOFA positions
    and a target direction.

    source_positions must have at least two columns:
    column 0 = azimuth,
    column 1 = elevation.
    """
    pos = np.asarray(source_positions)
    az = pos[:, 0]
    el = pos[:, 1]

    target_vec = az_el_to_unit_vector(az_target, el_target)

    az_norm = np.array([normalize_azimuth_deg(a) for a in az])
    az_rad = np.deg2rad(az_norm)
    el_rad = np.deg2rad(el)

    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)

    source_vecs = np.column_stack([x, y, z])

    dots = source_vecs @ target_vec
    dots = np.clip(dots, -1.0, 1.0)

    distances = np.rad2deg(np.arccos(dots))
    return distances


def find_nearest_index(sofa, az_target, el_target):
    """
    Finds the index of the SOFA direction closest to the target.

    Returns:
    - index,
    - actual position,
    - angular distance in degrees.
    """
    pos = np.asarray(sofa.SourcePosition)
    distances = angular_distance_spherical_deg(pos, az_target, el_target)
    idx = int(np.argmin(distances))
    return idx, pos[idx], float(distances[idx])


def mag_db(hrir, nfft=NFFT):
    """
    HRTF magnitude in dB from an HRIR.
    """
    H = np.fft.rfft(hrir, n=nfft)
    return 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))


def hrir_energy_db(hrir):
    """
    Energy of an HRIR in dB.
    Used only for the sanity check of the left/right convention.
    """
    return 10.0 * np.log10(np.sum(np.asarray(hrir) ** 2) + 1e-20)


def get_freq_axis(fs, nfft=NFFT):
    return np.fft.rfftfreq(nfft, 1.0 / fs)


def save_figure(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Salvata: {path}")
    plt.close()


def get_ear_name(ear):
    if ear == 0:
        return "left"
    if ear == 1:
        return "right"
    return f"ear_{ear}"


def get_hrir(sofa, idx, ear, min_len):
    return np.asarray(sofa.Data_IR[idx, ear, :min_len], dtype=float)


def get_direction_axis_for_target_set(set_name, targets):
    """
    Decides which spatial variable to use as the y-axis in the maps.

    - Horizontal plane: y = azimuth.
    - Vertical planes: y = elevation.
    - Principal directions: y = index/category.
    """
    if "horizontal" in set_name:
        values = np.array([az for _, az, _ in targets], dtype=float)
        axis_label = "Azimuth [deg] (+ = left, - = right)"
        tick_labels = [f"{int(v)}" for v in values]
        return values, axis_label, tick_labels

    if "median" in set_name or "lateral" in set_name:
        values = np.array([el for _, _, el in targets], dtype=float)
        axis_label = "Elevation [deg]"
        tick_labels = [f"{int(v)}" for v in values]
        return values, axis_label, tick_labels

    values = np.arange(len(targets), dtype=float)
    axis_label = "Direction"
    tick_labels = [target_label for target_label, _, _ in targets]
    return values, axis_label, tick_labels


def centers_to_edges(values):
    """
    Converts bin centers to edges for pcolormesh.

    Example:
    if Z has shape (N_y, N_x), then pcolormesh requires:
    - x_edges with length N_x + 1;
    - y_edges with length N_y + 1.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])

    edges = np.zeros(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])

    return edges


def metric_frequency_mask(freq):
    return (freq >= FREQ_MIN_METRIC) & (freq <= FREQ_MAX_METRIC)


def plot_frequency_axis_limits(fs):
    fmax = min(FREQ_MAX_PLOT, fs / 2.0)
    plt.xlim(FREQ_MIN_PLOT, fmax)


def make_difference_norm(Z):
    """
    Creates a symmetric normalization centered on 0 dB for the difference maps.

    If USE_COMMON_DIFFERENCE_COLOR_SCALE is True, all difference maps
    use the same global limit COMMON_DIFFERENCE_COLOR_LIMIT_DB.
    """
    if USE_COMMON_DIFFERENCE_COLOR_SCALE and COMMON_DIFFERENCE_COLOR_LIMIT_DB is not None:
        vmax_abs = float(COMMON_DIFFERENCE_COLOR_LIMIT_DB)
    elif DIFFERENCE_COLOR_LIMIT_DB is not None:
        vmax_abs = float(DIFFERENCE_COLOR_LIMIT_DB)
    else:
        vmax_abs = float(np.nanpercentile(np.abs(Z), COMMON_DIFFERENCE_COLOR_PERCENTILE))
        if not np.isfinite(vmax_abs) or vmax_abs <= 0.0:
            vmax_abs = 1.0

    return TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)


def estimate_broadband_gain_offsets(
    hrtf_cache,
    target_sets,
    freq,
    reference_dataset,
    target_datasets,
    f_low,
    f_high,
):
    """
    Estimates a global broadband offset in dB for each target dataset.

    For each target_dataset it computes:
        offset = mean( H_reference - H_target )
    averaging over:
        - all target sets,
        - all directions included in the target sets,
        - both ears,
        - the band [f_low, f_high].

    If offset > 0, the target dataset is raised.
    If offset < 0, the target dataset is lowered.
    """
    if reference_dataset not in sofas:
        raise ValueError(f"Dataset di riferimento non trovato: {reference_dataset}")

    mask = (freq >= f_low) & (freq <= f_high)
    if not np.any(mask):
        raise ValueError(
            f"La banda di normalizzazione {f_low}-{f_high} Hz non contiene bin FFT validi."
        )

    offsets_db = {}
    rows = []

    for target_dataset in target_datasets:
        if target_dataset not in sofas:
            raise ValueError(f"Dataset target da normalizzare non trovato: {target_dataset}")

        local_offsets = []

        for set_name, targets in target_sets.items():
            for target_label, az_t, el_t in targets:
                for ear in [0, 1]:
                    H_ref = hrtf_cache[set_name][target_label][reference_dataset][ear]
                    H_tar = hrtf_cache[set_name][target_label][target_dataset][ear]

                    offset_here = float(np.mean(H_ref[mask] - H_tar[mask]))
                    local_offsets.append(offset_here)

                    rows.append({
                        "target_dataset": target_dataset,
                        "reference_dataset": reference_dataset,
                        "target_set": set_name,
                        "target_label": target_label,
                        "target_az_deg": az_t,
                        "target_el_deg": el_t,
                        "ear": get_ear_name(ear),
                        "offset_db_local": offset_here,
                        "normalization_f_low_hz": f_low,
                        "normalization_f_high_hz": f_high,
                    })

        offsets_db[target_dataset] = float(np.mean(local_offsets))

    return offsets_db, rows


def apply_broadband_gain_offsets(hrtf_cache, offsets_db):
    """
    Applies the global dB offsets to the HRTF cache in-place.

    Note: this normalization only modifies the HRTF magnitudes in dB
    used for plots, differences and spectral metrics. The time-domain
    HRIRs remain raw/unnormalized.
    """
    for dataset_name, offset_db in offsets_db.items():
        for set_name in hrtf_cache:
            for target_label in hrtf_cache[set_name]:
                for ear in hrtf_cache[set_name][target_label][dataset_name]:
                    hrtf_cache[set_name][target_label][dataset_name][ear] = (
                        hrtf_cache[set_name][target_label][dataset_name][ear] + offset_db
                    )


# ============================================================
# SOFA FILE READING
# ============================================================

print("\nLettura dei file SOFA...")

sofas = {}
for name, path in files.items():
    print(f"  {name}: {path}")
    sofas[name] = sf.read_sofa(path)

# Reference sample rate.
fs_values = {}
for name, sofa in sofas.items():
    fs_values[name] = float(np.atleast_1d(sofa.Data_SamplingRate).squeeze())

reference_name = list(sofas.keys())[0]
fs = fs_values[reference_name]

print("\nSample rate:")
for name, value in fs_values.items():
    print(f"  {name}: {value} Hz")

if not all(np.isclose(value, fs) for value in fs_values.values()):
    print("\nATTENZIONE: i file non hanno tutti lo stesso sample rate.")
    print("Lo script prosegue usando il sample rate del primo file come riferimento per l'asse frequenziale.")

# Common minimum length.
min_len = min(sofa.Data_IR.shape[2] for sofa in sofas.values())
print(f"\nLunghezza minima comune usata per HRIR: {min_len} campioni")

freq = get_freq_axis(fs, NFFT)
freq_plot_mask = (freq >= FREQ_MIN_PLOT) & (freq <= min(FREQ_MAX_PLOT, fs / 2.0))

# Quick check of the dimensions.
print("\nDimensioni Data_IR:")
for name, sofa in sofas.items():
    print(f"  {name}: {sofa.Data_IR.shape}")

# ============================================================
# AZIMUTH CONVENTION SANITY CHECK
# ============================================================

sanity_rows = []

if PRINT_AZIMUTH_SANITY_CHECK or SAVE_SANITY_CHECK_CSV:
    print("\n=== SANITY CHECK AZIMUTH CONVENTION ===")
    print("Atteso per convenzione SONICOM/SOFA:")
    print("  azimuth +90°  -> lato sinistro  -> energia left > energia right")
    print("  azimuth -90°  -> lato destro    -> energia right > energia left")

    for dataset_name, sofa in sofas.items():
        if PRINT_AZIMUTH_SANITY_CHECK:
            print(f"\nDataset: {dataset_name}")

        for az_test in [90.0, -90.0]:
            idx, pos, dist = find_nearest_index(sofa, az_test, 0.0)

            h_left = get_hrir(sofa, idx, 0, min_len)
            h_right = get_hrir(sofa, idx, 1, min_len)

            e_left = hrir_energy_db(h_left)
            e_right = hrir_energy_db(h_right)
            lr_diff = e_left - e_right

            if az_test > 0:
                expected_side = "left"
                expected_condition = "L-R > 0"
            else:
                expected_side = "right"
                expected_condition = "L-R < 0"

            if PRINT_AZIMUTH_SANITY_CHECK:
                print(
                    f"target az={az_test:+.0f}°, "
                    f"matched=({pos[0]:.1f}°, {pos[1]:.1f}°), "
                    f"E_left={e_left:.2f} dB, "
                    f"E_right={e_right:.2f} dB, "
                    f"L-R={lr_diff:.2f} dB, "
                    f"expected side={expected_side}"
                )

            sanity_rows.append({
                "dataset": dataset_name,
                "target_az_deg": az_test,
                "target_el_deg": 0.0,
                "matched_index": idx,
                "matched_az_deg": float(pos[0]),
                "matched_el_deg": float(pos[1]),
                "angular_distance_deg": dist,
                "energy_left_db": e_left,
                "energy_right_db": e_right,
                "left_minus_right_db": lr_diff,
                "expected_side": expected_side,
                "expected_condition": expected_condition,
                "azimuth_convention": AZIMUTH_CONVENTION_NOTE,
            })

if SAVE_SANITY_CHECK_CSV and sanity_rows:
    sanity_csv = output_root / "azimuth_convention_sanity_check.csv"
    with open(sanity_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sanity_rows)
    print(f"\nCSV sanity check convenzione salvato in: {sanity_csv}")
else:
    sanity_csv = None

# ============================================================
# DIRECTION MATCHING
# ============================================================

print("\nMatching delle direzioni target...")

matched = {}
selected_rows = []

for set_name, targets in target_sets.items():
    matched[set_name] = {}

    print(f"\n=== Target set: {set_name} ===")

    for target_label, az_t, el_t in targets:
        matched[set_name][target_label] = {}

        print(f"\nTarget: {target_label} | az={az_t}°, el={el_t}°")

        for dataset_name, sofa in sofas.items():
            idx, effective_pos, angular_dist = find_nearest_index(sofa, az_t, el_t)

            matched[set_name][target_label][dataset_name] = {
                "index": idx,
                "position": effective_pos,
                "angular_distance_deg": angular_dist,
                "target_az": az_t,
                "target_el": el_t,
            }

            effective_az = float(effective_pos[0])
            effective_el = float(effective_pos[1])

            print(
                f"  {dataset_name}: index={idx}, "
                f"pos=({effective_az:.2f}, {effective_el:.2f}), "
                f"dist={angular_dist:.3f}°"
            )

            selected_rows.append({
                "target_set": set_name,
                "target_label": target_label,
                "target_az_deg": az_t,
                "target_el_deg": el_t,
                "dataset": dataset_name,
                "matched_index": idx,
                "matched_az_deg": effective_az,
                "matched_el_deg": effective_el,
                "angular_distance_deg": angular_dist,
                "azimuth_convention": AZIMUTH_CONVENTION_NOTE,
            })

# Save CSV of the actual directions.
selected_csv = output_root / "selected_directions_all_sets.csv"
with open(selected_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
    writer.writeheader()
    writer.writerows(selected_rows)

print(f"\nCSV direzioni selezionate salvato in: {selected_csv}")

# ============================================================
# HRTF MAGNITUDE CACHE
# ============================================================

print("\nCalcolo cache HRTF magnitude...")

# Structure:
# hrtf_cache[set_name][target_label][dataset_name][ear] = mag_db array
hrtf_cache = {}

for set_name, targets in target_sets.items():
    hrtf_cache[set_name] = {}

    for target_label, _, _ in targets:
        hrtf_cache[set_name][target_label] = {}

        for dataset_name, sofa in sofas.items():
            hrtf_cache[set_name][target_label][dataset_name] = {}

            idx = matched[set_name][target_label][dataset_name]["index"]

            for ear in [0, 1]:
                hrir = get_hrir(sofa, idx, ear, min_len)
                hrtf_cache[set_name][target_label][dataset_name][ear] = mag_db(hrir, NFFT)

# ============================================================
# GLOBAL BROADBAND NORMALIZATION OF THE HRTF CACHE
# ============================================================

gain_norm_csv = None
gain_offsets_db = {}

if APPLY_BROADBAND_GAIN_NORMALIZATION:
    print("\n=== NORMALIZZAZIONE BROADBAND GLOBALE ===")
    print(f"Reference dataset: {GAIN_NORMALIZATION_REFERENCE_DATASET}")
    print(f"Target datasets: {GAIN_NORMALIZATION_TARGET_DATASETS}")
    print(
        f"Banda usata per stimare l'offset: "
        f"{GAIN_NORMALIZATION_FREQ_LOW:.0f}-{GAIN_NORMALIZATION_FREQ_HIGH:.0f} Hz"
    )

    gain_offsets_db, gain_normalization_rows = estimate_broadband_gain_offsets(
        hrtf_cache=hrtf_cache,
        target_sets=target_sets,
        freq=freq,
        reference_dataset=GAIN_NORMALIZATION_REFERENCE_DATASET,
        target_datasets=GAIN_NORMALIZATION_TARGET_DATASETS,
        f_low=GAIN_NORMALIZATION_FREQ_LOW,
        f_high=GAIN_NORMALIZATION_FREQ_HIGH,
    )

    for dataset_name, offset_db in gain_offsets_db.items():
        print(
            f"Offset globale applicato a {dataset_name}: "
            f"{offset_db:+.2f} dB "
            f"(rispetto a {GAIN_NORMALIZATION_REFERENCE_DATASET})"
        )

    apply_broadband_gain_offsets(hrtf_cache, gain_offsets_db)

    # Save both the local offsets used for the estimate and the final global offset.
    gain_norm_csv = output_root / "broadband_gain_normalization_offsets.csv"
    fieldnames = list(gain_normalization_rows[0].keys()) + ["offset_db_global_applied"]

    with open(gain_norm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in gain_normalization_rows:
            row_out = dict(row)
            row_out["offset_db_global_applied"] = gain_offsets_db[row["target_dataset"]]
            writer.writerow(row_out)

    print(f"CSV normalizzazione broadband salvato in: {gain_norm_csv}")
    print(
        "Nota: la normalizzazione viene applicata a plot HRTF, mappe e metriche spettrali. "
        "I plot HRIR nel tempo restano non normalizzati."
    )
else:
    print("\nNormalizzazione broadband globale disattivata.")

# ============================================================
# LOCAL PLOTS: HRIR, HRTF, DIFFERENCES
# ============================================================

def should_make_local_plots(set_name):
    if LOCAL_PLOTS_ONLY_FOR is None:
        return True
    return set_name in LOCAL_PLOTS_ONLY_FOR


print("\nGenerazione plot locali...")

for set_name, targets in target_sets.items():

    if not should_make_local_plots(set_name):
        continue

    local_dir = output_root / set_name / "local_plots"
    local_dir.mkdir(parents=True, exist_ok=True)

    for target_label, az_t, el_t in targets:
        target_safe = safe_filename(target_label)

        for ear in [0, 1]:
            ear_name = get_ear_name(ear)

            # ------------------------------------------------
            # Time-domain HRIR
            # ------------------------------------------------
            if SAVE_LOCAL_HRIR_PLOTS:
                plt.figure(figsize=(10, 4))

                for dataset_name, sofa in sofas.items():
                    idx = matched[set_name][target_label][dataset_name]["index"]
                    hrir = get_hrir(sofa, idx, ear, min_len)
                    plt.plot(hrir, label=dataset_name)

                plt.title(
                    f"HRIR - {set_name} - {target_label} "
                    f"(target az={az_t}°, el={el_t}°) - {ear_name}"
                )
                plt.xlabel("Campione")
                plt.ylabel("Ampiezza")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                save_figure(local_dir / f"HRIR_{target_safe}_{ear_name}.png")

            # ------------------------------------------------
            # HRTF magnitude
            # ------------------------------------------------
            if SAVE_LOCAL_HRTF_PLOTS:
                plt.figure(figsize=(10, 4))

                for dataset_name in sofas.keys():
                    H = hrtf_cache[set_name][target_label][dataset_name][ear]
                    plt.plot(freq, H, label=dataset_name)

                plt.title(
                    f"HRTF magnitude - {set_name} - {target_label} "
                    f"(target az={az_t}°, el={el_t}°) - {ear_name}"
                )
                plt.xlabel("Frequenza [Hz]")
                plt.ylabel("Magnitudine [dB]")
                plt.xscale("log")
                plot_frequency_axis_limits(fs)
                plt.grid(True, which="both")
                plt.legend()
                plt.tight_layout()

                save_figure(local_dir / f"HRTF_{target_safe}_{ear_name}.png")

            # ------------------------------------------------
            # Spectral differences between pairs
            # ------------------------------------------------
            if SAVE_LOCAL_DIFFERENCE_PLOTS:
                plt.figure(figsize=(10, 4))

                for a, b in comparison_pairs:
                    H_a = hrtf_cache[set_name][target_label][a][ear]
                    H_b = hrtf_cache[set_name][target_label][b][ear]
                    delta = H_a - H_b
                    plt.plot(freq, delta, label=f"{a} - {b}")

                plt.axhline(0.0, linestyle="--", linewidth=1.0)
                plt.title(
                    f"Delta HRTF magnitude - {set_name} - {target_label} "
                    f"(target az={az_t}°, el={el_t}°) - {ear_name}"
                )
                plt.xlabel("Frequenza [Hz]")
                plt.ylabel("Differenza [dB]")
                plt.xscale("log")
                plot_frequency_axis_limits(fs)
                plt.grid(True, which="both")
                plt.legend()
                plt.tight_layout()

                save_figure(local_dir / f"Delta_HRTF_{target_safe}_{ear_name}.png")

# ============================================================
# COMPACT MULTI-DIRECTION FIGURES
# ============================================================

def plot_grid_hrtf_for_set(set_name, targets, ear):
    """
    Figure with a grid of subplots.
    Each subplot corresponds to a target direction.
    """
    n_targets = len(targets)
    ncols = 4
    nrows = int(math.ceil(n_targets / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for i, (target_label, az_t, el_t) in enumerate(targets):
        ax = axes[i]

        for dataset_name in sofas.keys():
            H = hrtf_cache[set_name][target_label][dataset_name][ear]
            ax.plot(freq, H, label=dataset_name)

        ax.set_xscale("log")
        ax.set_xlim(FREQ_MIN_PLOT, min(FREQ_MAX_PLOT, fs / 2.0))
        ax.grid(True, which="both")
        ax.set_title(f"{target_label}\naz={az_t:.0f}°, el={el_t:.0f}°", fontsize=9)

        if i % ncols == 0:
            ax.set_ylabel("Mag [dB]")

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Freq [Hz]")

    for j in range(n_targets, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))

    ear_name = get_ear_name(ear)
    fig.suptitle(f"HRTF magnitude grid - {set_name} - {ear_name}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    grid_dir = output_root / set_name / "grid_plots"
    grid_dir.mkdir(parents=True, exist_ok=True)

    save_figure(grid_dir / f"HRTF_grid_{set_name}_{ear_name}.png")


if SAVE_GRID_HRTF_PLOTS:
    print("\nGenerazione figure compatte multi-direzione...")

    for set_name, targets in target_sets.items():
        for ear in [0, 1]:
            plot_grid_hrtf_for_set(set_name, targets, ear)

# ============================================================
# HRTF MAPS AND DIFFERENCE MAPS
# ============================================================

def build_hrtf_matrix(set_name, targets, dataset_name, ear):
    """
    HRTF matrix:
    rows = target directions,
    columns = frequencies.
    """
    rows = []

    for target_label, _, _ in targets:
        H = hrtf_cache[set_name][target_label][dataset_name][ear]
        rows.append(H)

    return np.vstack(rows)


def plot_hrtf_map(set_name, targets, dataset_name, ear, matrix):
    """
    Frequency-direction map of the HRTF magnitude.

    pcolormesh requires the X and Y axes to be edges if the matrix C has
    dimensions (N_y, N_x). Therefore:
    - f_edges must have length N_x + 1;
    - y_edges must have length N_y + 1.
    """
    axis_values, axis_label, tick_labels = get_direction_axis_for_target_set(set_name, targets)

    f = freq[freq_plot_mask]
    Z = matrix[:, freq_plot_mask]

    f_edges = centers_to_edges(f)
    y_edges = centers_to_edges(axis_values)

    plt.figure(figsize=(11, 5))

    mesh_kwargs = {}

    # Use a common scale for all absolute HRTF maps.
    hrtf_color_limits = HRTF_COLOR_LIMITS_DB
    if USE_COMMON_HRTF_COLOR_SCALE and COMMON_HRTF_COLOR_LIMITS_DB is not None:
        hrtf_color_limits = COMMON_HRTF_COLOR_LIMITS_DB

    if hrtf_color_limits is not None:
        mesh_kwargs["vmin"] = hrtf_color_limits[0]
        mesh_kwargs["vmax"] = hrtf_color_limits[1]

    mesh = plt.pcolormesh(
        f_edges,
        y_edges,
        Z,
        shading="auto",
        **mesh_kwargs,
    )

    plt.xscale("log")
    plt.xlabel("Frequenza [Hz]")
    plt.ylabel(axis_label)
    plt.title(f"HRTF magnitude map - {dataset_name} - {set_name} - {get_ear_name(ear)}")

    cbar = plt.colorbar(mesh)
    cbar.set_label("Magnitudine [dB]")

    if len(axis_values) <= 15:
        plt.yticks(axis_values, tick_labels)

    plt.grid(False)
    plt.tight_layout()

    maps_dir = output_root / set_name / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    filename = f"HRTF_map_{dataset_name}_{set_name}_{get_ear_name(ear)}.png"
    save_figure(maps_dir / filename)


def plot_difference_map(set_name, targets, pair_a, pair_b, ear, matrix_a, matrix_b):
    """
    Frequency-direction map of the spectral difference:
    H_pair_a - H_pair_b.

    The color is centered at 0 dB:
    - positive values: pair_a has greater magnitude than pair_b;
    - negative values: pair_a has smaller magnitude than pair_b.
    """
    axis_values, axis_label, tick_labels = get_direction_axis_for_target_set(set_name, targets)

    f = freq[freq_plot_mask]
    Z = (matrix_a - matrix_b)[:, freq_plot_mask]

    f_edges = centers_to_edges(f)
    y_edges = centers_to_edges(axis_values)
    norm = make_difference_norm(Z)

    plt.figure(figsize=(11, 5))

    mesh = plt.pcolormesh(
        f_edges,
        y_edges,
        Z,
        shading="auto",
        cmap="coolwarm",
        norm=norm,
    )

    plt.xscale("log")
    plt.xlabel("Frequenza [Hz]")
    plt.ylabel(axis_label)
    plt.title(
        f"Delta HRTF map: {pair_a} - {pair_b} - "
        f"{set_name} - {get_ear_name(ear)}"
    )

    cbar = plt.colorbar(mesh)
    cbar.set_label("Differenza [dB]")

    if len(axis_values) <= 15:
        plt.yticks(axis_values, tick_labels)

    plt.grid(False)
    plt.tight_layout()

    maps_dir = output_root / set_name / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    filename = f"Delta_HRTF_map_{pair_a}_minus_{pair_b}_{set_name}_{get_ear_name(ear)}.png"
    save_figure(maps_dir / filename)


def compute_common_map_color_scales():
    """
    Computes common color scales for all maps.

    - Absolute HRTF maps:
      common scale [vmin, vmax] in dB across all datasets, planes, ears and directions.
    - Difference maps:
      common symmetric scale [-vmax, +vmax] in dB across all pairs, planes and ears.
    """
    global COMMON_HRTF_COLOR_LIMITS_DB
    global COMMON_DIFFERENCE_COLOR_LIMIT_DB

    if USE_COMMON_HRTF_COLOR_SCALE and SAVE_HRTF_MAPS:
        if HRTF_COLOR_LIMITS_DB is not None:
            COMMON_HRTF_COLOR_LIMITS_DB = tuple(HRTF_COLOR_LIMITS_DB)
        else:
            hrtf_values = []

            for set_name, targets in target_sets.items():
                for ear in [0, 1]:
                    for dataset_name in sofas.keys():
                        matrix = build_hrtf_matrix(set_name, targets, dataset_name, ear)
                        Z = matrix[:, freq_plot_mask]
                        if Z.size > 0:
                            hrtf_values.append(Z.reshape(-1))

            if hrtf_values:
                all_hrtf_values = np.concatenate(hrtf_values)
                all_hrtf_values = all_hrtf_values[np.isfinite(all_hrtf_values)]

                if all_hrtf_values.size > 0:
                    vmin = float(np.nanpercentile(
                        all_hrtf_values,
                        COMMON_HRTF_COLOR_PERCENTILES[0],
                    ))
                    vmax = float(np.nanpercentile(
                        all_hrtf_values,
                        COMMON_HRTF_COLOR_PERCENTILES[1],
                    ))

                    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                        COMMON_HRTF_COLOR_LIMITS_DB = (vmin, vmax)

        if COMMON_HRTF_COLOR_LIMITS_DB is not None:
            print(
                "Scala comune mappe HRTF assolute: "
                f"{COMMON_HRTF_COLOR_LIMITS_DB[0]:.2f} to "
                f"{COMMON_HRTF_COLOR_LIMITS_DB[1]:.2f} dB"
            )

    if USE_COMMON_DIFFERENCE_COLOR_SCALE and SAVE_DIFFERENCE_MAPS:
        if DIFFERENCE_COLOR_LIMIT_DB is not None:
            COMMON_DIFFERENCE_COLOR_LIMIT_DB = float(DIFFERENCE_COLOR_LIMIT_DB)
        else:
            diff_values = []

            for set_name, targets in target_sets.items():
                for ear in [0, 1]:
                    matrices = {}
                    for dataset_name in sofas.keys():
                        matrices[dataset_name] = build_hrtf_matrix(
                            set_name,
                            targets,
                            dataset_name,
                            ear,
                        )

                    for pair_a, pair_b in comparison_pairs:
                        Z = (matrices[pair_a] - matrices[pair_b])[:, freq_plot_mask]
                        if Z.size > 0:
                            diff_values.append(np.abs(Z).reshape(-1))

            if diff_values:
                all_diff_values = np.concatenate(diff_values)
                all_diff_values = all_diff_values[np.isfinite(all_diff_values)]

                if all_diff_values.size > 0:
                    vmax_abs = float(np.nanpercentile(
                        all_diff_values,
                        COMMON_DIFFERENCE_COLOR_PERCENTILE,
                    ))

                    if not np.isfinite(vmax_abs) or vmax_abs <= 0.0:
                        vmax_abs = 1.0

                    COMMON_DIFFERENCE_COLOR_LIMIT_DB = vmax_abs

        if COMMON_DIFFERENCE_COLOR_LIMIT_DB is not None:
            print(
                "Scala comune mappe di differenza: "
                f"{-COMMON_DIFFERENCE_COLOR_LIMIT_DB:.2f} to "
                f"{COMMON_DIFFERENCE_COLOR_LIMIT_DB:.2f} dB"
            )


compute_common_map_color_scales()


if SAVE_HRTF_MAPS or SAVE_DIFFERENCE_MAPS:
    print("\nGenerazione mappe HRTF e mappe di differenza...")

    for set_name, targets in target_sets.items():
        for ear in [0, 1]:

            matrices = {}

            for dataset_name in sofas.keys():
                matrices[dataset_name] = build_hrtf_matrix(
                    set_name,
                    targets,
                    dataset_name,
                    ear,
                )

                if SAVE_HRTF_MAPS:
                    plot_hrtf_map(
                        set_name=set_name,
                        targets=targets,
                        dataset_name=dataset_name,
                        ear=ear,
                        matrix=matrices[dataset_name],
                    )

            if SAVE_DIFFERENCE_MAPS:
                for a, b in comparison_pairs:
                    plot_difference_map(
                        set_name=set_name,
                        targets=targets,
                        pair_a=a,
                        pair_b=b,
                        ear=ear,
                        matrix_a=matrices[a],
                        matrix_b=matrices[b],
                    )

# ============================================================
# PRELIMINARY DESCRIPTIVE METRICS ON THE SPECTRAL DIFFERENCES
# ============================================================

print("\nCalcolo metriche descrittive preliminari sulle differenze spettrali...")

spectral_rows = []

for set_name, targets in target_sets.items():
    for target_label, az_t, el_t in targets:
        for ear in [0, 1]:
            ear_name = get_ear_name(ear)

            for a, b in comparison_pairs:
                H_a = hrtf_cache[set_name][target_label][a][ear]
                H_b = hrtf_cache[set_name][target_label][b][ear]
                delta = H_a - H_b

                mask = metric_frequency_mask(freq)

                if np.any(mask):
                    delta_valid = delta[mask]
                    mean_abs_delta = float(np.mean(np.abs(delta_valid)))
                    median_abs_delta = float(np.median(np.abs(delta_valid)))
                    max_abs_delta = float(np.max(np.abs(delta_valid)))
                    rms_delta = float(np.sqrt(np.mean(delta_valid ** 2)))
                else:
                    mean_abs_delta = np.nan
                    median_abs_delta = np.nan
                    max_abs_delta = np.nan
                    rms_delta = np.nan

                row = {
                    "target_set": set_name,
                    "target_label": target_label,
                    "target_az_deg": az_t,
                    "target_el_deg": el_t,
                    "ear": ear_name,
                    "dataset_a": a,
                    "dataset_b": b,
                    "mean_abs_delta_db": mean_abs_delta,
                    "median_abs_delta_db": median_abs_delta,
                    "max_abs_delta_db": max_abs_delta,
                    "rms_delta_db": rms_delta,
                    "azimuth_convention": AZIMUTH_CONVENTION_NOTE,
                }

                for band_name, f_low, f_high in frequency_bands:
                    band_mask = (freq >= f_low) & (freq < f_high)

                    if np.any(band_mask):
                        row[f"mean_abs_delta_{band_name}_db"] = float(
                            np.mean(np.abs(delta[band_mask]))
                        )
                    else:
                        row[f"mean_abs_delta_{band_name}_db"] = np.nan

                spectral_rows.append(row)

spectral_csv = output_root / "spectral_difference_summary.csv"

with open(spectral_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(spectral_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(spectral_rows)

print(f"CSV differenze spettrali salvato in: {spectral_csv}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nAnalisi completata.")
print(f"Output salvati in: {output_root}")
print("\nFile principali generati:")
print(f"  - {selected_csv.name}")
if sanity_csv is not None:
    print(f"  - {sanity_csv.name}")
if gain_norm_csv is not None:
    print(f"  - {gain_norm_csv.name}")
print(f"  - {spectral_csv.name}")
print("\nNota:")
print("  Le metriche qui salvate sono differenze spettrali descrittive.")
print("  Non sostituiscono ITD, ILD e LSD calcolate con SAM.")
if APPLY_BROADBAND_GAIN_NORMALIZATION:
    print(
        "  Normalizzazione broadband attiva: "
        f"{GAIN_NORMALIZATION_TARGET_DATASETS} allineati a "
        f"{GAIN_NORMALIZATION_REFERENCE_DATASET} "
        f"nella banda {GAIN_NORMALIZATION_FREQ_LOW:.0f}-{GAIN_NORMALIZATION_FREQ_HIGH:.0f} Hz."
    )
if USE_COMMON_HRTF_COLOR_SCALE and COMMON_HRTF_COLOR_LIMITS_DB is not None:
    print(
        "  Scala comune HRTF maps: "
        f"{COMMON_HRTF_COLOR_LIMITS_DB[0]:.2f} to "
        f"{COMMON_HRTF_COLOR_LIMITS_DB[1]:.2f} dB."
    )
if USE_COMMON_DIFFERENCE_COLOR_SCALE and COMMON_DIFFERENCE_COLOR_LIMIT_DB is not None:
    print(
        "  Scala comune difference maps: "
        f"{-COMMON_DIFFERENCE_COLOR_LIMIT_DB:.2f} to "
        f"{COMMON_DIFFERENCE_COLOR_LIMIT_DB:.2f} dB."
    )
print(f"  Convenzione azimuth usata: {AZIMUTH_CONVENTION_NOTE}")
print(f"  Elevazioni verticali usate: {SONICOM_ELEVATIONS}")
