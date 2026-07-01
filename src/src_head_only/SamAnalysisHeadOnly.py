import os
import tempfile
import numpy as np
import pandas as pd
import sofar as sf
import matplotlib.pyplot as plt

from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf


# =========================================================
# PATHS OF THE THREE SOFA FILES TO COMPARE
# =========================================================
# Only change these three paths if you change subject/file.
# The three pairs are generated automatically below.

SOFA_PATHS = {
    "sim_nostra": "/home/capstone/Downloads/P067/SOFA_P067_Right_merged/HRIR_SONICOM_grid_48000.sofa",
    "sim_sonicom": "/home/capstone/Downloads/P067/HRIR_SONICOM_48000.sofa",
    "mis_sonicom": "/home/capstone/Downloads/P067/P0067_FreeFieldComp_48kHz.sofa",
}

COMPARISON_PAIRS = [
    ("sim_nostra", "mis_sonicom"),
    ("sim_sonicom", "mis_sonicom"),
    ("sim_nostra", "sim_sonicom"),
]

# Output folder
OUTPUT_DIR = "./comparison_output_three_pairs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# GLOBAL BROADBAND NORMALIZATION
# =========================================================
# This normalization is used to remove a global gain offset
# between the two files of the pair before computing the metrics, especially the LSD.
#
# Default:
# - for each pair, dataset_b is scaled to have the same average
#   broadband level as dataset_a in the GAIN_NORM_FREQ_LOW - GAIN_NORM_FREQ_HIGH band.
#
# Formula:
# offset_db = mean_directions_ears_freq( H_reference_db - H_target_db )
# target_hrir_normalized = target_hrir * 10^(offset_db / 20)
#
# Note:
# - ITD does not change significantly because it depends on the time delay.
# - ILD does not change if the same gain is applied to left and right.
# - LSD changes because the spectral comparison no longer includes the broadband offset.
APPLY_BROADBAND_GAIN_NORMALIZATION = True
GAIN_NORMALIZATION_MODE = "pairwise_b_to_a"
# Allowed values:
# - "pairwise_b_to_a": for each pair, normalize dataset_b relative to dataset_a
# - "none": no broadband normalization

GAIN_NORM_FREQ_LOW = 500.0
GAIN_NORM_FREQ_HIGH = 16000.0
GAIN_NORM_NFFT = 4096


# =========================================================
# PLOT OPTIONS
# =========================================================
MAKE_SCATTER_PLOTS = True
MAKE_HIST_PLOTS = True
TOP_N_DIRECTIONS = 10


# =========================================================
# SUPPORT FUNCTIONS
# =========================================================
def safe_name(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_sofa(path):
    return sf.read_sofa(path)


def same_sampling_rate(a, b):
    return np.allclose(
        np.atleast_1d(a.Data_SamplingRate),
        np.atleast_1d(b.Data_SamplingRate),
    )


def same_source_positions(a, b, atol=1e-6):
    if np.shape(a.SourcePosition) != np.shape(b.SourcePosition):
        return False
    return np.allclose(a.SourcePosition, b.SourcePosition, atol=atol)


def get_sampling_rate(sofa):
    return float(np.atleast_1d(sofa.Data_SamplingRate).squeeze())


def crop_sofa_to_length(sofa, new_len):
    sofa_new = sofa.copy()
    sofa_new.Data_IR = sofa_new.Data_IR[:, :, :new_len]
    return sofa_new


def apply_gain_db_to_sofa(sofa, gain_db):
    """
    Applies a linear gain to the HRIRs of the SOFA file.
    gain_db > 0 raises the level; gain_db < 0 lowers it.
    """
    sofa_new = sofa.copy()
    gain_linear = 10.0 ** (gain_db / 20.0)
    sofa_new.Data_IR = np.asarray(sofa_new.Data_IR, dtype=float) * gain_linear
    return sofa_new


def estimate_broadband_gain_offset_db(
    reference_sofa,
    target_sofa,
    fs,
    min_len,
    f_low=500.0,
    f_high=16000.0,
    nfft=4096,
):
    """
    Estimates the global broadband offset in dB needed to align
    target_sofa to reference_sofa.

    Returns:
    - offset_global_db: average offset to apply to the target;
    - offsets_per_location_ear: [M, R] matrix of local average offsets;
    - f_low_eff, f_high_eff: effective bounds of the band used.
    """
    ref_ir = np.asarray(reference_sofa.Data_IR[:, :, :min_len], dtype=float)
    tar_ir = np.asarray(target_sofa.Data_IR[:, :, :min_len], dtype=float)

    if ref_ir.shape[:2] != tar_ir.shape[:2]:
        raise ValueError(
            "Le due Data_IR non hanno stesso numero di direzioni/ricevitori "
            f"dopo il cropping: {ref_ir.shape} vs {tar_ir.shape}"
        )

    freq = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_high_eff = min(float(f_high), fs / 2.0)
    f_low_eff = float(f_low)
    mask = (freq >= f_low_eff) & (freq <= f_high_eff)

    if not np.any(mask):
        raise ValueError(
            f"La banda di normalizzazione {f_low_eff}-{f_high_eff} Hz "
            "non contiene bin FFT. Controlla fs/NFFT."
        )

    H_ref = np.fft.rfft(ref_ir, n=nfft, axis=2)
    H_tar = np.fft.rfft(tar_ir, n=nfft, axis=2)

    H_ref_db = 20.0 * np.log10(np.maximum(np.abs(H_ref), 1e-12))
    H_tar_db = 20.0 * np.log10(np.maximum(np.abs(H_tar), 1e-12))

    diff_db = H_ref_db[:, :, mask] - H_tar_db[:, :, mask]
    offsets_per_location_ear = np.mean(diff_db, axis=2)
    offset_global_db = float(np.mean(offsets_per_location_ear))

    return offset_global_db, offsets_per_location_ear, f_low_eff, f_high_eff


def save_gain_normalization_report(
    pair_output_dir,
    offsets_per_location_ear,
    positions,
    offset_global_db,
    gain_linear,
    reference_name,
    target_name,
    f_low,
    f_high,
):
    """
    Saves a diagnostic CSV with the local offset per direction and ear,
    plus the global offset applied.
    """
    rows = []
    positions = np.asarray(positions)

    for idx in range(offsets_per_location_ear.shape[0]):
        for ear in range(offsets_per_location_ear.shape[1]):
            rows.append({
                "index": idx,
                "azimuth_deg": positions[idx, 0],
                "elevation_deg": positions[idx, 1],
                "radius_m": positions[idx, 2] if positions.shape[1] > 2 else np.nan,
                "ear": "left" if ear == 0 else "right" if ear == 1 else f"ear_{ear}",
                "local_offset_db_reference_minus_target": offsets_per_location_ear[idx, ear],
                "global_offset_db_applied_to_target": offset_global_db,
                "gain_linear_applied_to_target": gain_linear,
                "reference_dataset": reference_name,
                "target_dataset": target_name,
                "normalization_freq_low_hz": f_low,
                "normalization_freq_high_hz": f_high,
            })

    report_path = os.path.join(pair_output_dir, "broadband_gain_normalization_offsets.csv")
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"Report normalizzazione broadband salvato in: {report_path}")
    return report_path


def save_temp_sofa(sofa, prefix):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".sofa")
    os.close(fd)
    sf.write_sofa(path, sofa)
    return path


def prepare_hrtf_objects(sofa1, sofa2):
    len1 = sofa1.Data_IR.shape[2]
    len2 = sofa2.Data_IR.shape[2]
    min_len = min(len1, len2)

    sofa1_c = crop_sofa_to_length(sofa1, min_len)
    sofa2_c = crop_sofa_to_length(sofa2, min_len)

    path1 = save_temp_sofa(sofa1_c, "tmp_hrtf1_")
    path2 = save_temp_sofa(sofa2_c, "tmp_hrtf2_")

    hrtf1 = ld.HRTF(path1)
    hrtf2 = ld.HRTF(path2)

    hrtf1, hrtf2 = ld.match_hrtf_locations(hrtf1, hrtf2)

    os.remove(path1)
    os.remove(path2)

    return hrtf1, hrtf2, min_len


def summarize_vector(x, name):
    x = np.asarray(x)
    return {
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_min": float(np.min(x)),
        f"{name}_max": float(np.max(x)),
    }


def top_locations(df, metric_col, top_n=10):
    return df.sort_values(metric_col, ascending=False).head(top_n)


def make_scatter_plot(df, value_col, title, output_path):
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df["azimuth_deg"], df["elevation_deg"], c=df[value_col], s=35)
    plt.colorbar(sc, label=value_col)
    plt.xlabel("Azimuth [deg]")
    plt.ylabel("Elevation [deg]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_hist_plot(df, value_col, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df[value_col], bins=30)
    plt.xlabel(value_col)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def check_pair_compatibility(name_a, sofa_a, name_b, sofa_b):
    """
    Checks minimal compatibility between two SOFA files.
    """
    print(f"\n=== Controllo compatibilità: {name_a} vs {name_b} ===")
    print(f"Shape Data_IR {name_a}:", sofa_a.Data_IR.shape)
    print(f"Shape Data_IR {name_b}:", sofa_b.Data_IR.shape)
    print(f"Shape SourcePosition {name_a}:", sofa_a.SourcePosition.shape)
    print(f"Shape SourcePosition {name_b}:", sofa_b.SourcePosition.shape)
    print("Stesso sample rate:", same_sampling_rate(sofa_a, sofa_b))
    print("Stesse SourcePosition:", same_source_positions(sofa_a, sofa_b))

    if not same_sampling_rate(sofa_a, sofa_b):
        raise ValueError(f"Sample rate diversi per {name_a} vs {name_b}.")
    if not same_source_positions(sofa_a, sofa_b):
        raise ValueError(f"SourcePosition diverse per {name_a} vs {name_b}.")


def compute_pair_metrics(name_a, name_b, sofa_a_raw, sofa_b_raw):
    """
    Computes ITD, ILD, and LSD for a single pair.
    Returns:
    - df: metrics for all directions;
    - summary: dictionary with global metrics;
    """
    pair_label = f"{name_a}_vs_{name_b}"
    pair_output_dir = os.path.join(OUTPUT_DIR, safe_name(pair_label))
    os.makedirs(pair_output_dir, exist_ok=True)

    check_pair_compatibility(name_a, sofa_a_raw, name_b, sofa_b_raw)

    fs = get_sampling_rate(sofa_a_raw)
    min_len_for_norm = min(sofa_a_raw.Data_IR.shape[2], sofa_b_raw.Data_IR.shape[2])

    sofa_a = sofa_a_raw.copy()
    sofa_b = sofa_b_raw.copy()

    gain_offset_db = 0.0
    gain_linear = 1.0
    normalization_report_path = None
    normalization_reference = "none"
    normalization_target = "none"

    # =====================================================
    # Broadband normalization per pair
    # =====================================================
    if APPLY_BROADBAND_GAIN_NORMALIZATION and GAIN_NORMALIZATION_MODE != "none":
        if GAIN_NORMALIZATION_MODE != "pairwise_b_to_a":
            raise ValueError(
                "GAIN_NORMALIZATION_MODE non valido. "
                "Valori ammessi: 'pairwise_b_to_a', 'none'."
            )

        reference_sofa = sofa_a
        target_sofa = sofa_b
        normalization_reference = name_a
        normalization_target = name_b

        gain_offset_db, offsets_per_location_ear, f_low_eff, f_high_eff = estimate_broadband_gain_offset_db(
            reference_sofa=reference_sofa,
            target_sofa=target_sofa,
            fs=fs,
            min_len=min_len_for_norm,
            f_low=GAIN_NORM_FREQ_LOW,
            f_high=GAIN_NORM_FREQ_HIGH,
            nfft=GAIN_NORM_NFFT,
        )

        gain_linear = 10.0 ** (gain_offset_db / 20.0)

        print("\n====================================")
        print(f"NORMALIZZAZIONE BROADBAND: {name_a} vs {name_b}")
        print("====================================")
        print(f"Reference dataset: {normalization_reference}")
        print(f"Target dataset:    {normalization_target}")
        print(f"Banda usata:       {f_low_eff:.1f} - {f_high_eff:.1f} Hz")
        print(f"Offset stimato:    {gain_offset_db:.3f} dB")
        print(f"Gain lineare:      {gain_linear:.6f}")
        print(f"Azione: {name_b} viene scalato di {gain_offset_db:.3f} dB")

        sofa_b = apply_gain_db_to_sofa(sofa_b, gain_offset_db)

        normalization_report_path = save_gain_normalization_report(
            pair_output_dir=pair_output_dir,
            offsets_per_location_ear=offsets_per_location_ear,
            positions=np.asarray(reference_sofa.SourcePosition),
            offset_global_db=gain_offset_db,
            gain_linear=gain_linear,
            reference_name=normalization_reference,
            target_name=normalization_target,
            f_low=f_low_eff,
            f_high=f_high_eff,
        )
    else:
        print(f"\nNormalizzazione broadband disattivata per {name_a} vs {name_b}.")

    hrtf_a, hrtf_b, min_len = prepare_hrtf_objects(sofa_a, sofa_b)

    print(f"\nLunghezza minima comune usata per {pair_label}: {min_len}")

    # =====================================================
    # ITD and ILD per individual dataset
    # =====================================================
    itd_a_s, itd_a_samps, _ = hf.itd_estimator_maxiacce(hrtf_a.hrir, hrtf_a.fs)
    itd_b_s, itd_b_samps, _ = hf.itd_estimator_maxiacce(hrtf_b.hrir, hrtf_b.fs)

    ild_a = hf.ild_estimator_rms(hrtf_a.hrir)
    ild_b = hf.ild_estimator_rms(hrtf_b.hrir)

    # Absolute differences per position
    itd_diff_per_loc = np.abs(itd_a_s - itd_b_s)
    ild_diff_per_loc = np.abs(ild_a - ild_b)

    # =====================================================
    # LSD per position
    # =====================================================
    lsd_mean, lsd_mat = hf.calculate_lsd_across_locations(hrtf_a.hrir, hrtf_b.hrir, hrtf_a.fs)
    lsd_per_loc = np.mean(lsd_mat, axis=1)

    print("\n====================================")
    print(f"RIASSUNTO GLOBALE: {name_a} vs {name_b}")
    print("====================================")
    print("ITD difference media [s]:", np.mean(itd_diff_per_loc))
    print("ITD difference media [us]:", np.mean(itd_diff_per_loc) * 1e6)
    print("ILD difference media [dB]:", np.mean(ild_diff_per_loc))
    print("LSD media [dB]:", lsd_mean)

    positions = np.asarray(sofa_a.SourcePosition)

    df = pd.DataFrame({
        "comparison_pair": pair_label,
        "dataset_a": name_a,
        "dataset_b": name_b,
        "index": np.arange(len(positions)),
        "azimuth_deg": positions[:, 0],
        "elevation_deg": positions[:, 1],
        "radius_m": positions[:, 2] if positions.shape[1] > 2 else np.nan,
        "itd_a_s": itd_a_s,
        "itd_b_s": itd_b_s,
        "itd_a_us": itd_a_s * 1e6,
        "itd_b_us": itd_b_s * 1e6,
        "ild_a_db": ild_a,
        "ild_b_db": ild_b,
        "ITD_diff_s": itd_diff_per_loc,
        "ITD_diff_us": itd_diff_per_loc * 1e6,
        "ILD_diff_db": ild_diff_per_loc,
        "LSD_db": lsd_per_loc,
        "broadband_normalization_applied": APPLY_BROADBAND_GAIN_NORMALIZATION and GAIN_NORMALIZATION_MODE != "none",
        "normalization_mode": GAIN_NORMALIZATION_MODE if APPLY_BROADBAND_GAIN_NORMALIZATION else "none",
        "normalization_reference": normalization_reference,
        "normalization_target": normalization_target,
        "normalization_offset_db_applied_to_target": gain_offset_db,
        "normalization_gain_linear_applied_to_target": gain_linear,
        "normalization_freq_low_hz": GAIN_NORM_FREQ_LOW if APPLY_BROADBAND_GAIN_NORMALIZATION else np.nan,
        "normalization_freq_high_hz": GAIN_NORM_FREQ_HIGH if APPLY_BROADBAND_GAIN_NORMALIZATION else np.nan,
    })

    # Save full table for the pair
    full_csv = os.path.join(pair_output_dir, "all_directions_metrics.csv")
    df.to_csv(full_csv, index=False)
    print(f"\nTabella completa della coppia salvata in: {full_csv}")

    # Top N per metric
    top_itd = top_locations(df, "ITD_diff_us", top_n=TOP_N_DIRECTIONS)
    top_ild = top_locations(df, "ILD_diff_db", top_n=TOP_N_DIRECTIONS)
    top_lsd = top_locations(df, "LSD_db", top_n=TOP_N_DIRECTIONS)

    top_itd_csv = os.path.join(pair_output_dir, "top10_itd_diff.csv")
    top_ild_csv = os.path.join(pair_output_dir, "top10_ild_diff.csv")
    top_lsd_csv = os.path.join(pair_output_dir, "top10_lsd.csv")

    top_itd.to_csv(top_itd_csv, index=False)
    top_ild.to_csv(top_ild_csv, index=False)
    top_lsd.to_csv(top_lsd_csv, index=False)

    print(f"Top {TOP_N_DIRECTIONS} ITD salvata in: {top_itd_csv}")
    print(f"Top {TOP_N_DIRECTIONS} ILD salvata in: {top_ild_csv}")
    print(f"Top {TOP_N_DIRECTIONS} LSD salvata in: {top_lsd_csv}")

    print("\n==============================")
    print(f"TOP {TOP_N_DIRECTIONS} DIREZIONI PEGGIORI: {name_a} vs {name_b}")
    print("==============================")

    print("\nTop ITD difference [us]:")
    print(top_itd[["index", "azimuth_deg", "elevation_deg", "radius_m", "ITD_diff_us"]].to_string(index=False))

    print("\nTop ILD difference [dB]:")
    print(top_ild[["index", "azimuth_deg", "elevation_deg", "radius_m", "ILD_diff_db"]].to_string(index=False))

    print("\nTop LSD [dB]:")
    print(top_lsd[["index", "azimuth_deg", "elevation_deg", "radius_m", "LSD_db"]].to_string(index=False))

    # =====================================================
    # Global plots over all directions per pair
    # =====================================================
    normalization_suffix = " con normalizzazione broadband" if (
        APPLY_BROADBAND_GAIN_NORMALIZATION and GAIN_NORMALIZATION_MODE != "none"
    ) else ""

    if MAKE_SCATTER_PLOTS:
        make_scatter_plot(
            df,
            "ITD_diff_us",
            f"ITD difference [us] - {name_a} vs {name_b}" + normalization_suffix,
            os.path.join(pair_output_dir, "scatter_itd_diff_us.png"),
        )

        make_scatter_plot(
            df,
            "ILD_diff_db",
            f"ILD difference [dB] - {name_a} vs {name_b}" + normalization_suffix,
            os.path.join(pair_output_dir, "scatter_ild_diff_db.png"),
        )

        make_scatter_plot(
            df,
            "LSD_db",
            f"LSD [dB] - {name_a} vs {name_b}" + normalization_suffix,
            os.path.join(pair_output_dir, "scatter_lsd_db.png"),
        )

    if MAKE_HIST_PLOTS:
        make_hist_plot(
            df,
            "ITD_diff_us",
            f"Distribuzione ITD difference [us] - {name_a} vs {name_b}" + normalization_suffix,
            os.path.join(pair_output_dir, "hist_itd_diff_us.png"),
        )

        make_hist_plot(
            df,
            "ILD_diff_db",
            f"Distribuzione ILD difference [dB] - {name_a} vs {name_b}" + normalization_suffix,
            os.path.join(pair_output_dir, "hist_ild_diff_db.png"),
        )

        make_hist_plot(
            df,
            "LSD_db",
            f"Distribuzione LSD [dB] - {name_a} vs {name_b}" + normalization_suffix,
            os.path.join(pair_output_dir, "hist_lsd_db.png"),
        )

    summary = {
        "comparison_pair": pair_label,
        "dataset_a": name_a,
        "dataset_b": name_b,
        "n_directions": len(df),
        "ITD_diff_us_mean": float(np.mean(itd_diff_per_loc) * 1e6),
        "ITD_diff_us_std": float(np.std(itd_diff_per_loc) * 1e6),
        "ITD_diff_us_min": float(np.min(itd_diff_per_loc) * 1e6),
        "ITD_diff_us_max": float(np.max(itd_diff_per_loc) * 1e6),
        "ILD_diff_db_mean": float(np.mean(ild_diff_per_loc)),
        "ILD_diff_db_std": float(np.std(ild_diff_per_loc)),
        "ILD_diff_db_min": float(np.min(ild_diff_per_loc)),
        "ILD_diff_db_max": float(np.max(ild_diff_per_loc)),
        "LSD_db_mean": float(lsd_mean),
        "LSD_db_std": float(np.std(lsd_per_loc)),
        "LSD_db_min": float(np.min(lsd_per_loc)),
        "LSD_db_max": float(np.max(lsd_per_loc)),
        "normalization_applied": APPLY_BROADBAND_GAIN_NORMALIZATION and GAIN_NORMALIZATION_MODE != "none",
        "normalization_mode": GAIN_NORMALIZATION_MODE if APPLY_BROADBAND_GAIN_NORMALIZATION else "none",
        "normalization_reference": normalization_reference,
        "normalization_target": normalization_target,
        "normalization_offset_db_applied_to_target": gain_offset_db,
        "normalization_gain_linear_applied_to_target": gain_linear,
        "normalization_report_path": normalization_report_path,
        "pair_output_dir": pair_output_dir,
    }

    return df, summary


# =========================================================
# MAIN
# =========================================================
print("\n====================================")
print("LETTURA DEI TRE FILE SOFA")
print("====================================")

sofas = {}
for dataset_name, path in SOFA_PATHS.items():
    print(f"{dataset_name}: {path}")
    sofas[dataset_name] = load_sofa(path)

# Check that all pairs reference existing datasets
for name_a, name_b in COMPARISON_PAIRS:
    if name_a not in sofas:
        raise KeyError(f"Dataset non trovato in SOFA_PATHS: {name_a}")
    if name_b not in sofas:
        raise KeyError(f"Dataset non trovato in SOFA_PATHS: {name_b}")

all_pair_dfs = []
summary_rows = []

for name_a, name_b in COMPARISON_PAIRS:
    df_pair, summary_pair = compute_pair_metrics(
        name_a=name_a,
        name_b=name_b,
        sofa_a_raw=sofas[name_a],
        sofa_b_raw=sofas[name_b],
    )
    all_pair_dfs.append(df_pair)
    summary_rows.append(summary_pair)

# =========================================================
# GLOBAL OUTPUTS OVER ALL PAIRS
# =========================================================
all_pairs_df = pd.concat(all_pair_dfs, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

all_pairs_csv = os.path.join(OUTPUT_DIR, "all_directions_metrics_all_pairs.csv")
summary_csv = os.path.join(OUTPUT_DIR, "summary_metrics_by_pair.csv")

all_pairs_df.to_csv(all_pairs_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\n====================================")
print("RIEPILOGO FINALE - TUTTE LE COPPIE")
print("====================================")
print(summary_df[[
    "comparison_pair",
    "ITD_diff_us_mean",
    "ILD_diff_db_mean",
    "LSD_db_mean",
    "normalization_offset_db_applied_to_target",
]].to_string(index=False))

print(f"\nCSV globale tutte le direzioni: {all_pairs_csv}")
print(f"CSV summary per coppia: {summary_csv}")
print(f"Output salvati nella cartella: {OUTPUT_DIR}")
