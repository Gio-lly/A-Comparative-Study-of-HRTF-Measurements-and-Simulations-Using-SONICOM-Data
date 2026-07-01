import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sofar as sf
import matplotlib.pyplot as plt

from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf


# =========================================================
# SAM VALIDATION 100 Hz - 10 kHz: HEAD-ONLY, HEAD+TORSO AND SIM SONICOM AGAINST THE SONICOM MEASUREMENT
# =========================================================
# Comparisons:
#   1) sim_head_only067  vs mis_sonicom067
#   2) sim_head_torso067 vs mis_sonicom067
#   3) sim_sonicom067    vs mis_sonicom067
#
# SAM metrics:
#   - ITD difference
#   - ILD difference
#   - LSD
#
# Normalization:
#   The measured data is normalized broadband with respect to the neutral reference:
#       mean(sim_head_only067, sim_head_torso067)
#   This avoids directly favoring one of the two simulations.
#
# Note:
#   - ITD is essentially independent of the global gain.
#   - ILD is essentially unchanged if the same gain is applied to left/right.
#   - LSD is affected by the global gain, so the normalization mainly serves LSD.
# =========================================================


# =========================================================
# PATH CONFIGURATION
# =========================================================

def resolve_first_existing(label, candidates):
    for candidate in candidates:
        p = Path(candidate).expanduser()
        if p.exists():
            print(f"{label}: uso file trovato -> {p}")
            return str(p)

    print(f"\nATTENZIONE: nessun file trovato per {label}.")
    print("Candidati controllati:")
    for candidate in candidates:
        print(f"  - {candidate}")
    print("\nModifica il path nella sezione CONFIGURAZIONE PATH prima di eseguire lo script.")
    return str(Path(candidates[0]).expanduser())


SOFA_PATHS = {
    "sim_head_only067": resolve_first_existing(
        "sim_head_only067",
        [
            "/home/capstone/Downloads/P067/SOFA_P067_Right_merged/HRIR_SONICOM_grid_48000.sofa",
        ],
    ),
    "sim_head_torso067": resolve_first_existing(
        "sim_head_torso067",
        [
            "/home/capstone/Downloads/P067/P067_torso_HRIR_SONICOM_grid_48000.sofa",
        ],
    ),
    "sim_sonicom067": resolve_first_existing(
        "sim_sonicom067",
        [
            "/home/capstone/Downloads/P067/HRIR_SONICOM_48000.sofa",
        ],
    ),
    "mis_sonicom067": resolve_first_existing(
        "mis_sonicom067",
        [
            "/home/capstone/Downloads/P067/P0067_FreeFieldComp_48kHz.sofa",
        ],
    ),
}

COMPARISON_PAIRS = [
    ("sim_head_only067", "mis_sonicom067", "head_only_vs_measured"),
    ("sim_head_torso067", "mis_sonicom067", "head_torso_vs_measured"),
    ("sim_sonicom067", "mis_sonicom067", "sim_sonicom_vs_measured"),
]

PAIR_LABELS = {
    "head_only_vs_measured": "Head-only P067 vs measured P067",
    "head_torso_vs_measured": "Head+torso P067 vs measured P067",
    "sim_sonicom_vs_measured": "SONICOM simulated P067 vs measured P067",
}

# SONICOM/SOFA angular convention:
# azimuth 0°    = frontal
# azimuth +90°  = listener's left side
# azimuth -90°  = listener's right side
# azimuth ±180° = rear
AZIMUTH_CONVENTION_NOTE = "SOFA/SONICOM: +90 deg = left, -90 deg = right"


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = "./sam_head_vs_torso_sonicom_validation_P067_100_10k_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# BROADBAND NORMALIZATION OF THE MEASURED DATA
# =========================================================

APPLY_BROADBAND_GAIN_NORMALIZATION = True
GAIN_NORMALIZATION_TARGET_DATASET = "mis_sonicom067"
GAIN_NORMALIZATION_REFERENCE_DATASETS = ["sim_head_only067", "sim_head_torso067"]
GAIN_NORM_FREQ_LOW = 500.0
GAIN_NORM_FREQ_HIGH = 6000.0
GAIN_NORM_NFFT = 4096
GAIN_NORMALIZATION_METHOD = "mis_sonicom067 aligned to mean(sim_head_only067, sim_head_torso067) over 500-6000 Hz; sim_sonicom067 is evaluated as an external baseline"

# =========================================================
# BAND-LIMITED SAM ANALYSIS 100 Hz - 10 kHz
# =========================================================
# To make the comparison consistent with the HRTF analysis up to 10 kHz,
# the HRIRs passed to SAM are ideally filtered in frequency:
# all components above SAM_ANALYSIS_FREQ_HIGH are zeroed out.
# This way ITD, ILD and LSD are computed on band-limited HRIR.
# Note: the band-limited ITD can change slightly compared to the full-band ITD.
APPLY_SAM_BANDLIMIT = True
SAM_ANALYSIS_FREQ_LOW = 100.0
SAM_ANALYSIS_FREQ_HIGH = 10000.0
SAM_ANALYSIS_BAND_NOTE = "SAM metrics computed on HRIR band-limited from 100 Hz to 10 kHz"
LSD_NFFT = 4096
LSD_EPS = 1e-12

# Important note:
# The SAM function calculate_lsd_across_locations does not allow explicitly
# passing a frequency mask here. If a band-limited HRIR is used and then the
# full-band LSD is computed, near-zero bins appear above 10 kHz that can
# produce inf. For the 100 Hz-10 kHz LSD we therefore use the same log-spectral
# principle, but with an explicit 100 Hz-10 kHz mask and a numerical floor.


# =========================================================
# PLOT
# =========================================================

MAKE_SCATTER_PLOTS = True
MAKE_HISTOGRAMS = True
SCATTER_SIZE = 35
PLOT_DPI = 200


# =========================================================
# SOFA UTILITIES
# =========================================================

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


def bandlimit_sofa_hrir(sofa, f_high, f_low=0.0):
    """
    Returns a copy of the SOFA with HRIR band-limited via FFT.

    Components below f_low and above f_high are zeroed out.
    For our case f_low=100 Hz and f_high=10000 Hz.

    This operation is only performed on the copies passed to SAM;
    the original files are not modified.
    """
    sofa_new = sofa.copy()
    fs_local = get_sampling_rate(sofa_new)
    ir = np.asarray(sofa_new.Data_IR, dtype=float)
    n = ir.shape[2]

    freq_local = np.fft.rfftfreq(n, 1.0 / fs_local)
    H = np.fft.rfft(ir, n=n, axis=2)

    mask_keep = (freq_local >= float(f_low)) & (freq_local <= min(float(f_high), fs_local / 2.0))
    H[:, :, ~mask_keep] = 0.0

    ir_bandlimited = np.fft.irfft(H, n=n, axis=2)
    sofa_new.Data_IR = ir_bandlimited
    return sofa_new


def save_temp_sofa(sofa, prefix):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".sofa")
    os.close(fd)
    sf.write_sofa(path, sofa)
    return path


def prepare_hrtf_objects(sofa_a, sofa_b):
    """
    Creates temporary SAM HRTF objects, cropping the IRs to the common minimum length.
    Then uses SAM's match_hrtf_locations.
    """
    len_a = sofa_a.Data_IR.shape[2]
    len_b = sofa_b.Data_IR.shape[2]
    min_len = min(len_a, len_b)

    sofa_a_c = crop_sofa_to_length(sofa_a, min_len)
    sofa_b_c = crop_sofa_to_length(sofa_b, min_len)

    path_a = save_temp_sofa(sofa_a_c, "tmp_sam_hrtf_a_")
    path_b = save_temp_sofa(sofa_b_c, "tmp_sam_hrtf_b_")

    try:
        hrtf_a = ld.HRTF(path_a)
        hrtf_b = ld.HRTF(path_b)
        hrtf_a, hrtf_b = ld.match_hrtf_locations(hrtf_a, hrtf_b)
    finally:
        if os.path.exists(path_a):
            os.remove(path_a)
        if os.path.exists(path_b):
            os.remove(path_b)

    return hrtf_a, hrtf_b, min_len


def calculate_lsd_across_locations_limited_band(hrir_a, hrir_b, fs, f_low, f_high, nfft=LSD_NFFT, eps=LSD_EPS):
    """
    Computes an LSD per direction limited to a frequency band.

    Output:
    - lsd_mean: global average over directions;
    - lsd_per_loc: [M] vector with LSD averaged over receivers for each direction;
    - lsd_per_loc_ear: [M, R] matrix with LSD per direction and receiver.

    Formula for each direction/ear:
        LSD = sqrt(mean_f((20log10(|H_a|) - 20log10(|H_b|))^2))

    We use an eps floor to avoid log10(0) and an explicit frequency mask,
    so the band above 10 kHz does not enter the LSD.
    """
    a = np.asarray(hrir_a, dtype=float)
    b = np.asarray(hrir_b, dtype=float)

    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"Shape HRIR incompatibili per LSD: {a.shape} vs {b.shape}")

    min_len = min(a.shape[2], b.shape[2])
    a = a[:, :, :min_len]
    b = b[:, :, :min_len]

    freq = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_high_eff = min(float(f_high), fs / 2.0)
    f_low_eff = float(f_low)
    mask = (freq >= f_low_eff) & (freq <= f_high_eff)
    if not np.any(mask):
        raise ValueError(f"La banda LSD {f_low_eff}-{f_high_eff} Hz non contiene bin FFT validi.")

    H_a = np.fft.rfft(a, n=nfft, axis=2)
    H_b = np.fft.rfft(b, n=nfft, axis=2)

    A_db = 20.0 * np.log10(np.maximum(np.abs(H_a), eps))
    B_db = 20.0 * np.log10(np.maximum(np.abs(H_b), eps))
    diff = A_db[:, :, mask] - B_db[:, :, mask]

    lsd_per_loc_ear = np.sqrt(np.mean(diff ** 2, axis=2))
    lsd_per_loc = np.mean(lsd_per_loc_ear, axis=1)
    lsd_mean = float(np.mean(lsd_per_loc))

    return lsd_mean, lsd_per_loc, lsd_per_loc_ear


def finite_values(series):
    """Returns only finite values from a numeric series/array."""
    x = np.asarray(series, dtype=float)
    return x[np.isfinite(x)]


# =========================================================
# BROADBAND NORMALIZATION: MEASURED VS MEAN OF THE TWO SIMULATIONS
# =========================================================

def estimate_measured_offset_to_mean_reference_db(
    sofas,
    fs,
    min_len,
    target_dataset,
    reference_datasets,
    f_low,
    f_high,
    nfft,
):
    """
    Estimates a global offset in dB to apply to the measured data.

    Neutral reference:
        H_ref_db = mean_dB(sim_head_only067, sim_head_torso067)

    Offset:
        offset_db = mean(H_ref_db - H_target_db)

    The average is taken over directions, ears and the f_low-f_high band.
    """
    target_sofa = sofas[target_dataset]
    reference_sofas = [sofas[name] for name in reference_datasets]

    target_ir = np.asarray(target_sofa.Data_IR[:, :, :min_len], dtype=float)
    reference_irs = [np.asarray(s.Data_IR[:, :, :min_len], dtype=float) for s in reference_sofas]

    for ref_ir, ref_name in zip(reference_irs, reference_datasets):
        if ref_ir.shape[:2] != target_ir.shape[:2]:
            raise ValueError(
                f"Shape direzioni/orecchie incompatibile tra {ref_name} e {target_dataset}: "
                f"{ref_ir.shape} vs {target_ir.shape}"
            )

    freq = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_high_eff = min(float(f_high), fs / 2.0)
    f_low_eff = float(f_low)
    mask = (freq >= f_low_eff) & (freq <= f_high_eff)

    if not np.any(mask):
        raise ValueError(
            f"La banda di normalizzazione {f_low_eff}-{f_high_eff} Hz non contiene bin FFT."
        )

    H_target = np.fft.rfft(target_ir, n=nfft, axis=2)
    H_target_db = 20.0 * np.log10(np.maximum(np.abs(H_target), 1e-12))

    H_refs_db = []
    for ref_ir in reference_irs:
        H_ref = np.fft.rfft(ref_ir, n=nfft, axis=2)
        H_refs_db.append(20.0 * np.log10(np.maximum(np.abs(H_ref), 1e-12)))

    H_ref_mean_db = np.mean(np.stack(H_refs_db, axis=0), axis=0)

    diff_db = H_ref_mean_db[:, :, mask] - H_target_db[:, :, mask]
    offsets_per_location_ear = np.mean(diff_db, axis=2)
    offset_global_db = float(np.mean(offsets_per_location_ear))

    return offset_global_db, offsets_per_location_ear, f_low_eff, f_high_eff


def save_gain_normalization_report(
    offsets_per_location_ear,
    positions,
    offset_global_db,
    gain_linear,
    reference_datasets,
    target_dataset,
    f_low,
    f_high,
):
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
                "local_offset_db_reference_mean_minus_target": offsets_per_location_ear[idx, ear],
                "global_offset_db_applied_to_target": offset_global_db,
                "gain_linear_applied_to_target": gain_linear,
                "reference_datasets": "+".join(reference_datasets),
                "target_dataset": target_dataset,
                "normalization_method": GAIN_NORMALIZATION_METHOD,
                "normalization_freq_low_hz": f_low,
                "normalization_freq_high_hz": f_high,
            })

    report_path = os.path.join(OUTPUT_DIR, "broadband_gain_normalization_mean_sim_reference.csv")
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"Report normalizzazione broadband salvato in: {report_path}")
    return report_path


# =========================================================
# ANGULAR GROUPS
# =========================================================

def normalize_azimuth_deg(az):
    return ((az + 180.0) % 360.0) - 180.0


def angle_close(values, target, tol=1.0):
    values = np.asarray(values, dtype=float)
    target = normalize_azimuth_deg(target)
    diff = normalize_azimuth_deg(values - target)
    return np.abs(diff) <= tol


def build_group_masks(positions, tol=1.0):
    positions = np.asarray(positions)
    az = normalize_azimuth_deg(positions[:, 0])
    el = positions[:, 1]

    horizontal = np.abs(el - 0.0) <= tol
    median_front = angle_close(az, 0.0, tol=tol)
    median_back = angle_close(az, 180.0, tol=tol) | angle_close(az, -180.0, tol=tol)
    lateral_left = angle_close(az, 90.0, tol=tol)
    lateral_right = angle_close(az, -90.0, tol=tol)

    masks = {
        "global_all": np.ones(len(positions), dtype=bool),
        "horizontal_plane": horizontal,
        "median_plane": median_front | median_back,
        "median_front": median_front,
        "median_back": median_back,
        "lateral_plane": lateral_left | lateral_right,
        "lateral_left": lateral_left,
        "lateral_right": lateral_right,
    }
    return masks


# =========================================================
# METRICS
# =========================================================

def compute_sam_metrics_for_pair(
    pair_label,
    sofa_a,
    sofa_b,
    positions,
    normalization_info,
    sofa_a_for_itd=None,
    sofa_b_for_itd=None,
):
    """
    Computes the metrics for a pair of datasets.

    Important methodological choice:
    - ITD is computed on the copies that are NOT gain-normalized, so the
      temporal estimate cannot be influenced by the broadband normalization.
    - ILD and LSD are computed on the possibly normalized copies, because the
      normalization was introduced to make level/spectrum-dependent metrics
      consistent. A global normalization common to left/right should not
      change the internal ILD anyway.
    """
    # Datasets used for ILD and LSD: the ones passed to the function, so already
    # normalized if the target is the measured data.
    if APPLY_SAM_BANDLIMIT:
        sofa_a_for_sam = bandlimit_sofa_hrir(sofa_a, SAM_ANALYSIS_FREQ_HIGH, SAM_ANALYSIS_FREQ_LOW)
        sofa_b_for_sam = bandlimit_sofa_hrir(sofa_b, SAM_ANALYSIS_FREQ_HIGH, SAM_ANALYSIS_FREQ_LOW)
    else:
        sofa_a_for_sam = sofa_a
        sofa_b_for_sam = sofa_b

    hrtf_a, hrtf_b, min_len = prepare_hrtf_objects(sofa_a_for_sam, sofa_b_for_sam)

    print(f"\nCalcolo metriche SAM per {pair_label}...")
    print(f"Lunghezza minima comune usata per ILD/LSD: {min_len}")
    if APPLY_SAM_BANDLIMIT:
        print(f"Band-limit SAM: {SAM_ANALYSIS_FREQ_LOW:.1f} - {SAM_ANALYSIS_FREQ_HIGH:.1f} Hz")

    # Datasets used for ITD: raw/non-normalized copies.
    # If not passed, fall back to the standard metric datasets.
    if sofa_a_for_itd is None:
        sofa_a_for_itd = sofa_a
    if sofa_b_for_itd is None:
        sofa_b_for_itd = sofa_b

    if APPLY_SAM_BANDLIMIT:
        sofa_a_itd_band = bandlimit_sofa_hrir(sofa_a_for_itd, SAM_ANALYSIS_FREQ_HIGH, SAM_ANALYSIS_FREQ_LOW)
        sofa_b_itd_band = bandlimit_sofa_hrir(sofa_b_for_itd, SAM_ANALYSIS_FREQ_HIGH, SAM_ANALYSIS_FREQ_LOW)
    else:
        sofa_a_itd_band = sofa_a_for_itd
        sofa_b_itd_band = sofa_b_for_itd

    hrtf_a_itd, hrtf_b_itd, min_len_itd = prepare_hrtf_objects(sofa_a_itd_band, sofa_b_itd_band)
    print(f"Lunghezza minima comune usata per ITD non normalizzata: {min_len_itd}")

    # ITD per single dataset: computed before the gain normalization.
    itd_a_s, itd_a_samps, _ = hf.itd_estimator_maxiacce(hrtf_a_itd.hrir, hrtf_a_itd.fs)
    itd_b_s, itd_b_samps, _ = hf.itd_estimator_maxiacce(hrtf_b_itd.hrir, hrtf_b_itd.fs)

    # ILD per single dataset: computed on the datasets also used for LSD.
    ild_a = hf.ild_estimator_rms(hrtf_a.hrir)
    ild_b = hf.ild_estimator_rms(hrtf_b.hrir)

    itd_diff_s = np.abs(itd_a_s - itd_b_s)
    itd_diff_us = itd_diff_s * 1e6
    ild_diff = np.abs(ild_a - ild_b)

    # LSD per position, explicitly limited to the SAM_ANALYSIS_FREQ_LOW-SAM_ANALYSIS_FREQ_HIGH band.
    # We do not use hf.calculate_lsd_across_locations on the band-limited HRIR, because the function
    # would compute a full-band LSD and the zeroed bins above 10 kHz could produce inf.
    lsd_mean, lsd_per_loc, lsd_per_loc_ear = calculate_lsd_across_locations_limited_band(
        hrtf_a.hrir,
        hrtf_b.hrir,
        hrtf_a.fs,
        SAM_ANALYSIS_FREQ_LOW if APPLY_SAM_BANDLIMIT else 0.0,
        SAM_ANALYSIS_FREQ_HIGH if APPLY_SAM_BANDLIMIT else hrtf_a.fs / 2.0,
        nfft=LSD_NFFT,
        eps=LSD_EPS,
    )

    df = pd.DataFrame({
        "pair_label": pair_label,
        "pair_description": PAIR_LABELS.get(pair_label, pair_label),
        "index": np.arange(len(positions)),
        "azimuth_deg": positions[:, 0],
        "azimuth_norm_deg": normalize_azimuth_deg(positions[:, 0]),
        "elevation_deg": positions[:, 1],
        "radius_m": positions[:, 2] if positions.shape[1] > 2 else np.nan,
        "itd_dataset_a_s": itd_a_s,
        "itd_dataset_b_s": itd_b_s,
        "itd_dataset_a_us": itd_a_s * 1e6,
        "itd_dataset_b_us": itd_b_s * 1e6,
        "ITD_diff_s": itd_diff_s,
        "ITD_diff_us": itd_diff_us,
        "itd_computed_before_gain_normalization": True,
        "itd_gain_normalization_note": "ITD estimated on raw/non-gain-normalized HRIRs, then band-limited if enabled",
        "ild_dataset_a_db": ild_a,
        "ild_dataset_b_db": ild_b,
        "ILD_diff_db": ild_diff,
        "LSD_db": lsd_per_loc,
        "LSD_mean_from_sam_db": lsd_mean,
        "normalization_applied": normalization_info["applied"],
        "normalization_method": normalization_info["method"],
        "measured_offset_db_applied": normalization_info["offset_db"],
        "normalization_freq_low_hz": normalization_info["f_low"],
        "normalization_freq_high_hz": normalization_info["f_high"],
        "sam_bandlimit_applied": APPLY_SAM_BANDLIMIT,
        "sam_analysis_freq_low_hz": SAM_ANALYSIS_FREQ_LOW if APPLY_SAM_BANDLIMIT else np.nan,
        "sam_analysis_freq_high_hz": SAM_ANALYSIS_FREQ_HIGH if APPLY_SAM_BANDLIMIT else np.nan,
        "sam_analysis_band_note": SAM_ANALYSIS_BAND_NOTE if APPLY_SAM_BANDLIMIT else "full-band/raw HRIR",
        "azimuth_convention": AZIMUTH_CONVENTION_NOTE,
    })

    return df


def summarize_metric(series):
    x = finite_values(series)
    if x.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p95": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p95": float(np.percentile(x, 95)),
    }


def build_summary(all_metrics_df, group_masks):
    rows = []
    metric_cols = [
        ("ITD_diff_us", "us"),
        ("ILD_diff_db", "dB"),
        ("LSD_db", "dB"),
    ]

    for pair_label in all_metrics_df["pair_label"].unique():
        df_pair = all_metrics_df[all_metrics_df["pair_label"] == pair_label].copy()

        for group_name, mask in group_masks.items():
            df_group = df_pair[mask]
            if len(df_group) == 0:
                continue

            for metric_col, unit in metric_cols:
                stats = summarize_metric(df_group[metric_col])
                rows.append({
                    "pair_label": pair_label,
                    "pair_description": PAIR_LABELS.get(pair_label, pair_label),
                    "analysis_group": group_name,
                    "metric": metric_col,
                    "unit": unit,
                    "n_directions": len(df_group),
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "p95": stats["p95"],
                    "sam_bandlimit_applied": APPLY_SAM_BANDLIMIT,
                    "sam_analysis_freq_low_hz": SAM_ANALYSIS_FREQ_LOW if APPLY_SAM_BANDLIMIT else np.nan,
                    "sam_analysis_freq_high_hz": SAM_ANALYSIS_FREQ_HIGH if APPLY_SAM_BANDLIMIT else np.nan,
                })

    return pd.DataFrame(rows)


def build_decision_summary(summary_df):
    """
    Builds a decision table for each spatial group and metric.

    Now includes three comparisons:
      - head_only_vs_measured
      - head_torso_vs_measured
      - sim_sonicom_vs_measured

    Lower is better for all metrics considered.
    """
    rows = []
    groups = summary_df["analysis_group"].unique()
    metrics = summary_df["metric"].unique()

    def get_mean_unit(group, metric, pair_label):
        row = summary_df[
            (summary_df["pair_label"] == pair_label)
            & (summary_df["analysis_group"] == group)
            & (summary_df["metric"] == metric)
        ]
        if row.empty:
            return np.nan, ""
        return float(row.iloc[0]["mean"]), row.iloc[0]["unit"]

    for group in groups:
        for metric in metrics:
            mean_ho, unit = get_mean_unit(group, metric, "head_only_vs_measured")
            mean_ht, _ = get_mean_unit(group, metric, "head_torso_vs_measured")
            mean_ss, _ = get_mean_unit(group, metric, "sim_sonicom_vs_measured")

            if not (np.isfinite(mean_ho) and np.isfinite(mean_ht)):
                continue

            improvement_ho_minus_ht = mean_ho - mean_ht
            ratio_ht_over_ho = mean_ht / mean_ho if mean_ho != 0 else np.nan

            if np.isfinite(mean_ss):
                improvement_ss_minus_ht = mean_ss - mean_ht
                improvement_ho_minus_ss = mean_ho - mean_ss
                ratio_ss_over_ho = mean_ss / mean_ho if mean_ho != 0 else np.nan
                ht_lower_than_ss = bool(mean_ht < mean_ss)
                ss_lower_than_ho = bool(mean_ss < mean_ho)
            else:
                improvement_ss_minus_ht = np.nan
                improvement_ho_minus_ss = np.nan
                ratio_ss_over_ho = np.nan
                ht_lower_than_ss = False
                ss_lower_than_ho = False

            rows.append({
                "analysis_group": group,
                "metric": metric,
                "unit": unit,
                "head_only_vs_measured_mean": mean_ho,
                "head_torso_vs_measured_mean": mean_ht,
                "sim_sonicom_vs_measured_mean": mean_ss,
                "improvement_head_only_minus_head_torso": improvement_ho_minus_ht,
                "ratio_head_torso_over_head_only": ratio_ht_over_ho,
                "head_torso_lower_than_head_only": bool(mean_ht < mean_ho),
                "improvement_sim_sonicom_minus_head_torso": improvement_ss_minus_ht,
                "head_torso_lower_than_sim_sonicom": ht_lower_than_ss,
                "improvement_head_only_minus_sim_sonicom": improvement_ho_minus_ss,
                "ratio_sim_sonicom_over_head_only": ratio_ss_over_ho,
                "sim_sonicom_lower_than_head_only": ss_lower_than_ho,
                "sam_bandlimit_applied": APPLY_SAM_BANDLIMIT,
                "sam_analysis_freq_low_hz": SAM_ANALYSIS_FREQ_LOW if APPLY_SAM_BANDLIMIT else np.nan,
                "sam_analysis_freq_high_hz": SAM_ANALYSIS_FREQ_HIGH if APPLY_SAM_BANDLIMIT else np.nan,
            })

    return pd.DataFrame(rows)

# =========================================================
# PLOTS
# =========================================================

def make_scatter_plot(df, value_col, title, output_name, vmin=None, vmax=None):
    df_plot = df[np.isfinite(df[value_col].astype(float))].copy()
    if df_plot.empty:
        print(f"ATTENZIONE: nessun valore finito per scatter {value_col} - {output_name}. Plot saltato.")
        return

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df_plot["azimuth_norm_deg"],
        df_plot["elevation_deg"],
        c=df_plot[value_col],
        s=SCATTER_SIZE,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(sc, label=value_col)
    plt.xlabel("Azimuth [deg] (+ = left, - = right)")
    plt.ylabel("Elevation [deg]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=PLOT_DPI)
    plt.close()


def make_hist_plot(df, value_col, title, output_name, xlim=None):
    x = finite_values(df[value_col])
    if x.size == 0:
        print(f"ATTENZIONE: nessun valore finito per istogramma {value_col} - {output_name}. Plot saltato.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=30)
    plt.xlabel(value_col)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    if xlim is not None and np.all(np.isfinite(xlim)):
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=PLOT_DPI)
    plt.close()


def make_all_plots(all_metrics_df):
    metrics = ["ITD_diff_us", "ILD_diff_db", "LSD_db"]
    limits = {}
    for metric in metrics:
        finite_metric = finite_values(all_metrics_df[metric])
        if finite_metric.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(finite_metric, 99.0))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(np.max(finite_metric))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        limits[metric] = (0.0, vmax)

    for pair_label in all_metrics_df["pair_label"].unique():
        df_pair = all_metrics_df[all_metrics_df["pair_label"] == pair_label]
        desc = PAIR_LABELS.get(pair_label, pair_label)
        safe_pair = pair_label.replace(" ", "_")

        for metric in metrics:
            vmin, vmax = limits[metric]
            make_scatter_plot(
                df_pair,
                metric,
                f"{metric} - {desc} (100 Hz-10 kHz band-limited)",
                f"scatter_{safe_pair}_{metric}.png",
                vmin=vmin,
                vmax=vmax,
            )
            make_hist_plot(
                df_pair,
                metric,
                f"Distribuzione {metric} - {desc} (100 Hz-10 kHz band-limited)",
                f"hist_{safe_pair}_{metric}.png",
                xlim=(vmin, vmax),
            )


# =========================================================
# MAIN
# =========================================================

print(f"Output directory: {OUTPUT_DIR}")
print(f"Convenzione azimuth: {AZIMUTH_CONVENTION_NOTE}")
print(f"Band-limit SAM attivo: {APPLY_SAM_BANDLIMIT}")
if APPLY_SAM_BANDLIMIT:
    print(f"Banda SAM: {SAM_ANALYSIS_FREQ_LOW:.1f} - {SAM_ANALYSIS_FREQ_HIGH:.1f} Hz")
print("\nLettura dei file SOFA...")

sofas = {name: load_sofa(path) for name, path in SOFA_PATHS.items()}

# Raw/non-normalized copy used exclusively for computing the ITD.
# The broadband normalization is applied only to `sofas`, not to `sofas_raw_for_itd`.
sofas_raw_for_itd = {name: sofa.copy() for name, sofa in sofas.items()}

for name, sofa in sofas.items():
    print(f"{name}: Data_IR={sofa.Data_IR.shape}, SourcePosition={sofa.SourcePosition.shape}, fs={get_sampling_rate(sofa)}")

# Compatibility checks.
names = list(sofas.keys())
for other_name in names[1:]:
    if not same_sampling_rate(sofas[names[0]], sofas[other_name]):
        raise ValueError(f"Sample rate diversi tra {names[0]} e {other_name}.")
    if not same_source_positions(sofas[names[0]], sofas[other_name]):
        raise ValueError(
            f"SourcePosition diverse tra {names[0]} e {other_name}. "
            "Questo script assume griglie coincidenti per la normalizzazione mean-reference."
        )

fs = get_sampling_rate(sofas[names[0]])
min_len_for_norm = min(sofa.Data_IR.shape[2] for sofa in sofas.values())
positions = np.asarray(sofas["sim_head_only067"].SourcePosition)

# Normalization of the measured data with respect to the mean of the two simulations.
gain_offset_db = 0.0
gain_linear = 1.0
normalization_report_path = None

if APPLY_BROADBAND_GAIN_NORMALIZATION:
    gain_offset_db, offsets_per_location_ear, f_low_eff, f_high_eff = estimate_measured_offset_to_mean_reference_db(
        sofas=sofas,
        fs=fs,
        min_len=min_len_for_norm,
        target_dataset=GAIN_NORMALIZATION_TARGET_DATASET,
        reference_datasets=GAIN_NORMALIZATION_REFERENCE_DATASETS,
        f_low=GAIN_NORM_FREQ_LOW,
        f_high=GAIN_NORM_FREQ_HIGH,
        nfft=GAIN_NORM_NFFT,
    )

    gain_linear = 10.0 ** (gain_offset_db / 20.0)

    print("\n====================================")
    print("NORMALIZZAZIONE BROADBAND MISURATA")
    print("====================================")
    print(f"Target dataset:    {GAIN_NORMALIZATION_TARGET_DATASET}")
    print(f"Reference dataset: mean({', '.join(GAIN_NORMALIZATION_REFERENCE_DATASETS)})")
    print(f"Banda usata:       {f_low_eff:.1f} - {f_high_eff:.1f} Hz")
    print(f"Offset stimato:    {gain_offset_db:.3f} dB")
    print(f"Gain lineare:      {gain_linear:.6f}")

    sofas[GAIN_NORMALIZATION_TARGET_DATASET] = apply_gain_db_to_sofa(
        sofas[GAIN_NORMALIZATION_TARGET_DATASET], gain_offset_db
    )

    normalization_report_path = save_gain_normalization_report(
        offsets_per_location_ear=offsets_per_location_ear,
        positions=positions,
        offset_global_db=gain_offset_db,
        gain_linear=gain_linear,
        reference_datasets=GAIN_NORMALIZATION_REFERENCE_DATASETS,
        target_dataset=GAIN_NORMALIZATION_TARGET_DATASET,
        f_low=f_low_eff,
        f_high=f_high_eff,
    )
else:
    f_low_eff = np.nan
    f_high_eff = np.nan
    print("\nNormalizzazione broadband disattivata.")

normalization_info = {
    "applied": APPLY_BROADBAND_GAIN_NORMALIZATION,
    "method": GAIN_NORMALIZATION_METHOD if APPLY_BROADBAND_GAIN_NORMALIZATION else "none",
    "offset_db": gain_offset_db,
    "gain_linear": gain_linear,
    "f_low": f_low_eff if APPLY_BROADBAND_GAIN_NORMALIZATION else np.nan,
    "f_high": f_high_eff if APPLY_BROADBAND_GAIN_NORMALIZATION else np.nan,
}

# Compute SAM metrics for the two pairs.
all_pair_dfs = []
for dataset_a, dataset_b, pair_label in COMPARISON_PAIRS:
    df_pair = compute_sam_metrics_for_pair(
        pair_label=pair_label,
        sofa_a=sofas[dataset_a],
        sofa_b=sofas[dataset_b],
        positions=positions,
        normalization_info=normalization_info,
        sofa_a_for_itd=sofas_raw_for_itd[dataset_a],
        sofa_b_for_itd=sofas_raw_for_itd[dataset_b],
    )
    df_pair["dataset_a"] = dataset_a
    df_pair["dataset_b"] = dataset_b
    all_pair_dfs.append(df_pair)

all_metrics_df = pd.concat(all_pair_dfs, ignore_index=True)

# Save the full table.
all_csv = os.path.join(OUTPUT_DIR, "all_directions_sam_metrics.csv")
all_metrics_df.to_csv(all_csv, index=False)
print(f"\nTabella completa SAM salvata in: {all_csv}")

# Top 10 worst for each metric and pair.
for pair_label in all_metrics_df["pair_label"].unique():
    df_pair = all_metrics_df[all_metrics_df["pair_label"] == pair_label]
    for metric in ["ITD_diff_us", "ILD_diff_db", "LSD_db"]:
        df_metric = df_pair[np.isfinite(df_pair[metric].astype(float))].copy()
        top = df_metric.sort_values(metric, ascending=False).head(10)
        top_path = os.path.join(OUTPUT_DIR, f"top10_{pair_label}_{metric}.csv")
        top.to_csv(top_path, index=False)
        print(f"Top 10 {metric} per {pair_label} salvata in: {top_path}")

# Summary by spatial groups.
group_masks = build_group_masks(positions, tol=1.0)
summary_df = build_summary(all_metrics_df, group_masks)
summary_csv = os.path.join(OUTPUT_DIR, "sam_summary_by_pair_group_metric.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Summary SAM salvato in: {summary_csv}")

# Decision summary: head+torso better or worse than head-only for each metric/group.
decision_df = build_decision_summary(summary_df)
decision_csv = os.path.join(OUTPUT_DIR, "sam_decision_summary.csv")
decision_df.to_csv(decision_csv, index=False)
print(f"Decision summary SAM salvato in: {decision_csv}")

# Plots.
if MAKE_SCATTER_PLOTS or MAKE_HISTOGRAMS:
    make_all_plots(all_metrics_df)
    print(f"Grafici salvati nella cartella: {OUTPUT_DIR}")

# Terminal summary.
print("\n" + "=" * 80)
print("RISULTATO PRINCIPALE - SAM METRICS CONTRO MISURA SONICOM")
print("=" * 80)
print("Gruppo globale: global_all. Valori medi su tutte le direzioni.")
if APPLY_SAM_BANDLIMIT:
    print(f"Metriche SAM calcolate su HRIR band-limited {SAM_ANALYSIS_FREQ_LOW:.0f}-{SAM_ANALYSIS_FREQ_HIGH:.0f} Hz.")
print("Lower is better per ITD_diff_us, ILD_diff_db e LSD_db.\n")

for metric in ["ITD_diff_us", "ILD_diff_db", "LSD_db"]:
    row_ho = decision_df[(decision_df["analysis_group"] == "global_all") & (decision_df["metric"] == metric)]
    if row_ho.empty:
        continue
    r = row_ho.iloc[0]
    print(
        f"{metric}: "
        f"head-only={r['head_only_vs_measured_mean']:.4f}, "
        f"head+torso={r['head_torso_vs_measured_mean']:.4f}, "
        f"sim_sonicom={r['sim_sonicom_vs_measured_mean']:.4f}, "
        f"impr_HO-HT={r['improvement_head_only_minus_head_torso']:.4f}, "
        f"ratio_HT/HO={r['ratio_head_torso_over_head_only']:.4f}, "
        f"HT_better_HO={r['head_torso_lower_than_head_only']}, "
        f"HT_better_SONICOM={r['head_torso_lower_than_sim_sonicom']}"
    )

print("\nDettaglio per piani principali:")
for group in ["horizontal_plane", "median_plane", "median_front", "median_back", "lateral_plane", "lateral_left", "lateral_right"]:
    print(f"\n{group}:")
    for metric in ["ITD_diff_us", "ILD_diff_db", "LSD_db"]:
        rows = decision_df[(decision_df["analysis_group"] == group) & (decision_df["metric"] == metric)]
        if rows.empty:
            continue
        r = rows.iloc[0]
        print(
            f"  {metric}: "
            f"head-only={r['head_only_vs_measured_mean']:.4f}, "
            f"head+torso={r['head_torso_vs_measured_mean']:.4f}, "
            f"sim_sonicom={r['sim_sonicom_vs_measured_mean']:.4f}, "
            f"HT_better_HO={r['head_torso_lower_than_head_only']}, "
            f"HT_better_SONICOM={r['head_torso_lower_than_sim_sonicom']}"
        )

print("\nFile principali generati:")
print(f"  - {os.path.basename(all_csv)}")
print(f"  - {os.path.basename(summary_csv)}")
print(f"  - {os.path.basename(decision_csv)}")
if normalization_report_path is not None:
    print(f"  - {os.path.basename(normalization_report_path)}")
print("  - top10_<pair>_<metric>.csv")
print("  - scatter_<pair>_<metric>.png")
print("  - hist_<pair>_<metric>.png")

print("\nNota:")
print("  ITD difference e' riportata anche in microsecondi.")
print("  ITD e' calcolata su HRIR raw/non normalizzate in gain, poi band-limited se richiesto.")
print("  ILD difference e LSD sono in dB.")
print("  ITD e ILD sono calcolate con SAM; LSD e' calcolata in modo log-spettrale sulla banda 100 Hz-10 kHz con maschera esplicita, per evitare inf dovuti ai bin azzerati sopra 10 kHz.")
print(f"  Band-limit SAM applicato: {APPLY_SAM_BANDLIMIT}")
if APPLY_SAM_BANDLIMIT:
    print(f"  Banda SAM: {SAM_ANALYSIS_FREQ_LOW:.0f}-{SAM_ANALYSIS_FREQ_HIGH:.0f} Hz")
print(f"  Normalizzazione broadband applicata: {APPLY_BROADBAND_GAIN_NORMALIZATION}")
if APPLY_BROADBAND_GAIN_NORMALIZATION:
    print(f"  Metodo normalizzazione: {GAIN_NORMALIZATION_METHOD}")
    print(f"  Offset misurata applicato: {gain_offset_db:+.3f} dB")
print(f"  Convenzione azimuth usata: {AZIMUTH_CONVENTION_NOTE}")
