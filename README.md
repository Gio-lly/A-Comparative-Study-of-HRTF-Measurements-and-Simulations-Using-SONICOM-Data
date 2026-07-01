# A Comparative Study of HRTF Measurements and Simulations Using SONICOM Data

Code, meshes and simulated data for the 2026 paper comparing **measured** and
**numerically simulated** Head-Related Transfer Functions (HRTFs), written for the
**MAE (Music and Acoustic Engineering) Capstone** course at Politecnico di Milano.

The work takes a subset of subjects from the [SONICOM](https://www.sonicom.eu/)
HRTF dataset, reconstructs their head (and, for some subjects, head+torso) meshes,
runs a **BEM acoustic simulation** with [Mesh2HRTF / NumCalc](https://github.com/Any2HRTF/Mesh2HRTF)
on the exact SONICOM measurement grid, and then compares the resulting HRTFs against:

- the **official SONICOM numerical simulation** (`sim_sonicom`), and
- the **SONICOM acoustic measurement** (`mis_sonicom`).

The comparison is done both **spectrally** (HRTF magnitude maps, per-direction spectra,
difference maps) and with **perceptually-motivated metrics** (ITD, ILD, LSD) computed
with the [`spatialaudiometrics`](https://github.com/Katarina-Poole/Spatial-Audio-Metrics)
(SAM) toolbox.

---

## Pipeline overview

```
 3D mesh (STL, per subject)
        │
        ▼
 [1] Mesh preprocessing        Mesh Preprocessing/
        │   fix_mesh_hrtf.py    → diagnose & repair the raw mesh
        │   Blender + grading   → per-ear "graded" meshes (…_gradedL / …_gradedR.blend)
        │   check_graded.py     → sanity-check the graded mesh before solving
        │   create_grid.py      → export the SONICOM source grid as EvaluationGrid.txt
        ▼
 [2] BEM simulation            (external: Mesh2HRTF / NumCalc — not included here)
        │                        produces one SOFA file per subject/ear, merged
        ▼
 [3] Simulated HRTFs           SOFA files/
        │   *_HRIR_*.sofa / *_HRTF_*.sofa  + preview plots (2D PDF, 3D JPEG)
        ▼
 [4] Analysis & comparison     src/
            SpectralAnalysis.py        → HRTF magnitude / difference maps + CSV
            SamAnalysisHeadOnly.py     → ITD / ILD / LSD metrics + CSV
            (head+torso variants under src/src_head_and_torso/)
```

Steps **[1]**, **[3]** and **[4]** live in this repository. Step **[2]**, the BEM solve,
is run with the external Mesh2HRTF/NumCalc tool and only its SOFA outputs are stored here.

---


### Naming conventions

- **Subject IDs** follow SONICOM: `P039`, `P067`, `P081`, `P143` …
- **`_owntorso`** — simulation performed with the subject's *own* reconstructed torso.
- **`_torsokemar`** — head simulated on a generic **KEMAR** manikin torso.
- **`_gradedL` / `_gradedR`** — the mesh *graded* (locally refined near the target ear)
  for the **left** and **right** ear respectively. Mesh2HRTF grades one mesh per ear.
- **`HRIR`** = impulse responses (time domain); **`HRTF`** = transfer functions (frequency
  domain). `grid` marks the SONICOM measurement grid; the trailing number is the sample
  rate (`48000` / `44100` Hz).
- Preview files `*_2D.pdf` and `*_3D_horizontal_plane.jpeg` are quick visual checks of
  each SOFA file (horizontal-plane HRTF).

---

## Requirements

The scripts are plain Python 3 (developed on Linux; paths in the scripts are Linux-style
and must be edited for your machine — see *Configuration* below).

```bash
# Mesh preprocessing
pip install trimesh pymeshlab numpy rtree netCDF4

# Analysis
pip install sofar numpy pandas matplotlib spatialaudiometrics
```

- `rtree` is optional but recommended: without it, `check_graded.py` skips the
  self-intersection test.
- Mesh grading itself is done with **Blender** + **Mesh2HRTF's `hrtf_mesh_grading`**, and
  the BEM solve with **NumCalc** — both external to this repo.

---

## The scripts

### Step 1 — Mesh preprocessing (`Mesh Preprocessing/`)

#### `fix_mesh_hrtf.py` — diagnose & repair a raw mesh
Checks a head/torso mesh *before* grading and, optionally, repairs it. A non-watertight,
non-manifold or fragmented mesh is the classic cause of a `Segmentation fault` in the
grading/BEM stage.

```bash
python fix_mesh_hrtf.py input.stl                     # DIAGNOSE only
python fix_mesh_hrtf.py input.stl output.ply          # diagnose + SOFT repair
python fix_mesh_hrtf.py input.stl output.ply --poisson # diagnose + POISSON reconstruction
```

- **Diagnosis** reports vertex/face count, bounding box, watertightness, winding
  consistency, number of shells, open (boundary) edges and non-manifold edges.
- **Soft repair** removes duplicates/null faces, keeps only the largest shell, repairs
  non-manifold edges/vertices, closes holes and re-orients faces coherently.
- **Poisson** rebuilds a guaranteed-closed surface (rounds off concavities — re-check the
  ear/pinna afterwards).

#### `check_graded.py` — sanity-check the graded mesh before NumCalc
Runs every geometric check known to make the BEM solver crash or diverge, and returns a
shell exit code you can use in a script.

```bash
python check_graded.py path/3Dmesh_graded_right.ply
python check_graded.py path/3Dmesh_graded_right.ply --fmax 20000   # target max frequency (Hz)
```

Checks performed: scale/units (Mesh2HRTF expects **metres**, head width 0.10–0.18 m),
connected components, watertight & manifold, outward-facing normals (positive volume),
degenerate/sliver triangles, self-intersections, **edge length vs. the λ/6 rule** at the
target frequency, and interaural centring at the origin.

**Exit codes:** `0` = ready to simulate · `1` = minor warnings (risky) · `2` = serious
problems (the solve will very likely fail).

#### `create_grid.py` — export the SONICOM evaluation grid
Reads the source positions from a SONICOM SOFA file and writes them as a clean
`EvaluationGrid.txt` (cartesian `X Y Z`) that Mesh2HRTF uses as its *EvaluationGrid*, so the
simulation is evaluated on **exactly the same directions** as the measurement. Spherical
`(az, el, r)` coordinates are auto-detected and converted to cartesian.

> Edit the hard-coded `sofa_path` at the top of the file before running:
> ```bash
> python create_grid.py     # writes EvaluationGrid.txt in the working directory
> ```

#### `Blender_projects/`
The grading step is done in Blender; the saved `.blend` projects contain the final
**graded** meshes, one file per ear (`…_gradedL.blend`, `…_gradedR.blend`) and, for the
head+torso subjects, the corresponding `…_torso_graded{L,R}.blend`.

### Step 4 — Analysis (`src/`)

Both analysis families come in a **head-only** variant (`src/src_head_only/`) and a
**head+torso** variant (`src/src_head_and_torso/`). They compare three SOFA datasets at a
time and write plots + CSV summaries.

Datasets compared:

| key            | meaning                                             |
| -------------- | --------------------------------------------------- |
| `sim_nostra` / `sim_head_only` | our BEM simulation (head only)      |
| `sim_head_torso` | our BEM simulation with torso (head+torso variant) |
| `sim_sonicom`  | official SONICOM numerical simulation               |
| `mis_sonicom`  | SONICOM acoustic **measurement** (the reference)    |

#### `SpectralAnalysis.py` / `HeadAndTorso_SpectralAnalysis.py` — spectral comparison
For a set of target directions (principal directions, horizontal plane every 30°, median
front/back, lateral left/right) it:

- matches each target to the nearest measured direction (true angular distance on the sphere),
- caches HRTF magnitude (dB) via FFT of the HRIR,
- optionally applies a **broadband gain normalisation** so datasets are compared on spectral
  *shape*, not on a global level offset,
- saves per-direction **HRIR / HRTF / difference** plots, compact multi-direction grids, and
  **frequency×direction magnitude maps** and **difference maps** (shared colour scales),
- writes CSV summaries: `selected_directions_all_sets.csv`,
  `azimuth_convention_sanity_check.csv`, `broadband_gain_normalization_offsets.csv`,
  `spectral_difference_summary.csv`.

Output goes to a `plots_compare_multi_directions_broadband_norm/` folder next to the script.
The head+torso variant restricts the band to 100–10000 Hz (more robust) and normalises the
measurement against a *neutral* reference (mean of head-only and head+torso).

#### `SamAnalysisHeadOnly.py` / `HeadAndTorso_SAM_Analysis.py` — SAM metrics
Uses the `spatialaudiometrics` toolbox to compute, per direction and as global summaries:

- **ITD** difference (max-IACC estimator),
- **ILD** difference (RMS estimator),
- **LSD** (Log-Spectral Distortion),

after matching HRTF locations and applying the same pairwise broadband normalisation. It
writes, per comparison pair, `all_directions_metrics.csv`, `top10_{itd,ild,lsd}*.csv`, and
scatter/histogram plots, plus global `all_directions_metrics_all_pairs.csv` and
`summary_metrics_by_pair.csv`, into `comparison_output_three_pairs/`.

> ITD is essentially unaffected by a global gain; ILD is unaffected if the same gain is
> applied to both ears; **LSD** is the metric the broadband normalisation mainly protects.

---

## Configuration / how to run

All four analysis scripts have their SOFA file paths **hard-coded at the top** (Linux paths
like `/home/capstone/Downloads/P067/…`). Before running:

1. Open the script and edit the `files` / `SOFA_PATHS` dictionary to point at your local
   SOFA files (one entry per dataset key).
2. Run the script directly:
   ```bash
   cd src/src_head_only
   python SpectralAnalysis.py
   python SamAnalysisHeadOnly.py
   ```
3. Outputs (plots + CSVs) are created in a subfolder next to the script.

### Coordinate / azimuth convention

Throughout, the **SONICOM/SOFA** convention is used:

- azimuth `0°` = front, `+90°` = **listener's left**, `-90°` = **listener's right**, `±180°` = back;
- cartesian axes: `+x` front, `+y` left, `+z` up.

`SpectralAnalysis.py` prints and saves an *azimuth sanity check* (left/right HRIR energy at
±90°) so you can confirm the convention holds for each file before trusting the results.

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE). © 2026 Giorgio Mattina.


