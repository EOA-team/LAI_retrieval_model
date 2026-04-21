# Soil-informed LAI Retrieval from Sentinel-2

**Soil-informed, large-scale winter wheat LAI retrieval using PROSAIL and neural networks**

This repository contains the code accompanying:

> Ledain S., Gilgen A., Aasen H. (2026) "Soil-informed PROSAIL modelling improves scalable retrieval of leaf area index: evidence from multi-year, multi-country winter wheat observations." *Under review.*

If you use this code or the methods presented here, please **cite our paper** and consider leaving a ⭐ on GitHub.  
If your work relies substantially on our data, please [get in touch](https://www.eoa-team.net/) and consider co-authorship.

For access to the data, please contact us.

---

## Overview

We present a pipeline that improves satellite-based Leaf Area Index (LAI) retrieval for winter wheat by explicitly incorporating **local bare soil spectral information** into PROSAIL radiative transfer model (RTM) simulations. These simulations are used to train neural network ensemble models that retrieve LAI from Sentinel-2 reflectances. The approach is validated across multiple countries (Switzerland, Bulgaria, Italy, Poland) and years.

### Key contributions

- **Soil-informed PROSAIL**: Each RTM simulation uses an observed bare soil spectrum from the target region, making the look-up table (LUT) geographically representative.
- **Multi-scale evaluation**: Models trained and validated at field, site, and cross-country scales.
- **Noise-robust training**: Sensor noise (calibrated from ESA SNAP) is injected during training to improve robustness to real Sentinel-2 observations.
- **Ensemble prediction**: Five independently trained networks are averaged to reduce variance.
- **SNAP baseline**: ESA's operational S2 Biophysical Processor (SNAP LAI) is re-implemented in Python for reproducible comparison.

---

## Repository structure

```
LAI_retrieval_model/
├── code/
│   ├── baresoil/                   # Step 2: Bare soil spectra extraction & clustering
│   │   ├── bare_soil_multicountry.py
│   │   ├── bare_soil_multisite.py
│   │   ├── bare_soil_sites.py
│   │   └── bare_soil_GEE*.py       # Google Earth Engine extraction
│   ├── ProSAIL_forward/            # Step 3: RTM forward simulations
│   │   ├── simulate_S2_spectra_soil.py
│   │   ├── RTM_config.yaml
│   │   ├── lut_params/             # PROSAIL parameter distributions (per country)
│   │   └── rtm_inv/                # LUT generation helpers
│   ├── train.py                    # Step 4: Train neural network ensemble
│   ├── tune.py                     # Step 4: Hyperparameter tuning (Optuna)
│   ├── test.py                     # Evaluate on in-situ validation data
│   ├── snap_baseline.py            # ESA SNAP LAI baseline
│   ├── compare_models.py           # Statistical comparison & plots
│   └── noise_snap.csv              # Per-band noise levels (from SNAP ATBD)
├── configs/
│   ├── config_NN.yaml              # Training configuration (main)
│   └── config_NN_field.yaml        # Field-level configuration variant
├── models/
│   ├── NN.py                       # NeuralNetworkRegressor (PyTorch)
│   ├── snap.py                     # SNAP LAI re-implementation
│   └── __init__.py
├── data/
│   ├── insitu_S2/                  # Paired in-situ LAI + S2 reflectance data
│   └── S2_baresoil_GEE/            # Bare soil composites from GEE
└── requirements.txt
```

---

## Workflow

### Step 1 — Prepare validation data

Validation data should be pickled pandas DataFrames where each row is a pixel and columns are Sentinel-2 band reflectances plus the corresponding field-measured LAI value.

Required columns: `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`, `lai`

### Step 2 — Bare soil spectra representation

Extract, cluster, and sample bare soil spectra across countries and study fields using the DLR SoilSuite hyperspectral dataset, masked to arable land (CORINE land cover).

```bash
cd code/baresoil

python bare_soil_multicountry.py   # Cross-country soil dataset (k=5 KMeans clusters)
python bare_soil_multisite.py      # Multi-site soil dataset
python bare_soil_sites.py          # Per-field soil dataset
```

**Inputs:**
- DLR SoilSuite hyperspectral bare soil spectra
- CORINE land use classification (arable land mask)
- Field boundaries (shapefiles)

**Outputs:**

| File | Description |
|---|---|
| `sampled_soil_spectra_{country}_{field}_1nm.pkl` | Per-field soil spectra upsampled to 1 nm |
| `sampled_soil_spectra_multifield_1nm.pkl` | All fields combined |
| `sampled_spectra_k5_n1000_uniform_1nm.pkl` | Cross-country dataset (k=5 clusters, n=1000/cluster) |
| `kmeans_soil_k5_countries.pkl` | Trained KMeans model |

### Step 3 — PROSAIL forward simulations

Run PROSAIL in forward mode to generate a look-up table (LUT) of simulated Sentinel-2 reflectances paired with biophysical variables. The key innovation is replacing PROSAIL's default background with observed soil spectra.

```bash
cd code/ProSAIL_forward
python simulate_S2_spectra_soil.py
```

Configuration is read from `RTM_config.yaml`:

| Key | Description |
|---|---|
| `lut_params` | Path to PROSAIL parameter distribution CSV (see `lut_params/` for per-country files) |
| `codistribution` | Path to parameter co-distribution file (`codistribution_snap.csv` mirrors SNAP ATBD) |
| `lut_size` | Number of simulations (default: 50 000 per sensor) |
| `sampling_method` | Sampling strategy — `FRS` (Full Random Sampling) supported |
| `sensor` | `Sentinel2A` or `Sentinel2B` |
| `fpath_srf` | Path to Sentinel-2 spectral response functions (`.xlsx`) |
| `soil_path` | Path to soil spectra `.pkl` (set to `null` for PROSAIL default background) |
| `traits` | Biophysical variables stored in the LUT, e.g. `['lai', 'cab', 'ccc', 'car']` |
| `out_dir` | Output directory |
| `remove_invalid_green_peaks` | Remove simulations with implausible green reflectance peaks (recommended: `true`) |
| `apply_glai_ccc_constraint` | Enforce co-distribution constraint between green LAI and CCC |
| `apply_chlorophyll_carotiniod_constraint` | Enforce Cab/Car ratio constraint |

Per-country parameter distribution files are provided in `code/ProSAIL_forward/lut_params/`:
- `prosail_danner-etal_europe_soil_snap.csv` — cross-country (default, follows SNAP ATBD)
- `prosail_danner-etal_switzerland_soil.csv`, `..._bulgaria_soil.csv`, `..._italy_soil.csv` — country-specific
- `prosail_danner-etal_multifield_soil_snap.csv` — multi-site variant

> **Note:** Edit the `fpath_lut` variable inside `generate_spectra_soil()` to set the output filename.

PROSAIL parameter ranges follow the SNAP S2 Toolbox ATBD (Danner et al.) to ensure comparability with the ESA operational processor.

**Output:** A pickled DataFrame (rows = simulations, columns = PROSAIL parameters + S2 band reflectances).

### Step 4 — LAI retrieval model

#### 4a. Hyperparameter tuning

```bash
python tune.py ../configs/config_NN.yaml
```

Uses [Optuna](https://optuna.org/) Bayesian optimisation. Search space: batch size, hidden layer size, optimizer, learning rate, LR scheduler. Results saved to `tuning_results/{model_name}_tuning.xlsx`.

#### 4b. Training

```bash
python train.py ../configs/config_NN.yaml
```

Trains an **ensemble of 5 neural networks** (one per random seed). Configuration is set in `configs/config_NN.yaml`:

```yaml
Model:
  save_path: ../models/NN_europe_soil_tuned.pkl  # Seed index appended automatically per seed
  score_path: ../model_results/NN_europe_soil_tuned.xlsx
  noise: True   # Apply SNAP-calibrated sensor noise during training
  gpu: False

Data:
  data_path:       # PROSAIL LUTs used for training
    - ProSAIL_forward/results/prosail_danner-etal_europe_soil_snap_soil_S2A_lut.pkl
    - ProSAIL_forward/results/prosail_danner-etal_europe_soil_snap_S2B_lut.pkl
  test_data_path:  # Held-out LUT subset for model selection
    - ProSAIL_forward/results/test/prosail_danner-etal_europe_soil_snap_S2A_lut.pkl
    - ProSAIL_forward/results/test/prosail_danner-etal_europe_soil_snap_S2B_lut.pkl
  baresoil_samples:  # Optional: real bare soil pixels anchored at LAI=0
    - ../code/baresoil/soil_spectra_k5_n1000_uniform_countries.csv #Sentinel-2 band resolution
  train_cols: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
  target_col: lai
  normalize: True

Tuning:
  n_trials: 10

Seed: [0, 1, 2, 3, 4]
```

**Model architecture (`models/NN.py`):**
- Input: 10 Sentinel-2 bands (B02–B12, excluding B10)
- Hidden: 1 fully connected layer (48 neurons, ReLU)
- Output: 1 neuron (LAI)
- Normalisation: MinMaxScaler saved alongside each model as `{save_path}{seed}_scaler.pkl`

**Training details:**
- Sensor noise (additive + multiplicative, from `noise_snap.csv`) is injected into training samples
- If `baresoil_samples` is provided, bare soil pixels are included with LAI = 0 as anchor points
- Final prediction = mean across all 5 ensemble members

### Step 5 — Evaluation and comparison

#### Evaluate a trained model on in-situ validation data

```bash
python test.py ../configs/config_NN.yaml
```

Loads each of the 5 trained ensemble models, runs inference on the in-situ paired data specified in `val_data_path`, and writes per-seed metrics (RMSE, nRMSE, R²) to the Excel score file. Generates scatter plots of predicted vs. measured LAI (optionally coloured by site or site-year).

#### SNAP baseline

```bash
python snap_baseline.py
```

Applies the ESA SNAP S2 Biophysical Processor LAI algorithm (re-implemented in `models/snap.py`) to the validation datasets.

#### Cross-model comparison

```bash
python compare_models.py
```

Loads trained models for all country/scale/configuration variants, runs inference on all in-situ validation sets, and produces comparative boxplots of RMSE broken down by country and soil group.

---

## Installation

```bash
git clone https://github.com/<your-org>/LAI_retrieval_model.git
cd LAI_retrieval_model
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Role |
|---|---|---|
| `prosail` | 2.0.5 | PROSAIL RTM |
| `torch` | — | Neural network (PyTorch) |
| `optuna` | 4.2.1 | Hyperparameter optimisation |
| `scikit-learn` | 1.6.1 | Scaling, clustering, metrics |
| `pandas` / `numpy` | 2.1.3 / 1.26.2 | Data handling |
| `geopandas` / `rasterio` | 0.13.2 / 1.3.9 | Geospatial I/O |
| `scipy` | 1.13.0 | Interpolation, statistics |
| `statsmodels` | 0.14.2 | Friedman test, Wilcoxon test |
| `matplotlib` / `seaborn` | 3.10.1 / 0.13.0 | Visualisation |

---

## Citation

```bibtex
@article{ledain2026soilinformed,
  title   = {Soil-informed {PROSAIL} modelling improves scalable retrieval of leaf area index:
             evidence from multi-year, multi-country winter wheat observations},
  author  = {Ledain, Selene and Gilgen, Anna and Aasen, Helge},
  journal = {Under review},
  year    = {2026}
}
```

---

## License

Please contact the authors before reusing or redistributing this code.  
See [eoa-team.net](https://www.eoa-team.net/) for contact information.
