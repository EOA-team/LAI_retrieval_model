# Leaf Area Index Retrieval
## Soil-informed, large-scale, winter wheat Sentinel-2 LAI model


This repository contains the Python code and data required to re-run the analysis and results presented in

> **_PAPER:_**  Ledain S., Gilgen A., Aasen H. (2026) "Soil-informed PROSAIL modelling improves scalable retrieval of leaf area index: evidence from multi-year, multi-country winter wheat observations". *Under review*.

We therefore kindly ask you to **acknowledge our work** by

* **citing** our research properly whenever you use the data and/or methods presented here
* leave a **star on GitHub** and/or fork our repository

This helps us to continue the labor and cost-intensive process of data acquisition, preparation and, ultimately, publication to benefit science and society.

If your work relies substantially on our data please also [get in touch with us](https://www.eoa-team.net/) and consider offering co-authorship.


## Content

### Data
For access to the data, please contact us.

### Code

#### 1. Baresoil spectra representation
Use a bare soil composite (DLR SoilSuite) and extract, cluster and sample soil spectra across countries and study fields. This code creates soil spectra datasets at different scales.

Data inputs: 
- bare soil spectra collected from DLR SoilSuite
- CORINE land use classification to identify agricultural areas
- Field boundaries (for site/multisite datasets)

Outputs:
- For each field in each country
  - CSV file containing dataset of soil spectra: `{country.lower()}_fields/sampled_soil_spectra_{country}_{field}.csv`
  - Dataset upsampled to 1nm (CSV and .pkl): `{country.lower()}_fields/sampled_soil_spectra_{country}_{field}_1nm.csv`, `{country.lower()}_fields/sampled_soil_spectra_{country.lower()}_{field}_1nm.pkl`
- A dataset for all fields at once: `sampled_soil_spectra_multifield.csv`, `sampled_soil_spectra_multifield_1nm.pkl`
- A dataset to represent soil spectra across all countries: 'sampled_spectra_k5_n1000_uniform_1nm.csv', 'sampled_spectra_k5_n1000_uniform_1nm.pkl', 'soil_spectra_k5_n1000_uniform_countries.csv'
- A Kmeans clustering model trained to cluster soil spectra (arable land) into 5 clusters: `kmeans_soil_k5_countries.pkl`

```
# In 'baresoil' folder
python bare_soil_multicountry.py # Across large scale
python bare_soil_multisite.py # Soil dataset across study sites
python bare_soil_sites.py # Soil dataset per study site
```


#### 2. Radiative transfer model
This code allows to run PROSAIL in forward mode. The input parameters are set to the same ranges and distributions as for SNAP LAI (http://step.esa.int/docs/extra/ATBD_S2ToolBox_V2.1.pdf), as well as the parameter codistribution. The runs output Sentinel-2 like relfectances. The background soil used in PROSAIL is modified, using the spectra collected in the sections above.

To generate PROSAIL simulations:
```
# In ProSAIL_forward
python simulate_s2_spectra_soil.py
```
A dataframe (row=observation, columns=ProSAIL parameters and S2 bands) is created and saved to `.pkl` file.

The script reads from `ProSAIL_forward/RTM_config.yaml`:
- parameters passed to PROSAIL in `lut_params`. The parameter settings used in this project are saved in `ProSAIL_forward/lut_params`.
- Number of simulations in `lut_size` (50k for Sentinel-2A and 50k for Sentinel-2B, 10k each for test LUTs)
- How to codistribute the parameters by passing a file to `codistribution`
- sensor type in `sensor` (Sentinel-2A or Sentinel-2B)
- where files are written in `out_dir`
- soil spectra to use in `soil_path`. Expects a `.pkl` file containing a dataframe where each row is a spectra with 1nm resoltuion (columns should be the bands between 400 and 2100nm). If `None`, the background spectra used are those provided in `ProSAIL_forward/prosail`.

> [!NOTE]
> Edit the name of the output file (look-up table saved as a pickled dataframe), edit the `fpath_lut` variable in the `generate_spectra_soil` function (or `generate_spectra` if no soil data is passed)


#### 3. LAI retrieval model
Once the look-up tables are generated with PROSAIL, they can be used to train neural network-based LAI retrieval models. An ensemble of 5 neural network with different seeds are trained, and results are always the average of predictions.

First, configure the model and set up in `configs/config_NN.yaml`:
- In the `Model` section: specific model parameters can be passed, as well as where the model and results will be saved.
- In the `Data` section: path to LUTs for traiing and testing, paths to bare soil samples if needed, relevant columns for train/predict.
- In the `Tuning` section: define the number of hyperparameter trials (Optuna package used for tuning)

> [!TIP]
> If no soil data is to be used for the model development, comment out `baresoil_samples` in the config file
 
Tune the model:
```
python tune.py ../configs/config_NN.yaml
```
will use the hyperparameter ranges passed in the `objective` function. The hyperparameter combinations and scores are saved in an excel file in `tuning_results/` as `{model_name}_tuning.xlsx` (model name is based on save path in config file).

Train the model with specific hyperparameters:
```
python train.py ../configs/config_NN.yaml
```
The models are saved at the path defined in the config as `{save_path}{seed_nbr}.pkl`. The model results (scores for each seed) as saved in the `score_path` provided in the config file.

> [!NOTE]
> Noise is added to each band using the levels defined in `noise_snap.csv`
 


#### 4. Analysis

Compute SNAP on validation data
```
python snap_baseline.py
```

Script used to compare models and generate plots
```
python compare_models.py
```

### Data

### Results