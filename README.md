# Leaf Area Index Retrieval
## Soil-informed, large-scale, winter wheat Sentinel-2 LAI model


This repository contains the Python code and data required to re-run the analysis and results presented in

> **_PAPER:_**  Ledain S., Gilgen A., Aasen, H. (2026) "Soil-informed PROSAIL modelling improves scalable retrieval of leaf area index: evidence from multi-year, multi-country winter wheat observations". *Under review*.

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
Extract, cluster and sample soil spectra across countries and study fields. 

Data inputs: 
- bare soil spectra collected from DLR SoilSuite
- CORINE land use classification to identify agricultural areas
- Field boundaries 

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


#### 3. LAI retrieval model
#### 4. Analysis


### Data

### Results