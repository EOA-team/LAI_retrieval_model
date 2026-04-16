import os
import pandas as pd



# Put the soil spectra of the different sites into one file
# Hyperspectral data
# S2 resolution data

countries = ['switzerland', 'italy', 'bulgaria']

data_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/code/baresoil')

hyperspectral = []
s2 = []
for country in countries:

    field_soil_dir = os.path.join(data_dir, f'{country}_fields')
    for f in os.listdir(field_soil_dir):
        # Combine hypersecptral data (subsample per field)
        if f.endswith('.pkl') and 'F2' not in f:
            df = pd.read_pickle(os.path.join(field_soil_dir, f)).sample(5, random_state=42)
            sampled_soils = df.index
            hyperspectral.append(df)
            # Combine S2 data (subsample per field)
            df = pd.read_csv(os.path.join(field_soil_dir, f'{f.split("_1nm.pkl")[0]}.csv'))
            df = df.iloc[sampled_soils]
            s2.append(df)
    

hyperspectral = pd.concat(hyperspectral, ignore_index=True)
s2 = pd.concat(s2, ignore_index=True)

hyperspectral.to_pickle(os.path.join(data_dir, 'sampled_soil_spectra_multifield_1nm.pkl'))
s2.to_csv(os.path.join(data_dir, 'sampled_soil_spectra_multifield.csv'), index=False)