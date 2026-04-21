import os
from pathlib import Path
import sys
base_dir = str(Path(os.path.dirname(os.path.realpath("__file__"))).parent) + '/models/'
sys.path.append(base_dir)
from snap import predict_snap_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np


# Load snap model
# Apply on fields per country, all fields at once
# Maybe only licor for CH?

angles = {
  'switzerland': {'mean_solar_zenith':35, 'mean_sensor_zenith':7, 'relative_azimuth':80},
  'bulgaria': {'mean_solar_zenith':45, 'mean_sensor_zenith':7, 'relative_azimuth':90},
  'italy': {'mean_solar_zenith':40, 'mean_sensor_zenith':6, 'relative_azimuth':90}
}

########################
# PARAMS

country = 'italy' #'bulgaria' #'switzerland' # , 
fields_seperate = True
all_countries = ['switzerland', 'bulgaria', 'italy'] #[] 

data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_S2/{country.lower()}_fields')
out_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/snap_baseline/{country}')
os.makedirs(out_dir, exist_ok=True)
results_file = os.path.join(out_dir, 'snap_predictions_clean.xlsx')

mean_solar_zenith = angles[country]['mean_solar_zenith']
mean_sensor_zenith = angles[country]['mean_sensor_zenith']
relative_azimuth = angles[country]['relative_azimuth']


########################
# PREDICT WITH SNAP AND SAVE RESULTS (excel and plots)

# Open each field data and apply model
if fields_seperate:
  with pd.ExcelWriter(results_file, engine="xlsxwriter") as writer:
    for f in os.listdir(data_dir):

      if country == 'switzerland':
        if 'licor' not in f or 'clean' not in f:
          continue
      
      if 'clean' in f:
        field_name = f.split('_')[-2].split('.pkl')[0]
      else:
        field_name = f.split('_')[-1].split('.pkl')[0]
      df = pd.read_pickle(os.path.join(data_dir, f))
 
      # Add angle data
      df['mean_solar_zenith'] = mean_solar_zenith
      df['mean_sensor_zenith'] = mean_sensor_zenith
      df['relative_azimuth'] = relative_azimuth

      df['snap_LAI'] = predict_snap_df(df) # needs to be reflectance data!
      df.to_excel(writer, sheet_name=field_name, index=False)

      # Compute scores
      if len(df)>1:
        rmse = root_mean_squared_error(df['lai'], df['snap_LAI'])
        nrmse = rmse/(df['lai'].max() - df['lai'].min())
        r2 = r2_score(df['lai'], df['snap_LAI'])
      else:
        rmse = root_mean_squared_error(df['lai'], df['snap_LAI'])
        nrmse = np.nan
        r2 = np.nan
    

      # Plot results for a field
      fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
      y = df['snap_LAI'].to_numpy(dtype=float)
      y_true = df['lai'].to_numpy(dtype=float)
      scatter = axs.scatter(y_true, y)
      axs.set_xlabel('Measured LAI', size=18)
      axs.set_ylabel('SNAP LAI', size=18)
      valmin = min(y_true.min(), y.min())
      valmax = max(y_true.max(), y.max())
      buffer = (valmax-valmin)*0.1
      if buffer == 0:  # all points identical
          buffer = 2  # or some small default value
      axs.set_xlim((valmin-buffer, valmax+buffer))
      axs.set_ylim((valmin-buffer, valmax+buffer))
      axs.tick_params(axis='both', which='major', labelsize=16) 

      props = dict(boxstyle='round', facecolor='white', alpha=0.5)
      textstr = f'RMSE: {rmse:.3f}\nnRMSE: {nrmse:.3f}\n$R^2$: {r2:.3f}'
      axs.text(0.03, 0.8, textstr, transform=axs.transAxes, fontsize=16, bbox=props)

      # Plot y=x line
      axs.plot([0, 10], [0, 10], color='gray', linestyle='--')

      # Hide the top and right spine
      axs.spines["right"].set_visible(False)
      axs.spines["top"].set_visible(False)

      if 'clean' in f:
        plt.savefig(os.path.join(out_dir, f'val_{field_name.replace(" ", "")}_clean.png'))
      else:
        plt.savefig(os.path.join(out_dir, f'val_{field_name.replace(" ", "")}.png'))
      

else:

  all_ytrue = []
  all_ypred = []
  field_data = []

  # Combine the results of all the fields and compute global scores
  xls = pd.ExcelFile(results_file)
  for sheet_name in xls.sheet_names:
      df = pd.read_excel(xls, sheet_name=sheet_name)
      all_ytrue.append(df['lai'])
      all_ypred.append(df['snap_LAI'])
      field_data.append((sheet_name, df['lai'], df['snap_LAI']))
  
  all_ytrue = np.concatenate(all_ytrue)
  all_ypred = np.concatenate(all_ypred)

  # Compute scores
  rmse = root_mean_squared_error(all_ytrue, all_ypred)
  nrmse = rmse/(all_ytrue.max() - all_ytrue.min())
  r2 = r2_score(all_ytrue, all_ypred)

  #############
  # PLOT RESULTS OF ALL FIELDS TOGETHER
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
  y = all_ypred
  y_true = all_ytrue

  scatter = axs.scatter(y_true, y)
  axs.set_xlabel('Measured LAI', size=18)
  axs.set_ylabel('SNAP LAI', size=18)
  valmin = min(y_true.min(), y.min())
  valmax = max(y_true.max(), y.max())
  buffer = (valmax-valmin)*0.1
  axs.set_xlim((valmin-buffer, valmax+buffer))
  axs.set_ylim((valmin-buffer, valmax+buffer))
  axs.tick_params(axis='both', which='major', labelsize=16) 

  props = dict(boxstyle='round', facecolor='white', alpha=0.5)
  textstr = f'RMSE: {rmse:.3f}\nnRMSE: {nrmse:.3f}\n$R^2$: {r2:.3f}'
  axs.text(0.03, 0.83, textstr, transform=axs.transAxes, fontsize=16, bbox=props)

  # Plot y=x line
  axs.plot([0, 10], [0, 10], color='gray', linestyle='--')

  # Hide the top and right spine
  axs.spines["right"].set_visible(False)
  axs.spines["top"].set_visible(False)

  plt.savefig(os.path.join(out_dir, 'val_allfields.png'))

  #############
  # PLOT RESULTS, COLOR BY FIELD
  df = pd.DataFrame([
      {"field_name": field_name, "y_true": yt, "y_pred": yp}
      for field_name, y_true, y_pred in field_data
      for yt, yp in zip(y_true, y_pred)
  ])

  plt.figure(figsize=(6, 6))
  sns.scatterplot(data=df, x="y_true", y="y_pred", palette='tab10', hue="field_name", alpha=0.7)
  if country=='italy':
    plt.plot([0, 8],
            [0, 8],
            'k--', label='1:1 line')
  else:
    plt.plot([df["y_true"].min(), df["y_true"].max()],
            [df["y_true"].min(), df["y_true"].max()],
            'k--', label='1:1 line')
  plt.legend()
  plt.title(f"SNAP baseline - {country.title()}", fontsize=18)
  plt.xlabel('Measured LAI [m2/m2]', fontsize=18)
  plt.ylabel('Predicted LAI [m2/m2]', fontsize=18)
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(axis='both', which='major', labelsize=16) 
  plt.savefig(os.path.join(out_dir, 'val_allfields_colored.png'))


  #############
  # PLOT RESULTS, COLOR BY SOIL GROUP
  df = pd.DataFrame([
      {"field_name": field_name, "y_true": yt, "y_pred": yp}
      for field_name, y_true, y_pred in field_data
      for yt, yp in zip(y_true, y_pred)
  ])
  field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
  field_soil_group = field_soil_group[field_soil_group['country']==country]
  df = pd.merge(df, field_soil_group, on='field_name')

  colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
  cluster_labels = np.arange(1,6)
  cluster_color_map = dict(zip(cluster_labels, colors))
  df['color'] = df['soil_group'].map(cluster_color_map)

  plt.figure(figsize=(6, 6))
  sns.scatterplot(data=df, x="y_true", y="y_pred", hue="soil_group", palette=cluster_color_map)
  if country=='italy':
    plt.plot([0, 8],
          [0, 8],
          'k--', label='1:1 line')
  else:
    plt.plot([df["y_true"].min(), df["y_true"].max()],
          [df["y_true"].min(), df["y_true"].max()],
          'k--', label='1:1 line')
  plt.legend(title="Soil Group")
  plt.title(f"SNAP baseline - {country.title()}", fontsize=18)
  plt.xlabel('Measured LAI [m2/m2]', fontsize=18)
  plt.ylabel('Predicted LAI [m2/m2]', fontsize=18)
  ax = plt.gca()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(axis='both', which='major', labelsize=16) 
  plt.savefig(os.path.join(out_dir, 'val_allfields_soilgroup.png'))


  

# Plot for all countries at once

if len(all_countries):

  all_ytrue = []
  all_ypred = []

  for country in all_countries:
    results_file = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/snap_baseline/{country}/snap_predictions_clean.xlsx')

    # Combine the results of all the fields and compute global scores
    xls = pd.ExcelFile(results_file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        all_ytrue.append(df['lai'])
        all_ypred.append(df['snap_LAI'])
    
  all_ytrue = np.concatenate(all_ytrue)
  all_ypred = np.concatenate(all_ypred)

  # Compute scores
  rmse = root_mean_squared_error(all_ytrue, all_ypred)
  nrmse = rmse/(all_ytrue.max() - all_ytrue.min())
  r2 = r2_score(all_ytrue, all_ypred)

  # Plot results for a field
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
  y = all_ypred
  y_true = all_ytrue

  scatter = axs.scatter(y_true, y)
  axs.set_xlabel('Measured LAI', size=18)
  axs.set_ylabel('SNAP LAI', size=18)
  valmin = min(y_true.min(), y.min())
  valmax = max(y_true.max(), y.max())
  buffer = (valmax-valmin)*0.1
  axs.set_xlim((valmin-buffer, valmax+buffer))
  axs.set_ylim((valmin-buffer, valmax+buffer))
  axs.tick_params(axis='both', which='major', labelsize=16) 

  props = dict(boxstyle='round', facecolor='white', alpha=0.5)
  textstr = f'RMSE: {rmse:.3f}\nnRMSE: {nrmse:.3f}\n$R^2$: {r2:.3f}'
  axs.text(0.03, 0.8, textstr, transform=axs.transAxes, fontsize=16, bbox=props)

  # Plot y=x line
  axs.plot([0, 10], [0, 10], color='gray', linestyle='--')

  # Hide the top and right spine
  axs.spines["right"].set_visible(False)
  axs.spines["top"].set_visible(False)

  parent_dir = os.path.dirname(out_dir)
  plt.savefig(os.path.join(parent_dir, 'val_allfields_clean.png'))

