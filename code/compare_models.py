"""
Compare performance of LAI predictions with field-level models, multi-field, large scale models, as well as baselines

Sélène Ledain
12 Nov 2025
"""

import os
import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore") 
import os
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# ---- Simple MLP ----
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def build_model(config: dict):
  ''' 
  Instantiated model

  :param config: dictionary of configuration parameters
  :returns: model
  '''
  model_name = config['Model']['name']
  if model_name not in MODELS:
    raise ValueError(f"Invalid model type: {model_name}")
  else:
    #if model_name == 'NN':
      #torch.manual_seed(config['Seed'])
    # Model hypereparameters can be set in the config, else default values used
    model_params = {key: value for key, value in config['Model'].items() if key != 'name'}  # Pass only hyperparams
    model_params['random_state'] = config['Seed']
    model = MODELS[model_name](**model_params)
  
  return model


def prepare_data_test(data_path, model_name):

    df = pd.read_pickle(data_path)
    df = df[~df['lai'].isna()]
    X = df[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']]
    y = df['lai']

    # Load scaler
    scaler_path = model_name.split('.pkl')[0] + '_scaler.pkl'
    with open(scaler_path, 'rb') as f:
      scaler = pickle.load(f)
    # Normalize
    X = scaler.transform(X)
    return X, y.values


def test_model(model_basename, data_path):

    y_pred_seed = []
    for seed in range(5):

      model_name = model_basename.split('.pkl')[0] + f'{seed}.pkl'

      #####################
      # DATA
      X_test, y_test = prepare_data_test(data_path=data_path, model_name=model_name)

      #####################
      # MODEL
      # Move data to CUDA
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      if device == torch.device('cuda'):
          X_test, y_test = (
              torch.FloatTensor(X_test).to(device),
              torch.FloatTensor(y_test).view(-1, 1).to(device)
          )

      #model_name = "NN"
      try:
        config_path = '../configs/config_NN_clean.yaml'
        with open(config_path, "r") as config_file:
          config = yaml.safe_load(config_file)
        model = build_model(config=config)
        state_dict = torch.load(model_name, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
      except:       
        with open(model_name, 'rb') as f:
          model = pickle.load(f)

        # Move model to CUDA if GPUs are available
        if device == torch.device('cuda'):
            model.to(device)
      
      #############################################
      # TEST
      try:
        y_pred = model.predict(X_test=X_test)
      except:
         with torch.no_grad():
            y_pred = model(X_test).cpu().numpy()

      if not np.isnan(y_pred.flatten()).any():
        y_pred_seed.append(y_pred)
  
    y_pred = np.mean(y_pred_seed, axis=0)

    return y_test, y_pred



#################
# GET VALIDATION PREDS OF SINGLE FIELD-MODEL
"""
country = 'switzerland'
field_to_plot = 'Ruetteli'

val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
if country == 'switzerland':
  field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
  country_code = 'CH'
elif country == 'bulgaria':
  country_code = 'BG'
elif country == 'italy':
  country_code = 'IT'


field_preds = []
for field in field_valdata:

    if field_to_plot not in field:
      continue
      
    # Save the preds/ground-truth
    if 'clean' in field and country=='switzerland':
      field_name = field.split('_')[-2].split('.pkl')[0]
    else:
      field_name = field.split('_')[-1].split('.pkl')[0]
    data_path = os.path.join(val_data_dir, field)

    model_basename = f'../models/NN_soil_{country_code}_{field_name}_tuned.pkl' 

    y_test, y_pred = test_model(model_basename, data_path)
    
    # Get RMSE of that model
    rmse = root_mean_squared_error(y_test.cpu().detach().numpy(), y_pred)
    print(field, rmse)
"""

#################
# PLOT VALIDATION PREDS OF ALL FIELD-MODELS ON ONE PLOT (PER COUNTRY)
"""
countries = ['switzerland', 'bulgaria', 'italy']
for country in countries:

    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        
        # Run test for each field and save the preds/ground-truth
        if 'clean' in field and country=='switzerland':
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        model_basename = f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl' 

        y_test, y_pred = test_model(model_basename, data_path)
        
        # Get RMSE of that model
        rmse = root_mean_squared_error(y_test.cpu().detach().numpy(), y_pred)
        print(field, rmse)
        
        # Save preds and GT
        field_preds.append((field_name, y_test.cpu().detach().numpy(), y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"field_name": field_name, "y_test": yt[0], "y_pred": yp[0]}
        for field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])


    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="y_test", y="y_pred", hue="field_name", palette='tab10', alpha=0.7)
    plt.plot([0,8],
            [0,8],
            'k--', label='1:1 line')
    #plt.plot([df["y_test"].min(), df["y_test"].max()],
    #         [df["y_test"].min(), df["y_test"].max()],
    #         'k--', label='1:1 line')
    plt.legend()
    plt.title(f"Field models - {country.title()}", fontsize=18)
    plt.xlabel('Measured LAI [m2/m2]', fontsize=18)
    plt.ylabel('Predicted LAI [m2/m2]', fontsize=18)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16) 
    plt.savefig(f'../model_results/{country}_fields/field_models_{country}.png')
"""

#################
# PLOT VALIDATION PREDS OF MULTI-FIELD MODEL ON ONE PLOT (PER COUNTRY)
"""
countries = ['italy'] #'switzerland', 'bulgaria', ]
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'

    model_basename = f'../models/NN_multifield_soil_tuned.pkl' 
    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth

        if 'clean' in field and country=='switzerland':
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        y_test, y_pred = test_model(model_basename, data_path)
        
        # Save preds and GT
        field_preds.append((field_name, y_test.cpu().detach().numpy(), y_pred))


    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"field_name": field_name, "y_test": yt[0], "y_pred": yp[0]}
        for field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    # Compute RMSE and R2 per field and save results
    field_rmses = []
    fields_r2s = []
    for i, df_gb in df.groupby('field_name'):
      rmse =  root_mean_squared_error(df_gb['y_test'], df_gb['y_pred'])
      field_rmses.append(rmse)
      if country!='italy':
        r2 = r2_score(df_gb['y_test'], df_gb['y_pred'])
        fields_r2s.append(r2)
      print(i, rmse)
      
    if not len(fields_r2s):
      fields_r2s = [np.nan]*len(field_rmses)
    field_results = pd.DataFrame({"field_name": df.groupby('field_name').indices.keys(), "RMSE":field_rmses, "R2": fields_r2s})
    field_results.to_excel(f'../model_results/{country}_fields/NN_multifield_soil_{country}.xlsx', index=False)

    # Global score
    rmse = root_mean_squared_error(df['y_test'], df['y_pred'])
    r2 = r2_score(df['y_test'], df['y_pred'])
    textstr = f'RMSE: {rmse:.3}\n$R^2$: {r2:.3f}'

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="y_test", y="y_pred", hue="field_name", palette='tab10', alpha=0.7)
    plt.plot([0,10],
            [0,10],
            'k--', label='1:1 line')
    plt.legend()
    plt.title(f"Multi-field model - {country.title()}", fontsize=18)
    plt.xlabel('Measured LAI [m2/m2]', fontsize=18)
    plt.ylabel('Predicted LAI [m2/m2]', fontsize=18)
    plt.xlim(0,10)
    plt.ylim(0,10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16) 

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.73, 0.8, textstr, transform=ax.transAxes, fontsize=16, bbox=props)

    plt.savefig(f'../model_results/{country}_fields/multifield_model_{country}.png')
"""

#################
# PLOT VALIDATION PREDS OF LARGE SCALE MODEL ON ONE PLOT (PER COUNTRY)
"""
countries = ['italy'] #'switzerland', 'bulgaria', ]
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'

    model_basename = '../models/NN_europe_soil_tuned.pkl' 
    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth

        if 'clean' in field and country=='switzerland':
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        y_test, y_pred = test_model(model_basename, data_path)
        
        # Save preds and GT
        field_preds.append((field_name, y_test.cpu().detach().numpy(), y_pred))


    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"field_name": field_name, "y_test": yt[0], "y_pred": yp[0]}
        for field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    # Compute RMSE and R2 per field and save results
    field_rmses = []
    fields_r2s = []
    for i, df_gb in df.groupby('field_name'):
      rmse =  root_mean_squared_error(df_gb['y_test'], df_gb['y_pred'])
      field_rmses.append(rmse)
      if country!='italy':
        r2 = r2_score(df_gb['y_test'], df_gb['y_pred'])
        fields_r2s.append(r2)
      print(i, rmse)
      
    if not len(fields_r2s):
      fields_r2s = [np.nan]*len(field_rmses)
    field_results = pd.DataFrame({"field_name": df.groupby('field_name').indices.keys(), "RMSE":field_rmses, "R2": fields_r2s})
    field_results.to_excel(f'../model_results/{country}_fields/NN_europe_soil_{country}.xlsx', index=False)

    # Global score
    rmse = root_mean_squared_error(df['y_test'], df['y_pred'])
    r2 = r2_score(df['y_test'], df['y_pred'])
    textstr = f'RMSE: {rmse:.3}\n$R^2$: {r2:.3f}'

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="y_test", y="y_pred", hue="field_name", palette='tab10', alpha=0.7)
    plt.plot([0,10],
            [0,10],
            'k--', label='1:1 line')
    plt.legend()
    plt.title(f"Large-scale model - {country.title()}", fontsize=18)
    plt.xlabel('Measured LAI [m2/m2]', fontsize=18)
    plt.ylabel('Predicted LAI [m2/m2]', fontsize=18)
    plt.xlim(0,10)
    plt.ylim(0,10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16) 

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.73, 0.8, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
    plt.savefig(f'../model_results/{country}_fields/europe_model_{country}.png')
"""

#################
# PLOT VALIDATION PREDS OF NOSOIL MODEL ON ONE PLOT (PER COUNTRY)
"""
countries = ['italy'] #, 'bulgaria', 'italy']
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'

    model_basename = f'../models/NN_europe_nosoil_tuned.pkl' 
    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        
        if 'clean' in field and country=='switzerland':
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        y_test, y_pred = test_model(model_basename, data_path)
        
        # Save preds and GT
        field_preds.append((field_name, y_test.cpu().detach().numpy(), y_pred))


    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"field_name": field_name, "y_test": yt[0], "y_pred": yp[0]}
        for field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    # Compute RMSE and R2 per field and save results
    field_rmses = []
    fields_r2s = []
    for i, df_gb in df.groupby('field_name'):
      rmse =  root_mean_squared_error(df_gb['y_test'], df_gb['y_pred'])
      field_rmses.append(rmse)
      if country!='italy':
        r2 = r2_score(df_gb['y_test'], df_gb['y_pred'])
        fields_r2s.append(r2)
      print(i, rmse)
      
    if not len(fields_r2s):
      fields_r2s = [np.nan]*len(field_rmses)
    field_results = pd.DataFrame({"field_name": df.groupby('field_name').indices.keys(), "RMSE":field_rmses, "R2": fields_r2s})
    field_results.to_excel(f'../model_results/{country}_fields/NN_nosoil_europe_{country}.xlsx', index=False)

    # Global score
    rmse = root_mean_squared_error(df['y_test'], df['y_pred'])
    r2 = r2_score(df['y_test'], df['y_pred'])
    textstr = f'RMSE: {rmse:.3}\n$R^2$: {r2:.3f}'

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="y_test", y="y_pred", hue="field_name", palette='tab10', alpha=0.7)
    plt.plot([0,10],
            [0,10],
            'k--', label='1:1 line')
    plt.legend()
    plt.title(f"No-soil model - {country.title()}", fontsize=18)
    plt.xlabel('Measured LAI [m2/m2]', fontsize=18)
    plt.ylabel('Predicted LAI [m2/m2]', fontsize=18)
    plt.xlim(0,10)
    plt.ylim(0,10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16) 

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.73, 0.8, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
    plt.savefig(f'../model_results/{country}_fields/europe_nosoil_model_{country}.png')
"""

#################
# PLOT VALIDATION OF DIFFERENT MODELS (PER COUNTRY)
"""
countries = ['switzerland', 'bulgaria', 'italy']

for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_soil_{country_code}_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions{"_clean" if country=="switzerland" else ""}.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0]}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    n_models = len(df["model"].unique())
    model_types = df["model"].unique()
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

    if n_models == 1:
        axes = [axes]  

    for ax, model_type in zip(axes, model_types):
        df_model = df[df.model==model_type]
        
        sns.scatterplot(ax=ax, data=df_model, x="y_test", y="y_pred", hue="field_name", palette='tab10')
        ax.plot([0, 10], [0, 10], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
        ax.set_title(f"{model_type} model", fontsize=22)
        ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)
        ax.get_legend().remove()

        rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
        r2 = r2_score(df_model['y_test'], df_model['y_pred'])
        textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.8, textstr, transform=ax.transAxes, fontsize=16, bbox=props)


    axes[0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
    axes[-1].legend(title='Field', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=16, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(f'../model_results/{country}_fields/field_allmodels_{country}.png')
"""


#################
# PLOT VALIDATION OF DIFFERENT MODELS (ALL COUNTRIES TOGHETHER)
"""
countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

if n_models == 1:
    axes = [axes]  

for ax, model_type in zip(axes, model_types):
    df_model = df_all[df_all.model==model_type]
    
    sns.scatterplot(ax=ax, data=df_model, x="y_test", y="y_pred", palette='tab10') #hue="country", 
    ax.plot([0, 10], [0, 10], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
    ax.set_title(f"{model_type} model", fontsize=22)
    ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=20)
    #ax.get_legend().remove()

    rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
    r2 = r2_score(df_model['y_test'], df_model['y_pred'])
    nrmse = rmse *100/ (df_model['y_test'].max() - df_model['y_test'].min())
    textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.78, textstr, transform=ax.transAxes, fontsize=16, bbox=props)


axes[0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
#axes[-1].legend(title='Country', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=16, title_fontsize=18)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels.png')


# Plot for LAI 0-1 range
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

if n_models == 1:
    axes = [axes]  

for ax, model_type in zip(axes, model_types):
    df_model = df_all[df_all.model==model_type]
    df_model = df_model[df_model['y_test']<=1]
    
    sns.scatterplot(ax=ax, data=df_model, x="y_test", y="y_pred", palette='tab10') # hue="country", 
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
    ax.set_title(f"{model_type} model", fontsize=22)
    ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=20)
    #ax.get_legend().remove()

    rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
    r2 = r2_score(df_model['y_test'], df_model['y_pred'])
    nrmse = rmse *100/ (df_model['y_test'].max() - df_model['y_test'].min())
    textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.78, textstr, transform=ax.transAxes, fontsize=16, bbox=props)


axes[0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
#axes[-1].legend(title='Country', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=16, title_fontsize=18)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_LAI01.png')
"""

#################
# PLOT VALIDATION OF DIFFERENT MODELS (ALL COUNTRIES TOGHETHER), WITH SOIL GROUPS
"""
def pval_with_star(p):
    if p < 0.001:
        star = '***'
        return f"p = {p:.3f} ({star})"
    elif p < 0.01:
        star = '**'
        return f"p = {p:.3f} ({star})"
    elif p < 0.05:
        star = '*'
        return f"p = {p:.3f} ({star})"
    else:
        star = ''
        return f"p = {p:.3f}"



countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# Add soil information 
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
field_soil_group['field_name'] = field_soil_group['field_name'].str.replace(' ', '', regex=False)
df_all = pd.merge(df_all, field_soil_group, on=['country', 'field_name'], how='left')
colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))
# Keep only colors in map that are present in data
cluster_color_map = {
    k: v for k, v in cluster_color_map.items()
    if k in df_all["soil_group"].dropna().unique()
}

# Significance data
if os.path.exists(f'../model_results/model_vs_SNAP_significance_LAI0-15.csv'):
  sig = True
  sig_df = pd.read_csv(f'../model_results/model_vs_SNAP_significance_LAI0-15.csv')

df_all["soil_group"] = pd.Categorical(df_all["soil_group"], categories=cluster_labels, ordered=True)
n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

if n_models == 1:
    axes = [axes]  

for ax, model_type in zip(axes, model_types):
    df_model = df_all[df_all.model==model_type]
    df_model["soil_group"] = df_model["soil_group"].cat.remove_unused_categories()
    sns.scatterplot(ax=ax, data=df_model, x="y_test", y="y_pred", hue='soil_group', palette=cluster_color_map) #hue="country", 
    ax.plot([0, 10], [0, 10], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
    ax.set_title(f"{model_type} model", fontsize=22)
    ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=20)
    ax.get_legend().remove()

    rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
    mae = mean_absolute_error(df_model['y_test'], df_model['y_pred'])
    r2 = r2_score(df_model['y_test'], df_model['y_pred'])
    nrmse = rmse*100/(df_model['y_test'].mean()) #rmse*100/(df_model['y_test'].max() - df_model['y_test'].min())
    textstr = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
    y_box = 0.73 #0.78
    if sig and model_type!='SNAP':
        p_val = sig_df.loc[sig_df.model == model_type, 'p_value'].values[0]
        text = pval_with_star(p_val)
        textstr += f'\n{text}'
        y_box = 0.66 #0.7
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, y_box, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
    #ax.text(0.6, y_box, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
  

axes[0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
axes[-1].legend(title='Soil group', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=16, title_fontsize=18)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_soilgroup_significance_mae.png')
"""

#################
# PLOT VALIDATION OF DIFFERENT MODELS (ALL COUNTRIES TOGHETHER), WITH SOIL GROUPS + NRMSE BARPLOTS
"""
countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# Add soil information 
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
df_all = pd.merge(df_all, field_soil_group, on=['country', 'field_name'], how='left')

colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))
df_all["soil_group"] = pd.Categorical(df_all["soil_group"], categories=cluster_labels, ordered=True)

# Compute NRMSE per LAI bins
step = 2
lai_bins = np.arange(0, 11, step)
lai_labels = [f"{i}-{i+step}" for i in lai_bins[:-1]]
df_all["lai_bin"] = pd.cut(df_all["y_test"], bins=lai_bins, labels=lai_labels, include_lowest=True)
df_all["lai_bin"] = pd.Categorical(df_all["lai_bin"], categories=lai_labels, ordered=True)
def nrmse(y_true, y_pred):
    if len(y_true) <= 1:  # Check if there is only one or no data point
        return np.nan  # Return NaN or another placeholder value
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse * 100 / (np.max(y_true) - np.min(y_true))
nrmse_df = (
    df_all
    .groupby(["model", "lai_bin"])
    .apply(lambda x: nrmse(x["y_test"], x["y_pred"]))
    .reset_index(name="nrmse")
)

# Count samples per model and LAI bin
count_df = df_all.groupby(["model", "lai_bin"]).size().reset_index(name="n")
nrmse_df = pd.merge(nrmse_df, count_df, on=["model", "lai_bin"], how="left")
  

n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()
fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 5*2), sharey="row")

if n_models == 1:
    axes = np.array([[axes[0]], [axes[1]]])
for i, model_type in enumerate(model_types):

    df_model = df_all[df_all.model == model_type]
    df_nrmse_model = nrmse_df[nrmse_df.model == model_type]
    print(df_nrmse_model)
    # ----------------------
    # TOP: Scatter
    # ----------------------
    ax_scatter = axes[0, i]

    sns.scatterplot(ax=ax_scatter, data=df_model, x="y_test", y="y_pred", hue="soil_group", palette=cluster_color_map, hue_order=cluster_labels)
    ax_scatter.plot([0, 10], [0, 10], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
    ax_scatter.set_title(f"{model_type} model", fontsize=22)
    ax_scatter.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
    ax_scatter.set_xlim(0, 10)
    ax_scatter.set_ylim(0, 10)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.tick_params(axis='both', labelsize=20)
    ax_scatter.get_legend().remove()

    rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
    r2 = r2_score(df_model['y_test'], df_model['y_pred'])
    nrmse = rmse*100/(df_model['y_test'].max() - df_model['y_test'].min())
    textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax_scatter.text(0.05, 0.78, textstr, transform=ax_scatter.transAxes, fontsize=16, bbox=props)

    # ----------------------
    # BOTTOM: NRMSE barplot
    # ----------------------
    ax_bar = axes[1, i]

    sns.barplot(ax=ax_bar, data=df_nrmse_model, x="lai_bin", y="nrmse", color="lightslategrey")
    ax_bar.set_xlabel("LAI range", fontsize=20)
    ax_bar.tick_params(axis='y', labelsize=20)
    ax_bar.tick_params(axis='x', labelsize=13)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # ---- Add sample size above bars ----
    for j, bar in enumerate(ax_bar.patches):
        height = bar.get_height()
        n_value = df_nrmse_model.iloc[j]["n"]
        
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,   # small offset above bar
            f"n={n_value}",
            ha='center',
            va='bottom',
            fontsize=11
        )


axes[0, 0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
axes[1, 0].set_ylabel("NRMSE [%]", fontsize=20)
axes[0, -1].legend(title="Soil group", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16, title_fontsize=18)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_soilgroup_nrmse.png')
"""


#################
# DIFFERENT MODELS GROUPED NRMSE BARPLOTS
"""
countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# Compute NRMSE per LAI bins
lai_bins = list(np.arange(0, 7, 1)) + [np.inf]
lai_labels = [f"{i}-{i+1}" for i in range(0, 6)] + [">6"]
df_all["lai_bin"] = pd.cut(df_all["y_test"], bins=lai_bins, labels=lai_labels, include_lowest=True)
df_all["lai_bin"] = pd.Categorical(df_all["lai_bin"], categories=lai_labels, ordered=True)
def nrmse(y_true, y_pred):
    if len(y_true) <= 1:  # Check if there is only one or no data point
        return np.nan  # Return NaN or another placeholder value
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse * 100 / np.mean(y_true)#(np.max(y_true) - np.min(y_true))
nrmse_df = (
    df_all
    .groupby(["model", "lai_bin"])
    .apply(lambda x: nrmse(x["y_test"], x["y_pred"]))
    .reset_index(name="nrmse")
)

# Count samples per model and LAI bin
count_df = df_all.groupby(["model", "lai_bin"]).size().reset_index(name="n")
nrmse_df = pd.merge(nrmse_df, count_df, on=["model", "lai_bin"], how="left")

plt.figure(figsize=(20, 5))

model_order = ['Field', 'Multi-field', 'Large-scale', 'No-soil', 'SNAP']  # desired model order
ax = sns.barplot(
    data=nrmse_df,
    x="lai_bin",
    y="nrmse",
    hue="model",
    hue_order=model_order,
    palette=sns.color_palette("Blues", n_colors=len(df_all['model'].unique()))
)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("LAI range", fontsize=18)
plt.ylabel("NRMSE [%]", fontsize=18)
ax.tick_params(axis='y', labelsize=16)  # change 14 to any size you want
ax.legend(title="Model type", fontsize=14, title_fontsize=16, loc='upper right', bbox_to_anchor=(1, 1.2))
#plt.legend(title="Model type")

# Sort dataframe exactly like seaborn plotted it
model_order = nrmse_df["model"].unique()
lai_order = nrmse_df["lai_bin"].cat.categories

plot_df = (
    nrmse_df
    .sort_values(["lai_bin", "model"])
    .reset_index(drop=True)
)

# Add NRMSE to bars
for bar, (_, row) in zip(ax.patches, plot_df.iterrows()):
    height = bar.get_height()

    if not np.isnan(height):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.03,
            height + 0.5,
            f"{height:.1f}%", #\nn={int(row['n'])}",
            ha='center',
            va='bottom',
            fontsize=14,
            rotation=60
        )

# Add count per bins
n_per_bin = (
    nrmse_df
    .groupby("lai_bin")["n"]
    .first()  # or sum() if you want total across models
)
ax.set_xticks(range(len(lai_labels)))  # ensure tick positions
new_labels = [f"{label}\nn={n_per_bin[label]}" for label in lai_labels]
ax.set_xticklabels(new_labels, rotation=0, fontsize=18)

plt.tight_layout()
plt.savefig("../model_results/allmodels_grouped_nrmse_max6.png")
"""


#################
# PLOT VALIDATION OF DIFFERENT MODELS, FOR AN LAI RANGE (PER COUNTRY)

"""
def pval_with_star(p):
    if p < 0.001:
        star = '***'
        return f"p = {p:.3f} ({star})"
    elif p < 0.01:
        star = '**'
        return f"p = {p:.3f} ({star})"
    elif p < 0.05:
        star = '*'
        return f"p = {p:.3f} ({star})"
    else:
        star = ''
        return f"p = {p:.3f}"

        
countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)
maxlai = 4
df_all = df_all[df_all['y_test']<maxlai]

# Significance data
if os.path.exists(f'../model_results/model_vs_SNAP_significance_LAI0-{maxlai}.csv'):
  sig = True
  sig_df = pd.read_csv(f'../model_results/model_vs_SNAP_significance_LAI0-{maxlai}.csv')


# Add soil information 
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
field_soil_group['field_name'] = field_soil_group['field_name'].str.replace(' ', '', regex=False)
df_all = pd.merge(df_all, field_soil_group, on=['country', 'field_name'], how='left')
colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))
# Keep only colors in map that are present in data
cluster_color_map = {
    k: v for k, v in cluster_color_map.items()
    if k in df_all["soil_group"].dropna().unique()
}

df_all["soil_group"] = pd.Categorical(df_all["soil_group"], categories=cluster_labels, ordered=True)
n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

if n_models == 1:
    axes = [axes]  

for ax, model_type in zip(axes, model_types):
    df_model = df_all[df_all.model==model_type]
    df_model["soil_group"] = df_model["soil_group"].cat.remove_unused_categories()
    sns.scatterplot(ax=ax, data=df_model, x="y_test", y="y_pred", hue='soil_group', palette=cluster_color_map) #hue="country", 
    ax.plot([0, maxlai], [0, maxlai], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
    ax.set_title(f"{model_type} model", fontsize=22)
    ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
    ax.set_xlim(0, maxlai)
    ax.set_ylim(0, maxlai)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=20)
    ax.get_legend().remove()

    rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
    mae = mean_absolute_error(df_model['y_test'], df_model['y_pred'])
    r2 = r2_score(df_model['y_test'], df_model['y_pred'])
    nrmse = rmse*100/(df_model['y_test'].mean()) #rmse*100/(df_model['y_test'].max() - df_model['y_test'].min())
    textstr = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
    y_box = 0.05 #0.78
    if sig and model_type!='SNAP':
        p_val = sig_df.loc[sig_df.model == model_type, 'p_value'].values[0]
        text = pval_with_star(p_val)
        textstr += f'\n{text}'
        y_box = 0.05 #0.7
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    #ax.text(0.05, y_box, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
    ax.text(0.6, y_box, textstr, transform=ax.transAxes, fontsize=16, bbox=props)


axes[0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
axes[-1].legend(title='Soil group', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=16, title_fontsize=18)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_soilgroup_cutoff{maxlai}_significance_mae.png')
"""

#################
# PLOT VALIDATION OF DIFFERENT MODELS, STRATIFIED FOR LAI SHOWING SOIL GROUPS
"""
lai_ranges = [(0,1), (1, 3), (3, 6), (6, 12)]
lai_labels = ["LAI<1", "LAI 1-3", "LAI 3-6", "LAI >6"]

countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# Add soil information 
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
df_all = pd.merge(df_all, field_soil_group, on=['country', 'field_name'], how='left')

colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))
df_all["soil_group"] = pd.Categorical(df_all["soil_group"], categories=cluster_labels, ordered=True)

n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()

# Create one figure for all rows (LAI ranges)
fig, axes = plt.subplots(len(lai_ranges), n_models, figsize=(5 * n_models, 5 * len(lai_ranges))) #, sharey=True, sharex=True)
if len(lai_ranges) == 1:
    axes = [axes]
if n_models == 1:
    axes = [[ax] for ax in axes]  # make 2D for consistency

for i, (lai_range, label) in enumerate(zip(lai_ranges, lai_labels)):
    df_range = df_all[(df_all['y_test'] > lai_range[0]) & (df_all['y_test'] <= lai_range[1])]
    if len(df_range) == 0:
        # hide empty row
        for ax in axes[i]:
            ax.axis('off')
        continue

    for j, model_type in enumerate(model_types):
        ax = axes[i][j]
        df_model = df_range[df_range['model'] == model_type]
        if len(df_model) == 0:
            ax.axis('off')
            continue
        sns.scatterplot(ax=ax, data=df_model, x='y_test', y='y_pred', hue='soil_group', palette=cluster_color_map)
        ax.plot([0, lai_range[1]], [0, lai_range[1]], 'k--', lw=1)  # 1:1 line

        if i == 0:
          ax.set_title(f"{model_type} model", fontsize=22, pad=20)
        if j == 0:
            ax.set_ylabel(f"{label}\nPredicted LAI [m²/m²]", fontsize=20)
        else:
            ax.set_ylabel('')
        if i == len(lai_ranges)-1:
            ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
        else:
            ax.set_xlabel('')
"""
"""
        if lai_range == (0,2):
          ax.set_xlim(lai_range[0], lai_range[1]+1)
          ax.set_ylim(lai_range[0], lai_range[1]+1)
        if lai_range == (2,6):
          ax.set_xlim(lai_range[0]-1, lai_range[1])
          ax.set_ylim(lai_range[0]-1, lai_range[1])
        if lai_range == (6,12):
          ax.set_xlim(lai_range[0]-2, lai_range[1]-2)
          ax.set_ylim(lai_range[0]-2, lai_range[1]-2)
"""
"""
        if lai_range == (0,1):
          ax.set_xlim(lai_range[0], lai_range[1])
          ax.set_ylim(lai_range[0], lai_range[1])
        if lai_range == (1,3):
          ax.set_xlim(lai_range[0], lai_range[1])
          ax.set_ylim(lai_range[0], lai_range[1])
        if lai_range == (3,6):
          ax.set_xlim(lai_range[0]-1, lai_range[1])
          ax.set_ylim(lai_range[0]-1, lai_range[1])
        if lai_range == (6,12):
          ax.set_xlim(lai_range[0]-2, lai_range[1]-2)
          ax.set_ylim(lai_range[0]-2, lai_range[1]-2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)
        ax.get_legend().remove()
        
        # Metrics
        rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
        r2 = r2_score(df_model['y_test'], df_model['y_pred'])
        nrmse = rmse*100/(df_model['y_test'].max() - df_model['y_test'].min())
        textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.78, textstr, transform=ax.transAxes, fontsize=16, bbox=props)
 
# Single legend for entire figure
axes[0, -1].legend(title="Soil group", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16, title_fontsize=18)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_splitLAI_soilgroups.png', dpi=300)
"""


#################
# PLOT VALIDATION OF DIFFERENT MODELS, PER SOIL GROUP + NRMSE GROUPED BARPLOTS
"""
countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# Add soil information 
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
df_all = pd.merge(df_all, field_soil_group, on=['country', 'field_name'], how='left')

colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))
df_all["soil_group"] = pd.Categorical(df_all["soil_group"], categories=cluster_labels, ordered=True)

soil_groups = df_all["soil_group"].cat.categories
n_soil = len(soil_groups) 
n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()

fig, axes = plt.subplots(n_soil, n_models, figsize=(5 * n_models, 5 * n_soil), sharex=True, sharey=True)

for i, soil_group in enumerate(soil_groups):
    color = cluster_color_map[soil_group]
    for j, model_type in enumerate(model_types):
      ax_scatter = axes[i, j]

      # Filter data for this soil group and model
      df_plot = df_all[(df_all["soil_group"] == soil_group) & (df_all["model"] == model_type)]

      # ----------------------
      # TOP: Scatter
      # ----------------------

      sns.scatterplot(ax=ax_scatter, data=df_plot, x="y_test", y="y_pred", color=color, legend=False)
      ax_scatter.plot([0, 10], [0, 10], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
      ax_scatter.set_xlim(0, 10)
      ax_scatter.set_ylim(0, 10)
      ax_scatter.spines['top'].set_visible(False)
      ax_scatter.spines['right'].set_visible(False)
      ax_scatter.tick_params(axis='both', labelsize=20)

      # Titles and labels
      if i == 0:
          ax_scatter.set_title(f"{model_type} model", fontsize=20)
      if j == 0:
          ax_scatter.set_ylabel(f"Soil group {soil_group}\nPredicted LAI [m²/m²]", fontsize=20)
      if i == n_soil - 1:
          ax_scatter.set_xlabel("Measured LAI [m²/m²]", fontsize=20)

      rmse = root_mean_squared_error(df_plot['y_test'], df_plot['y_pred'])
      r2 = r2_score(df_plot['y_test'], df_plot['y_pred'])
      nrmse = rmse*100/(df_plot['y_test'].max() - df_plot['y_test'].min())
      # If RMSE is nan (only 1 data point), don't put metrics
      if np.isnan(r2):
          textstr = f'RMSE: {rmse:.3f}\n$R^2$: -\nNRMSE: -'
      else:
        textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\nNRMSE: {nrmse:.1f}%'
      props = dict(boxstyle='round', facecolor='white', alpha=0.5)
      ax_scatter.text(0.05, 0.78, textstr, transform=ax_scatter.transAxes, fontsize=16, bbox=props)

  
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_per_soil.png')
"""


####################
# GROUPED NRMSE BARPLOTS SHOWING THE NRMSE PER SOIL GROUP AND BINNED LAI
"""
countries = ['switzerland', 'bulgaria', 'italy']
df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0], "country": country}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# Add soil information 
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
df_all = pd.merge(df_all, field_soil_group, on=['country', 'field_name'], how='left')

colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))
df_all["soil_group"] = pd.Categorical(df_all["soil_group"], categories=cluster_labels, ordered=True)

n_models = len(df_all["model"].unique())
model_types = df_all["model"].unique()

# Compute NRMSE per model, LAI bin, and soil group
step = 2
lai_bins = np.arange(0, 11, step)
lai_labels = [f"{i}-{i+step}" for i in lai_bins[:-1]]
df_all["lai_bin"] = pd.cut(df_all["y_test"], bins=lai_bins, labels=lai_labels, include_lowest=True)
df_all["lai_bin"] = pd.Categorical(df_all["lai_bin"], categories=lai_labels, ordered=True)
def nrmse(y_true, y_pred):
    if len(y_true) <= 1:  # Check if there is only one or no data point
        return np.nan  # Return NaN or another placeholder value
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse * 100 / (np.max(y_true) - np.min(y_true))

nrmse_df = (
    df_all
    .groupby(["model", "lai_bin", "soil_group"])
    .apply(lambda x: nrmse(x["y_test"], x["y_pred"]))
    .reset_index(name="nrmse")
)

count_df = df_all.groupby(["model", "lai_bin", "soil_group"]).size().reset_index(name="n")
nrmse_df = pd.merge(nrmse_df, count_df, on=["model", "lai_bin", "soil_group"], how="left")

fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), sharey=True)

for i, model_type in enumerate(model_types):
    ax = axes[i]
    
    df_model = nrmse_df[nrmse_df.model == model_type]

    sns.barplot(
        ax=ax,
        data=df_model,
        x="lai_bin",
        y="nrmse",
        hue="soil_group",
        palette=cluster_color_map,
        hue_order=cluster_labels
    )

    ax.set_title(f"{model_type} model", fontsize=20)
    ax.set_xlabel("LAI range", fontsize=18)
    ax.set_ylabel("NRMSE [%]", fontsize=18)

    ax.tick_params(axis='x', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_legend().remove()

plt.legend(title="Soil group", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig(f'../model_results/allmodels_nrmse_groupedsoil.png')
"""

####################
# STATISTICAL COMPARISON OF MODELS
"""
countries = ['switzerland', 'bulgaria', 'italy']
lai_ranges = [(0,1), (0,4), (0,15)]

df_all = []
for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')
    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]

    if country == 'switzerland':
        field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
        country_code = 'CH'
    elif country == 'bulgaria':
        country_code = 'BG'
    elif country == 'italy':
        country_code = 'IT'

    field_preds = []
    for field in field_valdata:
        # Determine field_name
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
            field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))
            
        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten into dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": f"{country}_{field_name}", "y_test": yt[0], "y_pred": yp[0]}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df_all.append(df)


for r in lai_ranges:
    print('LAI in range', r)
    # Pairwise Wilcoxon test between model and SNAP
    df = pd.concat(df_all, ignore_index=True)

    df_r = df[(df['y_test'] >r[0]) &(df['y_test'] <=r[1])]   
    if len(df_r) == 0:  
        continue

    df_r['se'] = np.abs((df_r['y_test'] - df_r['y_pred'])) #**2
    df_wide = df_r.pivot_table(
        index=['field_name'],
        columns='model',
        values='se'
    )

    baseline = df_wide['SNAP']
    models = ['Field', 'Multi-field', 'Large-scale', 'No-soil']

    results = []
    alpha = 0.05 #s/ len(models)  # Bonferroni correction
    for m in models:
        stat, p = wilcoxon(df_wide[m], baseline)
        results.append({
            "model": m,
            "p_value": p,
            "significant": p < alpha
        })
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(f'../model_results/model_vs_SNAP_significance_LAI{r[0]}-{r[1]}.csv', index=False)

"""


#################
# FIND HIGH ERROR POINTS, ANAYLSE SPECTRA (BINNED LAI)
"""
countries = ['switzerland', 'bulgaria', 'italy'] #

for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')
    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]

    if country == 'switzerland':
        field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
        country_code = 'CH'
    elif country == 'bulgaria':
        country_code = 'BG'
    elif country == 'italy':
        country_code = 'IT'

    field_preds = []
    for field in field_valdata:
        # Determine field_name
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
            field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten into dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": round(float(yt[0]), 3), "y_pred": round(float(yp[0]), 3),}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df['abs_error'] = (df['y_pred'] - df['y_test']).abs()

    # Aggregate error across models
    df = df.groupby(['field_name', 'y_test']).agg({'abs_error':'mean'}).reset_index()


    # Plot in-situ spectra compared to rest of values having same LAI GT
    data_path = f'../data/insitu_s2/{country}_fields'
    if country in ['switzerland', 'bulgaria']:
        cols_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
        x = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1613.7, 2202.4]
    else:
        cols_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B11', 'B12']
        x = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4]
   
    df_spectra = []
    for f in os.listdir(data_path):
        if country=='switzerland':
            if 'licor' not in f or 'clean' not in f:
                continue
            field_name = f.split('_')[-2].split('.pkl')[0]
        else:
            field_name = f.split('_')[-1].split('.pkl')[0]
        df_f = pd.read_pickle(os.path.join(data_path, f))
        df_f['field_name'] = field_name
        df_spectra.append(df_f)

    df_spectra = pd.concat(df_spectra, ignore_index=True)

    # Define LAI bins and labels
    bins = [0,1,2,3,4,5,8]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '>5']
    df = df.copy()
    df.loc[:, 'lai_bin'] = pd.cut(df['y_test'], bins=bins, labels=labels, right=False)

    # Define a GLOBAL palette from ALL available fields
    all_fields = df_spectra['field_name'].unique()
    palette = dict(zip(all_fields, sns.color_palette('tab10', n_colors=len(all_fields))))

    # Plot high error spectra by LAI bin (rows)
    n_rows = len(labels)
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 3*n_rows), sharex=True, sharey=True)

    if axs.ndim == 1:
        axs = axs[np.newaxis, :]

    for i, lai_bin in enumerate(labels):

        # Find data in that LAI range, and get top/lowest errors
        df_bin = df[df['lai_bin'] == lai_bin]
        if df_bin.empty:
          continue

        # Top and bottom within this bin
        df_top_bin = df_bin.sort_values('abs_error', ascending=False).head(10)
        df_bottom_bin = df_bin.sort_values('abs_error', ascending=True).head(10)

        # Merge spectra
        df_spectra_error = df_spectra.merge(
            df_top_bin[['field_name', 'y_test']],
            left_on=['field_name', 'lai'],
            right_on=['field_name', 'y_test'],
            how='inner'
        )
        df_spectra_other = df_spectra.merge(
            df_bottom_bin[['field_name', 'y_test']],
            left_on=['field_name', 'lai'],
            right_on=['field_name', 'y_test'],
            how='inner'
        )

        # Assign spectrum ids
        df_spectra_error = df_spectra_error.copy()
        df_spectra_other = df_spectra_other.copy()

        df_spectra_error['spec_id'] = np.arange(len(df_spectra_error))
        df_spectra_other['spec_id'] = np.arange(len(df_spectra_other))

        df_spectra_error['spectrum_id'] = (
            df_spectra_error['field_name'] + "_" +
            df_spectra_error['spec_id'].astype(str)
        )

        df_spectra_other['spectrum_id'] = (
            df_spectra_other['field_name'] + "_" +
            df_spectra_other['spec_id'].astype(str)
        )
      
        # High error
        df_long = df_spectra_error.melt(
            id_vars=['spectrum_id', 'field_name'],
            value_vars=cols_bands,
            var_name='band',
            value_name='reflectance'
        )
        df_long['wavelength'] = df_long['band'].map(dict(zip(cols_bands, x)))
        if not df_long.empty:
          sns.lineplot(
              data=df_long,
              x='wavelength',
              y='reflectance',
              hue='field_name',
              units='spectrum_id',
              palette=palette,
              linewidth=1,
              estimator=None,
              ax=axs[i,0]
          )
          
        # Other spectra
        df_long = df_spectra_other.melt(
            id_vars=['spectrum_id', 'field_name'],
            value_vars=cols_bands,
            var_name='band',
            value_name='reflectance'
        )
        df_long['wavelength'] = df_long['band'].map(dict(zip(cols_bands, x)))
        if not df_long.empty:
          sns.lineplot(
              data=df_long,
              x='wavelength',
              y='reflectance',
              hue='field_name',
              units='spectrum_id',
              palette=palette,
              linewidth=1,
              estimator=None,
              ax=axs[i,1]
          )

        axs[i,0].set_title(f'High Error Spectra (LAI bin: {lai_bin})', fontsize=20)
        axs[i,1].set_title(f'Low Error Spectra (LAI bin: {lai_bin})', fontsize=20)

    # Despine and format
    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Wavelength [nm]', fontsize=18)
        ax.set_ylabel('Reflectance', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.get_legend().remove() if ax.get_legend() else None #ax.legend(title='Field', fontsize=16, title_fontsize=16, loc='upper right')

    # ---- GLOBAL LEGEND ----
    handles = [mpatches.Patch(color=palette[f], label=f) for f in all_fields]
    # Add a global legend
    fig.legend(
        handles=handles,
        title='Field',
        fontsize=16,
        title_fontsize=16,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.95)
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'../model_results/{country}_fields/spectra_per_error_LAI.png')

"""


#################
# FIND HIGH ERROR POINTS, ANAYLSE SOIL GROUP
"""
countries = ['italy'] 

for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')
    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]

    if country == 'switzerland':
        field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
        country_code = 'CH'
    elif country == 'bulgaria':
        country_code = 'BG'
    elif country == 'italy':
        country_code = 'IT'

    field_preds = []
    for field in field_valdata:
        # Determine field_name
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
            field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten into dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": round(float(yt[0]), 3), "y_pred": round(float(yp[0]), 3),}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])

    df['abs_error'] = (df['y_pred'] - df['y_test']).abs()

    # Aggregate error across models
    df = df.groupby(['field_name', 'y_test']).agg({'abs_error':'mean'}).reset_index()


    # Plot in-situ spectra compared to rest of values having same LAI GT
    data_path = f'../data/insitu_s2/{country}_fields'
    if country in ['switzerland', 'bulgaria']:
        cols_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
        x = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1613.7, 2202.4]
    else:
        cols_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B11', 'B12']
        x = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4]
   
    df_spectra = []
    for f in os.listdir(data_path):
        if country=='switzerland':
            if 'licor' not in f or 'clean' not in f:
                continue
            field_name = f.split('_')[-2].split('.pkl')[0]
        else:
            field_name = f.split('_')[-1].split('.pkl')[0]
        df_f = pd.read_pickle(os.path.join(data_path, f))
        df_f['field_name'] = field_name
        df_spectra.append(df_f)

    df_spectra = pd.concat(df_spectra, ignore_index=True)
    df_spectra['country'] = country

    # Add soil information
    field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
    df = pd.merge(df_spectra, field_soil_group, on=['country', 'field_name'], how='left')
    
    colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
    cluster_labels = np.arange(1,6)
    cluster_color_map = dict(zip(cluster_labels, colors))

    # Select a soil group and plot the spectra
    df_soilgroup = df[df['soil_group'] == 3]
    df_soilgroup['spec_id'] = np.arange(len(df_soilgroup))
    df_soilgroup['spectrum_id'] = (
        df_soilgroup['field_name'] + "_" +
        df_soilgroup['spec_id'].astype(str)
    )
    df_long = df_soilgroup.melt(
        id_vars=['spectrum_id', 'field_name'],
        value_vars=cols_bands,
        var_name='band',
        value_name='reflectance'
    )
    df_long['wavelength'] = df_long['band'].map(dict(zip(cols_bands, x)))

    # Also select spectra of similar LAI
    lai = df[df['soil_group'] == 3]['lai'].values[0]
    df_other = df[df['lai'] > lai-0.5][df['lai'] < lai+0.5]
    df_other['spec_id'] = np.arange(len(df_other))
    df_other['spectrum_id'] = (
        df_other['field_name'] + "_" +
        df_other['spec_id'].astype(str)
    )
    df_long_other = df_other.melt(
        id_vars=['spectrum_id', 'field_name'],
        value_vars=cols_bands,
        var_name='band',
        value_name='reflectance'
    )
    df_long_other['wavelength'] = df_long_other['band'].map(dict(zip(cols_bands, x)))

    fig, axs = plt.subplots(figsize=(8, 3))

    # Plot other spectra in exact gray
    for spectrum_id in df_long_other['spectrum_id'].unique():
        subset = df_long_other[df_long_other['spectrum_id'] == spectrum_id]
        axs.plot(subset['wavelength'], subset['reflectance'], color='gray', linewidth=1)

    # Plot soil group 3 lines in exact red
    for spectrum_id in df_long['spectrum_id'].unique():
        subset = df_long[df_long['spectrum_id'] == spectrum_id]
        axs.plot(subset['wavelength'], subset['reflectance'], color='red', linewidth=1)


    # manually create legend
    import matplotlib.lines as mlines
    red_line = mlines.Line2D([], [], color='red', label=f'Soil group 3 (LAI={lai:.2f})')
    gray_line = mlines.Line2D([], [], color='gray', label=f'Other spectra ({lai-0.5:.2f}<LAI<{lai+0.5:.2f})')
    axs.legend(handles=[red_line, gray_line])

    axs.set_xlabel('Wavelength [nm]', fontsize=18)
    axs.set_ylabel('Reflectance', fontsize=18)
    axs.tick_params(axis='both', labelsize=16)
    axs.set_ylim(0,0.5)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'../model_results/{country}_fields/spectra_soilgroup3.png')
  
"""
    

#################
# FIND HIGH ERROR POINTS, ANAYLSE SOIL GROUP 
"""
countries = ['switzerland', 'bulgaria', 'italy']

for country in countries:
    val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')

    field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
    if country == 'switzerland':
      field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
      country_code = 'CH'
    elif country == 'bulgaria':
      country_code = 'BG'
    elif country == 'italy':
      country_code = 'IT'


    field_preds = []
    for field in field_valdata:
        # Run test for each field and save the preds/ground-truth
        if country == 'switzerland':
          if 'clean' not in field or 'licor' not in field:
            continue
          field_name = field.split('_')[-2].split('.pkl')[0]
        else:
          field_name = field.split('_')[-1].split('.pkl')[0]
        data_path = os.path.join(val_data_dir, field)

        # Run models
        models = {
            'Field': f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl',
            'Multi-field': '../models/NN_multifield_soil_tuned.pkl',
            'Large-scale': '../models/NN_europe_soil_tuned.pkl',
            'No-soil': '../models/NN_europe_nosoil_tuned.pkl'
        }
        for mname, mpath in models.items():
            y_test, y_pred = test_model(mpath, data_path)
            field_preds.append((mname, field_name, y_test.cpu().detach().numpy(), y_pred))

        # SNAP
        snap_file = f'../model_results/snap_baseline/{country}/snap_predictions_clean.xlsx'
        snap_df = pd.read_excel(snap_file, sheet_name=field_name)
        y_test = [[v] for v in snap_df['lai'].values]
        y_pred = [[v] for v in snap_df['snap_LAI'].values]
        field_preds.append(('SNAP', field_name, y_test, y_pred))

    # Flatten your list of tuples into a dataframe
    df = pd.DataFrame([
        {"model": model_type, "field_name": field_name, "y_test": yt[0], "y_pred": yp[0]}
        for model_type, field_name, y_test, y_pred in field_preds
        for yt, yp in zip(y_test, y_pred)
    ])
    df['country'] = country
    
    # Add soil information
    field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
    df = pd.merge(df, field_soil_group, on=['country', 'field_name'], how='left')
    
    colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
    cluster_labels = np.arange(1,6)
    cluster_color_map = dict(zip(cluster_labels, colors))

    n_models = len(df["model"].unique())
    model_types = df["model"].unique()
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

    if n_models == 1:
        axes = [axes]  

    for ax, model_type in zip(axes, model_types):
        df_model = df[df.model==model_type]
        
        sns.scatterplot(ax=ax, data=df_model, x="y_test", y="y_pred", hue="soil_group", palette=cluster_color_map)
        ax.plot([0, 8], [0, 8], 'k--', lw=1, label='1:1 line')  # 1:1 line (adjust limits if needed)
        ax.set_title(f"{model_type} model", fontsize=22)
        ax.set_xlabel("Measured LAI [m²/m²]", fontsize=20)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)
        ax.get_legend().remove()

        rmse = root_mean_squared_error(df_model['y_test'], df_model['y_pred'])
        r2 = r2_score(df_model['y_test'], df_model['y_pred'])
        textstr = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.8, textstr, transform=ax.transAxes, fontsize=16, bbox=props)


    axes[0].set_ylabel("Predicted LAI [m²/m²]", fontsize=20)
    axes[-1].legend(title='Soil group', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=16, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(f'../model_results/{country}_fields/allmodels_{country}_soilgroup.png')
"""










#################
# PLOT RMSEs PER FIELD, COMPARING MODELS, SNAP WITH BOXPLOTS
"""
field_results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/') # /country_fields/snap_predictions.xlsx
multifield_results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/') # /country_fields/NN_soil_multifield_country.xlsx
largescale_results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/') # /country_fields/NN_soil_europe_country.xlsx
nosoil_results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/') # /country_fields/NN_nosoil_europe_country.xlsx
snap_results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/model_results/snap_baseline') # /country/snap_predictions.xlsx

df_val_fieldmodel = []
df_val_multifield = []
df_val_largescale = []
df_val_nosoil = []
df_val_snap = []

country_code = {'switzerland': 'CH', 'bulgaria': 'BG', 'italy': 'IT'}

for country in ['switzerland', 'bulgaria', 'italy']:

  # Get Val RMSE per field
  val_data_dir = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_s2/{country.lower()}_fields')
  field_valdata = [f for f in os.listdir(val_data_dir) if 's2_val' in f]
  if country == 'switzerland':
    field_valdata = [f for f in field_valdata if 'licor' in f and 'clean' in f]
    country_code = 'CH'
  elif country == 'bulgaria':
    country_code = 'BG'
  elif country == 'italy':
    country_code = 'IT'

  field_preds = []
  for field in field_valdata:
        
      if 'clean' in field and country=='switzerland':
        field_name = field.split('_')[-2].split('.pkl')[0]
      else:
        field_name = field.split('_')[-1].split('.pkl')[0]
      data_path = os.path.join(val_data_dir, field)

      model_basename = f'../models/NN_{country_code}_soil_{field_name}_tuned.pkl' 
      y_test, y_pred = test_model(model_basename, data_path)
      rmse = root_mean_squared_error(y_test.cpu().detach().numpy(), y_pred)
      df_val_fieldmodel.append((country, field_name, rmse))



  # Get Multi-field RMSE per field
  field_dir = os.path.join(multifield_results_dir, f'{country}_fields')
  df_val = pd.read_excel(os.path.join(field_dir, f'NN_multifield_soil_{country}.xlsx'))
  for field, rmse in zip(df_val['field_name'], df_val['RMSE']):
    df_val_multifield.append((country, field, rmse))
  
  # Get large scale RMSE per field
  field_dir = os.path.join(largescale_results_dir, f'{country}_fields')
  df_val = pd.read_excel(os.path.join(field_dir, f'NN_europe_soil_{country}.xlsx'))
  for field, rmse in zip(df_val['field_name'], df_val['RMSE']):
    df_val_largescale.append((country, field, rmse))

  # Get nosoil RMSE per field
  field_dir = os.path.join(nosoil_results_dir, f'{country}_fields')
  df_val = pd.read_excel(os.path.join(field_dir, f'NN_nosoil_europe_{country}.xlsx'))
  for field, rmse in zip(df_val['field_name'], df_val['RMSE']):
    df_val_nosoil.append((country, field, rmse))

  # Get SNAP RMSE per field
  snap_file = os.path.join(snap_results_dir, f'{country}', 'snap_predictions_clean.xlsx')
  xls = pd.ExcelFile(snap_file)
  for field_name in xls.sheet_names:
      df_snap = pd.read_excel(xls, sheet_name=field_name)
      rmse = root_mean_squared_error(df_snap['lai'], df_snap['snap_LAI'])
      df_val_snap.append((country, field_name, rmse))
  

# Combine into one df
df_val_fieldmodel = pd.DataFrame(df_val_fieldmodel, columns=['country', 'field_name', 'rmse'])
df_val_multifield = pd.DataFrame(df_val_multifield, columns=['country', 'field_name', 'rmse'])
df_val_largescale = pd.DataFrame(df_val_largescale, columns=['country', 'field_name', 'rmse'])
df_val_nosoil = pd.DataFrame(df_val_nosoil, columns=['country', 'field_name', 'rmse'])
df_val_snap = pd.DataFrame(df_val_snap, columns=['country', 'field_name', 'rmse'])

dfs = [
    df_val_fieldmodel.assign(model='Field'),
    df_val_multifield.assign(model='Multi-field'),
    df_val_largescale.assign(model='Large-scale'),
    df_val_nosoil.assign(model='No soil'),
    df_val_snap.assign(model='SNAP'),
]

df_all = pd.concat(dfs, ignore_index=True)
df_all['field_name'] = df_all['field_name'].astype(str)
merged = df_all.pivot_table(index=['country', 'field_name'], columns='model', values='rmse').reset_index()
merged.columns.name = None
merged = merged.rename_axis(None, axis=1)


# Add soil information
field_soil_group = pd.read_csv('baresoil/field_soil_groups.csv')
merged = pd.merge(merged, field_soil_group, on=['country', 'field_name'])

colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen']
cluster_labels = np.arange(1,6)
cluster_color_map = dict(zip(cluster_labels, colors))


# Plot regardless of soil type
df_melt = merged.melt(
    id_vars=['country', 'field_name'], #, 'soil_group'
    value_vars=['Field', 'Multi-field', 'Large-scale', 'No soil', 'SNAP'],  # add more models if needed
    var_name='Model',
    value_name='RMSE'
)

plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    boxprops=dict(alpha=0.3),      # transparency
    fliersize=0,                   # hide outlier dots (since we'll plot all points)
    linewidth=1
)
sns.stripplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    color='red',
    dodge=True,                    
    jitter=0.2,                   # horizontal jitter
    size=6                         # dot size
)
plt.title('Field-level RMSE', fontsize=18)
plt.xlabel('Model', fontsize=16)
plt.ylabel('RMSE [m²/m²]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
sns.despine(top=True, right=True)
plt.savefig('boxplot_rmse_compare_models.png')



# Plot with soil group as hue
df_melt = merged.melt(
    id_vars=['country', 'field_name', 'soil_group'],
    value_vars=['Field', 'Multi-field', 'Large-scale', 'No soil', 'SNAP'],  # add more models if needed
    var_name='Model',
    value_name='RMSE'
)

plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    #hue='soil_group',
    #palette=cluster_color_map,
    boxprops=dict(alpha=0.3),      # transparency
    fliersize=0,                   # hide outlier dots (since we'll plot all points)
    linewidth=1
)
sns.stripplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    hue='soil_group',
    dodge=True,                    # separate points by hue
    palette=cluster_color_map,
    #alpha=0.6,                     # make dots semi-transparent
    jitter=0.2,                   # horizontal jitter
    size=6                         # dot size
)
plt.legend(title='Soil Group', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=12, fontsize=12)
plt.xlabel('Model', fontsize=18)
plt.ylabel('Mean field RMSE [m²/m²]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
sns.despine(top=True, right=True)
plt.savefig('boxplot_rmse_compare_models_soil.png')


# Plot with hue depending on country
df_melt = merged.melt(
    id_vars=['country', 'field_name'],
    value_vars=['Field', 'Multi-field', 'Large-scale', 'No soil', 'SNAP'],  # add more models if needed
    var_name='Model',
    value_name='RMSE'
)
df_melt['country'] = df_melt['country'].str.capitalize()

plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    boxprops=dict(alpha=0.3),      # transparency
    fliersize=0,                   # hide outlier dots (since we'll plot all points)
    linewidth=1
)
sns.stripplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    hue='country',
    dodge=True,                    # separate points by hue
    #palette=cluster_color_map,
    #alpha=0.6,                     # make dots semi-transparent
    jitter=0.2,                   # horizontal jitter
    size=6                         # dot size
)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=12, fontsize=12)
plt.title('Field-level RMSE across models', fontsize=18)
plt.xlabel('Model', fontsize=16)
plt.ylabel('RMSE [m²/m²]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
sns.despine(top=True, right=True)
plt.savefig('boxplot_rmse_compare_models_countries.png')


# Plot with hue depending on country AND markers for soil group

df_melt = merged.melt(
    id_vars=['country', 'field_name', 'soil_group'],
    value_vars=['Field', 'Multi-field', 'Large-scale', 'No soil', 'SNAP'],  # add more models if needed
    var_name='Model',
    value_name='RMSE'
)

markers = {
    'Switzerland': 'o',  # circle
    'Bulgaria': '^',     # triangle
    'Italy': 's'         # square
}

plt.figure(figsize=(10, 5))

# Boxplot (without hue, since we use colors for soil_group in points)
sns.boxplot(
    data=df_melt,
    x='Model',
    y='RMSE',
    boxprops=dict(alpha=0.3),    
    fliersize=0,
    linewidth=1
)

# Stripplot for each country separately
for country, marker in markers.items():
    df_country = df_melt[df_melt['country'].str.capitalize() == country]
    sns.stripplot(
        data=df_country,
        x='Model',
        y='RMSE',
        hue='soil_group',
        dodge=True,
        jitter=0.2,
        size=6,
        marker=marker,
        palette=cluster_color_map,
        linewidth=0,
        legend=False  # avoid duplicate legend
    )

# Create a separate legend for soil groups
soil_groups = df_melt['soil_group'].unique()
handles_soil = [mpatches.Patch(color=cluster_color_map[i+1], label=i+1)
                for i in range(len(soil_groups))]
legend1 = plt.legend(handles=handles_soil, title='Soil Group',
                     bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=12)

# Create legend for markers (countries)
handles_countries = [Line2D([0], [0], marker=m, color='black', linestyle='',
                            markersize=8, label=country)
                     for country, m in markers.items()]
legend2 = plt.legend(handles=handles_countries, title='Country',
                     bbox_to_anchor=(1.05, 0.5), loc='upper left', fontsize=12, title_fontsize=12)
plt.gca().add_artist(legend1)
plt.title('Field-level RMSE', fontsize=18)
plt.xlabel('Model', fontsize=16)
plt.ylabel('RMSE [m²/m²]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
sns.despine(top=True, right=True)
plt.tight_layout()
plt.savefig('boxplot_rmse_compare_models_country_soil.png')
"""