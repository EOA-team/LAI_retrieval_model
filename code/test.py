'''
Test a trained model

@author Selene Ledain
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from argparse import ArgumentParser
import yaml
from typing import Dict, Tuple, Union, Any
import pickle
import torch
from scipy import stats
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from models import MODELS

def load_config(config_path: str) -> Dict:
  ''' 
  Load configuration file

  :param config_path: path to yaml file
  :returns: dictionary of parameters
  '''
  with open(config_path, "r") as config_file:
      config = yaml.safe_load(config_file)
  return config


def prepare_data_train(config: dict) -> Union[Tuple[np.array, np.array, np.array, np.array], None]:
  ''' 
  Load data and prepare training sets

  :param config: dictionary of configuration parameters
  :returns: X pd.DataFrame and y pd.Series for training and test sets 
  '''
  data_path = config['Data']['data_path']

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
    X = df[config['Data']['train_cols']]
    y = df[config['Data']['target_col']]
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])

    X_soil = pd.DataFrame()
    y_soil = pd.Series()
    if 'baresoil_samples' in config['Data'].keys():
      baresoil_dfs = [pd.read_pickle(path) for path in config['Data']['baresoil_samples']]
      concatenated_df = pd.concat(baresoil_dfs, axis=0, ignore_index=True)
      X_soil = concatenated_df[config['Data']['train_cols']]
      y_soil = pd.Series([0]*len(X_soil))

    X_train = pd.concat([X_train , X_soil], ignore_index=True)
    y_train = pd.concat([y_train , y_soil], ignore_index=True)

    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X_train.diff(axis=1)
      for col in X_train.columns[1:]:
          X_train[col + '_derivative'] = derivatives[col]
      derivatives = X_test.diff(axis=1)
      for col in X_test.columns[1:]:
          X_test[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X_train['ndvi'] = (X_train['B08'] - X_train['B04'])/(X_train['B08'] + X_train['B04'])
      X_test['ndvi'] = (X_test['B08'] - X_test['B04'])/(X_test['B08'] + X_test['B04'])

    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.pkl')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)
      return X_train, X_test, y_train, y_test
    else:
      return X_train.values, X_test.values, y_train, y_test

  elif isinstance(data_path, list):
    # Assuming all files in the list are pickled DataFrames
    dfs = [pd.read_pickle(path) for path in data_path]
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    # Sample 50000 data pairs
    #sampled_df = concatenated_df.sample(50000, random_state=config['Seed']) if len(concatenated_df) > 50000 else concatenated_df
    X = concatenated_df[config['Data']['train_cols']] #sampled_df[config['Data']['train_cols']] #
    y = concatenated_df[config['Data']['target_col']] #sampled_df[config['Data']['target_col']] #  
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])

    X_soil = pd.DataFrame()
    y_soil = pd.Series()
    if 'baresoil_samples' in config['Data'].keys():
      baresoil_dfs = [pd.read_pickle(path) for path in config['Data']['baresoil_samples']]
      concatenated_df = pd.concat(baresoil_dfs, axis=0, ignore_index=True)
      X_soil = concatenated_df[config['Data']['train_cols']]
      y_soil = pd.Series([0]*len(X_soil))
    
    X_train = pd.concat([X_train , X_soil], ignore_index=True)
    y_train = pd.concat([pd.Series(y_train), y_soil], ignore_index=True)

    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X_train.diff(axis=1)
      for col in X_train.columns[1:]:
          X_train[col + '_derivative'] = derivatives[col]
      derivatives = X_test.diff(axis=1)
      for col in X_test.columns[1:]:
          X_test[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X_train['ndvi'] = (X_train['B08'] - X_train['B04'])/(X_train['B08'] + X_train['B04'])
      X_test['ndvi'] = (X_test['B08'] - X_test['B04'])/(X_test['B08'] + X_test['B04'])

    #print(len(X_train), len(X_test))
    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.pkl')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)
      return X_train, X_test, y_train, y_test
    else:
      return X_train.values, X_test.values, y_train, y_test

  else:
      return None


def prepare_data_test(config: dict) -> Union[Tuple[np.array, np.array, np.array, np.array], None]:
  ''' 
  Load data and prepare test sets

  :param config: dictionary of configuration parameters
  :returns: X pd.DataFrame and y pd.Series for training and test sets 
  '''
  data_path = config['Data']['val_data_path']

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
    df = df[~df[config['Data']['target_col']].isna()]
    X = df[config['Data']['train_cols']]
    y = df[config['Data']['target_col']]
    sites = df['site']
    years = df['year']

    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.pkl')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X = scaler.transform(X)
      return X, y.values, sites, years
      #X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=config['Data']['test_size'], random_state=config['Seed'])
      #print('here')
      #return X_test, y_test
    else:
      return X, y.values, sites, years

  elif isinstance(data_path, list):
    # Assuming all files in the list are pickled DataFrames
    dfs = [pd.read_pickle(path) for path in data_path]
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    concatenated_df = concatenated_df[~concatenated_df[config['Data']['target_col']].isna()]
    X = concatenated_df[config['Data']['train_cols']] #  concatenated_df[config['Data']['train_cols']]
    y = concatenated_df[config['Data']['target_col']] #  concatenated_df[config['Data']['target_col']]
    sites = concatenated_df['site'] if 'site' in concatenated_df.columns else None
    years = concatenated_df['year'] if 'year' in concatenated_df.columns else None
 
    if config['Model']['name'] == 'RF':
      # Add derivatives
      derivatives = X.diff(axis=1)
      for col in X.columns[1:]:
          X[col + '_derivative'] = derivatives[col]
      # Add NDVI
      X['ndvi'] = (X['B08'] - X['B04'])/(X['B08'] + X['B04'])
      
          
    if config['Data']['normalize']:
      # Load scaler
      scaler_path = config['Model']['save_path'].split('.pkl')[0] + '_scaler.pkl'
      with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
      # Normalize
      X = scaler.transform(X)
      return X, y.values, sites, years
    else:
      return X, y.values, sites, years

  else:
      return None


def build_model(config: dict) -> Any:
  ''' 
  Instantiated model

  :param config: dictionary of configuration parameters
  :returns: model
  '''
  model_name = config['Model']['name']
  if model_name not in MODELS:
    raise ValueError(f"Invalid model type: {model_name}")
  else:
    # Model hypereparameters can be set in the config, else default values used
    model_params = {key: value for key, value in config['Model'].items() if key != 'name'}  # Pass only hyperparams
    model = MODELS[model_name](**model_params)
  
  return model


def compute_other_scores(y_test, y_pred, dataset, score_path):
  """ 
  Compute nromalised RMSE and pearson's r squared, and add to score_path
  """

  # Move y_pred to CPU if it's on CUDA device
  if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
      y_pred = y_pred.cpu().detach().numpy()
  if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
      y_test = y_test.cpu().detach().numpy()
  try:
    nrmse = root_mean_squared_error(y_test, y_pred)/(np.max(y_test) - np.min(y_test))
  except:
    nrmse = np.nan
  try:
    pearson = stats.pearsonr(y_test, y_pred).statistic
  except:
    pearson = np.nan

  print(f'nRMSE: {nrmse}')
  print(f'Pearson r2: {pearson**2}')
  """
  # Open excel file at score_path and append results
  score_data = {
      'Dataset': [dataset],
      'nRMSE': [nrmse],
      'r2': [pearson**2],
  }
  score_df = pd.DataFrame(score_data)
  """
  if score_path is not None:
    if os.path.exists(score_path):
        existing_df = pd.read_excel(score_path)
        if 'nRMSE' not in existing_df.columns:
            existing_df['nRMSE'] = [None]*len(existing_df)
        existing_df.loc[existing_df['Dataset'] == dataset, 'nRMSE']= nrmse
        if 'r2' not in existing_df.columns:
            existing_df['r2'] = [None]*len(existing_df)
        existing_df.loc[existing_df['Dataset'] == dataset, 'r2'] = pearson**2

        existing_df.to_excel(score_path, index=False)

  return


def plot_preds(y, y_true, save_path, score_path):

  score_df = pd.read_excel(score_path)
  score_df['Dataset'] = score_df['Dataset'].apply(lambda x: x.split(' ')[0])
  mean_scores = score_df.groupby('Dataset').mean().reset_index()
  std_scores = score_df.groupby('Dataset').std().reset_index()

  textstr_test = f'RMSE: {mean_scores[mean_scores.Dataset=="Val"].RMSE.values[0]:.3f}\nnRMSE: {mean_scores[mean_scores.Dataset=="Val"].nRMSE.values[0]:.3f}\n$R^2$: {mean_scores[mean_scores.Dataset=="Val"].r2.values[0]:.3f}'
  
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    # Calculate the point density
  xy = np.vstack([np.array(y).reshape(-1),np.array(y_true).reshape(-1) ])
  z = gaussian_kde(xy)(xy)

  scatter = axs.scatter(y, y_true, c=z, cmap='viridis')
  axs.set_xlabel('Val set LAI', size=18)
  axs.set_ylabel('Predicted LAI', size=18)
  if 'destructive' in save_path:
    axs.set_xlim((-0.3,3))
    axs.set_ylim((-0.3,3))
    ticks = np.arange(0, 3, 1)
  else:
    axs.set_xlim((-0.3,8))
    axs.set_ylim((-0.3,8)) 
    ticks = np.arange(0, 9, 2)  # Adjust the range and step size as needed
  axs.set_xticks(ticks)
  axs.set_yticks(ticks)
  axs.tick_params(axis='both', which='major', labelsize=16) 

  props = dict(boxstyle='round', facecolor='white', alpha=0.5)
  axs.text(0.03, 0.8, textstr_test, transform=axs.transAxes, fontsize=16, bbox=props)

  # Plot y=x line
  axs.plot([0, 10], [0, 10], color='gray', linestyle='--')

  # Hide the top and right spine
  axs.spines["right"].set_visible(False)
  axs.spines["top"].set_visible(False)

  cbar = fig.colorbar(scatter, ax=axs)
  cbar.set_label('Density', size=14)
  cbar.ax.tick_params(labelsize=14)

  plt.savefig(save_path)

  return


def plot_preds_per_site(y, y_true, sites, save_path, score_path):

    score_df = pd.read_excel(score_path)
    score_df['Dataset'] = score_df['Dataset'].apply(lambda x: x.split(' ')[0])
    mean_scores = score_df.groupby('Dataset').mean().reset_index()
    std_scores = score_df.groupby('Dataset').std().reset_index()

    textstr_test = f'RMSE: {mean_scores[mean_scores.Dataset=="Val"].RMSE.values[0]:.3f}\nnRMSE: {mean_scores[mean_scores.Dataset=="Val"].nRMSE.values[0]:.3f}\n$R^2$: {mean_scores[mean_scores.Dataset=="Val"].r2.values[0]:.3f}'
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    # Assign a unique color to each site
    unique_sites = sites.unique()
    cmap = plt.colormaps.get_cmap('tab10').resampled(len(unique_sites))
    site_color_map = {site: cmap(i) for i, site in enumerate(unique_sites)}
    colors = sites.map(site_color_map)

    scatter = axs.scatter(y, y_true, c=colors, alpha=0.8, edgecolor='k', linewidth=0.3)

    axs.set_xlabel('Val set LAI', size=18)
    axs.set_ylabel('Predicted LAI', size=18)

    if 'destructive' in save_path:
        axs.set_xlim((-0.3, 3))
        axs.set_ylim((-0.3, 3))
        ticks = np.arange(0, 3, 1)
    else:
        axs.set_xlim((-0.3, 8))
        axs.set_ylim((-0.3, 8))
        ticks = np.arange(0, 9, 2)

    axs.set_xticks(ticks)
    axs.set_yticks(ticks)
    axs.tick_params(axis='both', which='major', labelsize=16)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs.text(0.03, 0.8, textstr_test, transform=axs.transAxes, fontsize=16, bbox=props)

    # Plot y=x line
    axs.plot([0, 10], [0, 10], color='gray', linestyle='--')

    # Hide the top and right spines
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)

    # Create legend per site
    for site in unique_sites:
        axs.scatter([], [], color=site_color_map[site], label=site)
    axs.legend(title="Site", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    
    plt.savefig(save_path)

    return


def plot_preds_per_site_year(y, y_true, sites, years, save_path, score_path):

    score_df = pd.read_excel(score_path)
    score_df['Dataset'] = score_df['Dataset'].apply(lambda x: x.split(' ')[0])
    mean_scores = score_df.groupby('Dataset').mean().reset_index()
    std_scores = score_df.groupby('Dataset').std().reset_index()

    textstr_test = f'RMSE: {mean_scores[mean_scores.Dataset=="Val"].RMSE.values[0]:.3f}\nnRMSE: {mean_scores[mean_scores.Dataset=="Val"].nRMSE.values[0]:.3f}\n$R^2$: {mean_scores[mean_scores.Dataset=="Val"].r2.values[0]:.3f}'
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    # Assign a unique color to each site
    site_year = sites.astype(str) + '-' + years.astype(str)
    unique_site_years = site_year.unique()
    cmap = plt.colormaps.get_cmap('tab20').resampled(len(unique_site_years))
    siteyear_color_map = {sy: cmap(i) for i, sy in enumerate(unique_site_years)}
    colors = site_year.map(siteyear_color_map)

    scatter = axs.scatter(y, y_true, c=colors, alpha=0.8, edgecolor='k', linewidth=0.3)

    axs.set_xlabel('Val set LAI', size=18)
    axs.set_ylabel('Predicted LAI', size=18)

    if 'destructive' in save_path:
        axs.set_xlim((-0.3, 3))
        axs.set_ylim((-0.3, 3))
        ticks = np.arange(0, 3, 1)
    else:
        axs.set_xlim((-0.3, 8))
        axs.set_ylim((-0.3, 8))
        ticks = np.arange(0, 9, 2)

    axs.set_xticks(ticks)
    axs.set_yticks(ticks)
    axs.tick_params(axis='both', which='major', labelsize=16)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs.text(0.03, 0.8, textstr_test, transform=axs.transAxes, fontsize=16, bbox=props)

    # Plot y=x line
    axs.plot([0, 10], [0, 10], color='gray', linestyle='--')

    # Hide the top and right spines
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)

    # Create legend per site
    for sy in unique_site_years:
        axs.scatter([], [], color=siteyear_color_map[sy], label=sy)
    axs.legend(title="Site-Year", fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)

    return


def test_model(config: dict) -> None:
  ''' 
  Test model on a dataset and get scores

  :param config: dictionary of configuration parameters
  '''

  if not isinstance(config['Seed'], list):
    config['Seed'] = [config['Seed']]

  model_basename = config['Model']['save_path']  
  save_model = config['Model'].pop('save')
  score_path = config['Model'].pop('score_path') if 'score_path' in config['Model'].keys() else None
  plot = config['Model'].pop('plot') 

  y_pred_seed = []
  for seed in config['Seed']:
    print('Running with seed', seed)

    config['Model']['save_path'] = model_basename.split('.pkl')[0] + f'{seed}.pkl'

    #############################################
    # DATA
    X_test, y_test, sites, years = prepare_data_test(config=config) # unseen validation data (in situ)
    #_, X_train, _, y_train = prepare_data_train(config=config) # performance on training data

    # Move data to CUDA if GPUs requested and available
    device = torch.device('cuda' if config['Model'].get('gpu') and torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        X_test, y_test = (
            torch.FloatTensor(X_test).to(device),
            torch.FloatTensor(y_test).view(-1, 1).to(device)
        )
        """ 
        X_train, y_train = (
            torch.FloatTensor(X_train).to(device),
            torch.FloatTensor(y_train).view(-1, 1).to(device)
        )
        """

    #############################################
    # MODEL
    model_name = config['Model']['name']
    model_filename = config['Model'].pop('save_path') 
    with open(model_filename, 'rb') as f:
      model = pickle.load(f)

    # Move model to CUDA if GPUs are available
    if device == torch.device('cuda'):
        model.to(device)

    #############################################
    # TEST
    y_pred = model.predict(X_test=X_test)
    if not np.isnan(y_pred.flatten()).any():
      model.test_scores(y_test=y_test.flatten(), y_pred=y_pred.flatten(), dataset=f'Val {seed}', score_path=score_path)
      compute_other_scores(y_test=y_test.flatten(), y_pred=y_pred.flatten(), dataset=f'Val {seed}', score_path=score_path)
      y_pred_seed.append(y_pred)
  
  if plot:
    y_pred = np.mean(y_pred_seed, axis=0)
    if device == torch.device('cuda'):
      #y_pred = y_pred.cpu().detach().numpy()
      y_test = y_test.cpu().detach().numpy()
    plot_preds(y_test, y_pred, f'../model_results/val_preds_{model_filename.split("/")[-1].split(".pkl")[0]}.png', score_path)
    #plot_preds(y_test, y_pred, f'../model_results/val_preds_{model_filename.split("/")[-1].split(".pkl")[0]}_{config['Model']['val_plot_suffix']}.png', score_path)
    #plot_preds_per_site(y_test, y_pred, sites, f'../model_results/val_preds_{model_filename.split("/")[-1].split(".pkl")[0]}_{config['Model']['val_plot_suffix']}_persite.png', score_path)
    #plot_preds_per_site_year(y_test, y_pred, sites, years, f'../model_results/val_preds_{model_filename.split("/")[-1].split(".pkl")[0]}_{config['Model']['val_plot_suffix']}_persiteyear.png', score_path)
  return


    
if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'
  parser = ArgumentParser()
  parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
  args = parser.parse_args()

  config = load_config(args.setting)
  test_model(config)
