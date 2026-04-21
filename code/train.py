'''
Train a model to perform a RTM inversion

@author Selene Ledain
'''

import os
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
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import root_mean_squared_error
from scipy import stats

def load_config(config_path: str) -> Dict:
  ''' 
  Load configuration file

  :param config_path: path to yaml file
  :returns: dictionary of parameters
  '''
  with open(config_path, "r") as config_file:
      config = yaml.safe_load(config_file)
  return config


def prepare_data(config: dict) -> Union[Tuple[np.array, np.array, np.array, np.array], None]:
  ''' 
  Load data and prepare training and testing sets

  :param config: dictionary of configuration parameters
  :returns: X pd.DataFrame and y pd.Series for training and test sets 
  '''
  data_path = config['Data']['data_path']
  test_data_path = config['Data']['test_data_path']

  ##### Load test data

  if isinstance(test_data_path, str):
    df = pd.read_pickle(test_data_path)
    X_test = df[config['Data']['train_cols']]
    y_test = df[config['Data']['target_col']]

  elif isinstance(test_data_path, list):
    # Assuming all files in the list are pickled DataFrames
    dfs = [pd.read_pickle(path) for path in test_data_path]
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    X_test = concatenated_df[config['Data']['train_cols']]
    y_test = concatenated_df[config['Data']['target_col']] 
   
  ##### Load train data, normalize train and test

  if isinstance(data_path, str):
    df = pd.read_pickle(data_path)
    X_train = df[config['Data']['train_cols']]
    y_train = df[config['Data']['target_col']]
    
    X_soil = pd.DataFrame()
    y_soil = pd.Series()
    if 'baresoil_samples' in config['Data'].keys():
      baresoil_dfs = [pd.read_csv(path) for path in config['Data']['baresoil_samples']]
      concatenated_df = pd.concat(baresoil_dfs, axis=0, ignore_index=True)
      X_soil = concatenated_df[config['Data']['train_cols']]
      y_soil = pd.Series([0]*len(X_soil))

    X_train = pd.concat([X_train , X_soil], ignore_index=True)
    y_train = pd.concat([y_train , y_soil], ignore_index=True)

    if config['Data']['normalize']:
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train) # becomes an array
      X_test = scaler.transform(X_test)
      # Save for model inference
      scaler_path = config['Model']['save_path'].split('.pkl')[0] + '_scaler.pkl' \
        if 'save_path' in config['Model'].keys() \
        else config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_scaler.pkl' 
      os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
      
      with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        return X_train, X_test, y_train.values, y_test.values
    else:
      return X_train.values, X_test.values, y_train, y_test

  

  elif isinstance(data_path, list):
    # Assuming all files in the list are pickled DataFrames
    dfs = [pd.read_pickle(path) for path in data_path]
    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    #concatenated_df = concatenated_df.sample(10, random_state=config['Seed'])
    X_train = concatenated_df[config['Data']['train_cols']]
    y_train = concatenated_df[config['Data']['target_col']] 
    
    X_soil = pd.DataFrame()
    y_soil = pd.Series()
    if 'baresoil_samples' in config['Data'].keys():
      baresoil_dfs = [pd.read_csv(path) for path in config['Data']['baresoil_samples']]
      concatenated_df = pd.concat(baresoil_dfs, axis=0, ignore_index=True)
      # If there is extra variables in the train columns than the bands (like LAI), add them as 0
      if any(not b.startswith('B') for b in config['Data']['train_cols']):
        extra_cols = [b for b in config['Data']['train_cols'] if not b.startswith('B')]
        if extra_cols: 
          concatenated_df = concatenated_df.assign(**{b: 0 for b in extra_cols})
      X_soil = concatenated_df[config['Data']['train_cols']]
      y_soil = pd.Series([0]*len(X_soil))
    
    X_train = pd.concat([X_train , X_soil], ignore_index=True)
    y_train = pd.concat([pd.Series(y_train), y_soil], ignore_index=True)

    if config['Data']['normalize']:
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train) # becomes an array
      X_test = scaler.transform(X_test)
      # Save for model inference
      scaler_path = config['Model']['save_path'].split('.pkl')[0] + '_scaler.pkl' \
        if 'save_path' in config['Model'].keys() \
        else config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_scaler.pkl' 
      os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

      with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        return X_train, X_test, y_train.values, y_test
    else:
      return X_train.values, X_test.values, y_train, y_test

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
    #if model_name == 'NN':
      #torch.manual_seed(config['Seed'])
    # Model hypereparameters can be set in the config, else default values used
    model_params = {key: value for key, value in config['Model'].items() if key != 'name'}  # Pass only hyperparams
    model_params['random_state'] = config['Seed']
    model = MODELS[model_name](**model_params)
  
  return model


def plot_preds(y, y_true, save_path, score_path, dataset, trait):

  plot_params = {
    'lai': {'min':-0.3, 'max':8},
    'cab': {'min':-0.3, 'max':80},
    'car': {'min': -0.3, 'max':15},
    'ant': {'min': -0.3, 'max':2},
    'cw': {'min': -0.0005, 'max':0.02},
    'cm': {'min': -0.0005, 'max':0.05},
    'prot': {'min': 0.001, 'max':0.0025},
    'cbc': {'min': -0.001, 'max':0.01},
    'FAPAR': {'min': -0.001, 'max':1.1}
  }

  score_df = pd.read_excel(score_path)
  score_df['Dataset'] = score_df['Dataset'].apply(lambda x: x.split(' ')[0])
  mean_scores = score_df.groupby('Dataset').mean().reset_index()
  std_scores = score_df.groupby('Dataset').std().reset_index()

  textstr_test = f'RMSE: {mean_scores[mean_scores.Dataset==dataset].RMSE.values[0]:.3f}\nnRMSE: {mean_scores[mean_scores.Dataset==dataset].nRMSE.values[0]:.3f}\n$R^2$: {mean_scores[mean_scores.Dataset==dataset].r2.values[0]:.3f}'
  
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    # Calculate the point density
  xy = np.vstack([np.array(y).reshape(-1),np.array(y_true).reshape(-1) ])
  z = gaussian_kde(xy)(xy)

  scatter = axs.scatter(y, y_true, c=z, cmap='viridis')
  axs.set_xlabel(f'{dataset} set {trait}', size=18)
  axs.set_ylabel(f'Predicted {trait}', size=18)
  axs.set_xlim((plot_params[trait]['min'], plot_params[trait]['max']))
  axs.set_ylim((plot_params[trait]['min'], plot_params[trait]['max']))
  #ticks = np.arange(0, 9, 2)  # Adjust the range and step size as needed
  #axs.set_xticks(ticks)
  #axs.set_yticks(ticks)
  #axs.tick_params(axis='both', which='major', labelsize=16) 

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


def compute_other_scores(y_test, y_pred, dataset, score_path):
  """ 
  Compute nromalised RMSE and pearson's r squared, and add to score_path
  """

  # Move y_pred to CPU if it's on CUDA device
  if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
      y_pred = y_pred.cpu().detach().numpy()
  if isinstance(y_test, torch.Tensor) and y_test.device.type == 'cuda':
      y_test = y_test.cpu().detach().numpy()

  nrmse = root_mean_squared_error(y_test, y_pred)/(np.max(y_test) - np.min(y_test))
  pearson = stats.pearsonr(y_test, y_pred).statistic

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


def train_model(config: dict) -> None:
  ''' 
  Train model on training set, get scores on test set, save model

  :param config: dictionary of configuration parameters
  '''
  model_basename = config['Model'].pop('save_path') if 'save_path' in config['Model'].keys() else model_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl'
  save_model = config['Model'].pop('save')
  score_path = config['Model'].pop('score_path') if 'score_path' in config['Model'].keys() else None
  gpu = config['Model'].pop('gpu')
  plot = config['Model'].pop('plot')
  trait = config['Data']['target_col']
  noise = config['Model'].pop('noise')

  if not isinstance(config['Seed'], list):
    config['Seed'] = [config['Seed']]

  y_pred_seed = []
  y_train_seed = []
  for seed in config['Seed']:
    print('Running with seed', seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    config['Seed'] = seed

    #############################################
    # DATA
    config['Model']['save_path'] = model_basename.split('.pkl')[0] + f'{seed}.pkl' 
    X_train, X_test, y_train, y_test = prepare_data(config=config)  
    
    # Add noise
    if noise:
      noise_model = pd.read_csv('noise_snap.csv', delimiter=';')
      AD = noise_model['AD'].values  # Additive deviation per band
      AI = noise_model['AI'].values  # Additive intensity
      MD = noise_model['MD'].values  # Multiplicative deviation per band (in %)
      MI = noise_model['MI'].values  # Multiplicative intensity (in %)
      noise_multiplier = 1 + (MD + MI) / 100  # Multiplicative noise
      noise_additive = AD + AI  # Additive noise
      X_train = X_train * noise_multiplier + noise_additive
      X_test = X_test * noise_multiplier + noise_additive
    
    # Move data to CUDA if GPUs requested and available
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
      X_train, X_test, y_train, y_test = (
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(X_test).to(device),
        torch.FloatTensor(y_train).view(-1, 1).to(device),
        torch.FloatTensor(y_test).view(-1, 1).to(device),
      ) 
  
    #############################################
    # MODEL
    if gpu and torch.cuda.is_available():
      print('Using GPUs')
    
    model_name = config['Model']['name']
    model_filename = config['Model'].pop('save_path') # path to save trained model 
    model = build_model(config=config)
    if device == torch.device('cuda'):
      model.to(device)

    #############################################
    # TRAIN
    model.fit(X=X_train, y=y_train,  X_test=X_test, y_test=y_test)

    # Extract training results
    y_pred = model.predict(X_test=X_train)
    if not isinstance(y_train, pd.Series):
        y_train = y_train.flatten()
    if not np.isnan(y_pred.flatten()).any():
      model.test_scores(y_test=y_train, y_pred=y_pred.flatten(), dataset=f'Train {seed}', score_path=score_path)
      compute_other_scores(y_test=y_train, y_pred=y_pred.flatten(), dataset=f'Train {seed}', score_path=score_path)
      y_train_seed.append(y_pred)

    #############################################
    # TEST 
    y_pred = model.predict(X_test=X_test)
    if not isinstance(y_test, pd.Series):
        y_test = y_test.flatten()
    if not np.isnan(y_pred.flatten()).any():
      model.test_scores(y_test=y_test, y_pred=y_pred.flatten(), dataset=f'Test {seed}', score_path=score_path)
      compute_other_scores(y_test=y_test, y_pred=y_pred.flatten(), dataset=f'Test {seed}', score_path=score_path)
      y_pred_seed.append(y_pred)
      
    #############################################
    # SAVE 
    if save_model:
      model.save(model=model, model_filename=model_filename)

  #############################################
  # PLOT

  if plot:
    # Plot train predictions
    y_pred = np.mean(y_train_seed, axis=0)
    if device == torch.device('cuda'):
      y_train = y_train.cpu().detach().numpy()
    plot_preds(y_train, y_pred, f'../model_results/train_preds_{model_filename.split("/")[-1].split(".pkl")[0]}.png', score_path, dataset='Train', trait=trait)

    # Plot test predictions
    y_pred = np.mean(y_pred_seed, axis=0)
    if device == torch.device('cuda'):
      y_test = y_test.cpu().detach().numpy()
    plot_preds(y_test, y_pred, f'../model_results/test_preds_{model_filename.split("/")[-1].split(".pkl")[0]}.png', score_path, dataset='Test', trait=trait)

  return


    
if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'
  parser = ArgumentParser()
  parser.add_argument('setting', type = str, metavar='path/to/setting.yaml', help='yaml with all settings')
  args = parser.parse_args()

  config = load_config(args.setting)
  train_model(config)
