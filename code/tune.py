''' 
Perform hyperparameter tuning for a RTM-inversion model

@Selene Ledain
'''
# Read model architecture to tune
# Read parameter grid
# What default values and data path to use? Should put in the yaml anyways?
# Send parameters to train.py
# Save results somehow (.csv?)

from argparse import ArgumentParser
import yaml
from typing import Dict, Tuple, Union, Any
from datetime import datetime
import os
import csv
from subprocess import run, PIPE
import itertools
import pandas as pd
import optuna
import numpy as np
import torch

def load_config(config_path: str) -> Dict:
  ''' 
  Load configuration file

  :param config_path: path to yaml file
  :returns: dictionary of parameters
  '''
  with open(config_path, "r") as config_file:
      config = yaml.safe_load(config_file)
  return config


def update_config(names: tuple, values: list) -> Dict:
  '''
  Update the config with a specific hyperparameter setup

  :params names: hypereparameter names
  :params values: values to set
  '''
  for i, name in enumerate(names):
    if name=='lr':
      config['Model']['optim']['lr'] = values[i]
    elif name=='optim':
      config['Model']['optim']['name'] = values[i]
    else:
      config['Model'][name] = values[i]
  
  return config


def save_updated_config(updated_config, temp_config_path):
  ''' 
  Temporarily save config file fo hyperparam config

  :params updated_config: new config dict
  :param temp_config_path: path
  '''
  with open(temp_config_path, 'w') as temp_config_file:
      yaml.dump(updated_config, temp_config_file)
  return temp_config_path


def extract_scores_from_output(output_lines, hyperparam_grid, hyperparam_values, dataset):
  """
  Extract scores printed from running test.py

  :param output_lines: list of strings (printed lines)
  :param hyperparam_grid: dict with hyperparameter names as keys and lists of values as values
  :param hyperparam_values: list of hyperparameter values
  :param dataset: str. Test or Val set
  :returns: pd.DataFrame with 1 row and columns are hyperparams and scores
  """
  
  # Extract the test score from the output
  seed_lines = [line for line in output_lines if 'seed' in line]

  seeds = []
  rmses = []
  maes = []
  r2s = []
  slopes = []
  ints = []
  rmselows = []
  pearsons = []

  for i, seed_line in enumerate(seed_lines):
    seed = int(seed_line.split('seed')[-1])
    rmse = float([line for line in output_lines if 'RMSE' in line][i].split(': ')[1]) if len([line for line in output_lines if 'RMSE' in line]) else np.nan
    mae = float([line for line in output_lines if 'MAE' in line][i].split(': ')[1]) if len([line for line in output_lines if 'MAE' in line]) else np.nan
    r2 = float([line for line in output_lines if 'R2' in line][i].split(': ')[1]) if len([line for line in output_lines if 'R2' in line]) else np.nan
    slope = float([line for line in output_lines if 'Regression slope' in line][i].split(': ')[1]) if len([line for line in output_lines if 'Regression slope' in line]) else np.nan
    inter = float([line for line in output_lines if 'Regression intercept' in line][i].split(': ')[1]) if len([line for line in output_lines if 'Regression intercept' in line]) else np.nan
    rmselow = float([line for line in output_lines if 'rmselow' in line][i].split(': ')[1]) if len([line for line in output_lines if 'rmselow' in line]) else np.nan
    pearson = float([line for line in output_lines if 'Pearson' in line][i].split(': ')[1]) if len([line for line in output_lines if 'Pearson' in line]) else np.nan

    seeds.append(seed)
    rmses.append(rmse)
    maes.append(mae)
    r2s.append(r2)
    slopes.append(slope)
    ints.append(inter)
    rmselows.append(rmselow)
    pearsons.append(pearson)

  score_data = {
      **{h: hyperparam_values[i] for i, h in enumerate(hyperparam_grid.keys())},
      'Dataset': [dataset]*len(seeds),
      'Seed': seeds,
      'RMSE': rmses,
      'MAE': maes,
      'R2': r2s,
      'Slope': slopes,
      'Intercept': ints,
      'RMSElow': rmselows,
      'Pearson r2': pearsons
  }
  score_df = pd.DataFrame(score_data)

  return score_df


def tune_model(config: dict) -> None:
  ''' 
  Tune model based on grid of hyperparameter values

  :param config: dictionary of configuration parameters
  '''
  # Get hyperparam grid
  hyperparam_grid = config['Grid']
  hyperparam_combinations = list(itertools.product(*hyperparam_grid.values()))

  # Temporarily save the model and scaler to use during testing
  config['Model']['save_path'] = config['Model']['save_path'].split('.pkl')[0] + '_tmp.pkl'

  # Create a directory to store results
  results_dir = 'tuning_results/'
  os.makedirs(results_dir, exist_ok=True)

  # Open a CSV file to store the results
  results_file = os.path.join(results_dir, config['Model']['name'] + '_' + datetime.now().strftime("%Y%m%d") + '_tuning_LAI.xlsx')
  
  # Iterate over hyperparameter combinations
  for hyperparam_values in hyperparam_combinations:
    print('\n'.join([f'Training with hyperparameters ' + ', '.join([f'{key}: {value}' for key, value in zip(hyperparam_grid.keys(), hyperparam_values)])]))

    # Update the hyperparameters in the config
    config = update_config(hyperparam_grid.keys(), hyperparam_values)
    # Prepare for test
    val_data_path = config['Data']['val_data_path']
    config['Data']['val_data_path'] = config['Data']['test_data_path']
    config['Model']['score_path'] = None

    # Save the updated config to a temporary file
    temp_config_path = os.path.join(results_dir, 'temp_config.yaml')
    temp_config_path = save_updated_config(config, temp_config_path)

    ###########
    # TRAIN

    # Call train.py with the updated config
    train_cmd = ['python', 'train.py', temp_config_path]
    process = run(train_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    ###########
    # TEST

    # Call test.py (to test on TEST set) with the updated config
    test_cmd = ['python', 'test.py', temp_config_path]
    process = run(test_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)


    score_df = extract_scores_from_output(process.stdout.split('\n'), hyperparam_grid, hyperparam_values, 'Test')

    if results_file is not None:
      if os.path.exists(results_file):
          existing_df = pd.read_excel(results_file)
          updated_df = pd.concat([existing_df, score_df], ignore_index=True)
      else:
          updated_df = score_df
      updated_df.to_excel(results_file, index=False)
  
  
    ###########
    # VALIDATION

    # Prepare for validation
    config['Data']['val_data_path'] = val_data_path
    temp_config_path = save_updated_config(config, temp_config_path)

    # Call test.py (to test on VALIDATION set) with the updated config
    test_cmd = ['python', 'test.py', temp_config_path]
    process = run(test_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    score_df = extract_scores_from_output(process.stdout.split('\n'), hyperparam_grid, hyperparam_values, 'Val')

    if results_file is not None:
      if os.path.exists(results_file):
          existing_df = pd.read_excel(results_file)
          updated_df = pd.concat([existing_df, score_df], ignore_index=True)
      else:
          updated_df = score_df
      updated_df.to_excel(results_file, index=False)

    # Delete the temporary saved model and scaler
    for seed in config['Seed']:
      os.remove(config['Model']['save_path'].split('.')[0] + f'{seed}.pkl')
      os.remove(config['Model']['save_path'].split('.')[0] + f'{seed}_scaler.pkl')


  # Delete the temporary config file
  os.remove(temp_config_path)

  print(f'Tuning results saved in {results_file}')
  return


def extract_scores_trial(output_lines, hyperparam_names, hyperparam_values, dataset):
  """
  Extract scores printed from running test.py

  :param output_lines: list of strings (printed lines)
  :param hyperparam_grid: dict with hyperparameter names as keys and lists of values as values
  :param hyperparam_values: list of hyperparameter values
  :param dataset: str. Test or Val set
  :returns: pd.DataFrame with 1 row and columns are hyperparams and scores
  """
  
  # Extract the test score from the output
  seed_lines = [line for line in output_lines if 'seed' in line]

  seeds = []
  rmses = []
  maes = []
  r2s = []
  slopes = []
  ints = []
  rmselows = []
  pearsons = []

  for i, seed_line in enumerate(seed_lines):
    seed = int(seed_line.split('seed')[-1])
    rmse = float([line for line in output_lines if f'{dataset} {seed} RMSE' in line][0].split(': ')[1]) if len([line for line in output_lines if f'{dataset} {seed} RMSE' in line]) else np.nan
    mae = float([line for line in output_lines if f'{dataset} {seed} MAE' in line][0].split(': ')[1]) if len([line for line in output_lines if f'{dataset} {seed} MAE' in line]) else np.nan
    r2 = float([line for line in output_lines if f'{dataset} {seed} R2' in line][0].split(': ')[1]) if len([line for line in output_lines if f'{dataset} {seed} R2' in line]) else np.nan
    slope = float([line for line in output_lines if 'Regression slope' in line][i].split(': ')[1]) if len([line for line in output_lines if 'Regression slope' in line]) else np.nan
    inter = float([line for line in output_lines if 'Regression intercept' in line][i].split(': ')[1]) if len([line for line in output_lines if 'Regression intercept' in line]) else np.nan
    rmselow = float([line for line in output_lines if 'rmselow' in line][i].split(': ')[1]) if len([line for line in output_lines if 'rmselow' in line]) else np.nan
    pearson = float([line for line in output_lines if 'Pearson' in line][i].split(': ')[1]) if len([line for line in output_lines if 'Pearson' in line]) else np.nan

    seeds.append(seed)
    rmses.append(rmse)
    maes.append(mae)
    r2s.append(r2)
    slopes.append(slope)
    ints.append(inter)
    rmselows.append(rmselow)
    pearsons.append(pearson)

  score_data = {
      **{h: hyperparam_values[i] for i, h in enumerate(hyperparam_names)},
      'Dataset': [dataset]*len(seeds),
      'Seed': seeds,
      'RMSE': rmses,
      'MAE': maes,
      'R2': r2s,
      'Slope': slopes,
      'Intercept': ints,
      'RMSElow': rmselows,
      'Pearson r2': pearsons
  }
  score_df = pd.DataFrame(score_data)

  return score_df


def objective(trial, config, results_dir, results_file):
    """Objective function for Optuna to optimize hyperparameters."""
    
    ###########
    # HYPERPARAMS 
    # Suggest hyperparameters
    config['Model']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    #config['Model']['hidden_layer'] = trial.suggest_int('hidden_layer', 1, 3)
    config['Model']['hidden_size'] = trial.suggest_int('hidden_size', 32, 64, step=16)
    config['Model']['optim']['name'] = trial.suggest_categorical('optim', ['Adam', 'SGD'])  
    config['Model']['optim']['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # trial.suggest_categorical("lr", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]) #
    config['Model']['scheduler']['name'] = trial.suggest_categorical('lr_scheduler', ['ReduceLROnPlateau', 'CosineAnnealingLR'])
    if config['Model']['optim']['name'] =='SGD':
      config['Model']['optim']['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
    else:
      config['Model']['optim'].pop('momentum') if 'momentum' in config['Model']['optim'].keys() else None

    config['Model']['score_path'] = None # don't save individual trials results in excel
    
    # Save the updated config to a temporary file
    temp_config_path = os.path.join(results_dir, 'temp_config.yaml')
    temp_config_path = save_updated_config(config, temp_config_path)

    ###########
    # TRAIN (and test on synthetic data)
    # Call train.py with the updated config
    train_cmd = ['python', 'train.py', temp_config_path]
    process = run(train_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    ###########
    # SCORE
    hyperparam_names = ['batch_size', 'hidden_layer', 'hidden_size', 'optim', 'lr', 'scheduler']
    hyperparam_values = [config['Model']['batch_size'], config['Model']['hidden_layer'], config['Model']['hidden_size'],\
       config['Model']['optim']['name'], config['Model']['optim']['lr'], config['Model']['scheduler']['name']]
    
    score_df_train = extract_scores_trial(process.stdout.split('\n'), hyperparam_names, hyperparam_values, 'Train')
    score_df_test = extract_scores_trial(process.stdout.split('\n'), hyperparam_names, hyperparam_values, 'Test')
    test_rmse = score_df_test['RMSE'].mean()


    if results_file is not None:
      if os.path.exists(results_file):
          existing_df = pd.read_excel(results_file)
          updated_df = pd.concat([existing_df, score_df_train, score_df_test], ignore_index=True)
      else:
          updated_df = pd.concat([score_df_train, score_df_test], ignore_index=True)
      updated_df.to_excel(results_file, index=False)

    # Delete the temporary config file
    os.remove(temp_config_path)

    print(f'Tuning results saved in {results_file}')
    
    return test_rmse  # Optuna will minimize/maximize this value


def tune_model_optuna(config):
  ''' 
  Tune model using Optuna (Bayesian optimisation)

  :param config: dictionary of configuration parameters
  '''
  study = optuna.create_study(direction="minimize")
  n_trials = config['Tuning']['n_trials']
  # File where to save results
  results_dir = 'tuning_results/'
  os.makedirs(results_dir, exist_ok=True)
  results_file = os.path.join(results_dir, config['Model']['score_path'].split('/')[-1].split('.xlsx')[0] + '_tuning.xlsx')
  study.optimize(lambda trial: objective(trial, config, results_dir, results_file), n_trials=n_trials)

  return study



if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'
  parser = ArgumentParser()
  parser.add_argument('tune_settings', type = str, metavar='path/to/setting.yaml', help='yaml with settings for hyperparam tuning')
  args = parser.parse_args()

  config = load_config(args.tune_settings)
  tune_model_optuna(config)
