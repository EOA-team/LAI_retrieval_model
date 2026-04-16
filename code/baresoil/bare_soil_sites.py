'''
Extract bare soil spectra from S2 data

@Selene Ledain
'''
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
from typing import List
from shapely.geometry import Polygon, MultiPolygon, Point
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d, pchip_interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import time
import glob
import contextily as ctx
import calendar
import rasterio
import joblib
import xarray as xr
import rioxarray

import os
from pathlib import Path
import sys
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent


def upsample_spectra(df, wavelengths, new_wavelengths, method):
    '''
    Upsample spectra from a sensor

    :param df: dataframe containing pixel spectra all oroginaitng from one sensor
    :param wavelengths: sensor wavelengths
    :param new_wavelengths: wavelengths to upsample to    
    :param method: inteprolation method among ['spline', 'pchip']

    :returns: same dataframe but upsampled to new_wavelengths
    '''
    if method == 'spline':     
      df.insert(0, '400', df.apply(lambda row: row.min(), axis=1)) # Bound the values for the start of the spectra
      df.loc[:, '2500'] = df.apply(lambda row: row.min(), axis=1) # Bound the values for the end of the spectra
      f = interp1d([400] + wavelengths + [2500], df.values, kind='cubic', fill_value="extrapolate")
      interpolated_values = f(new_wavelengths)
      interpolated_df = pd.DataFrame(interpolated_values, columns=new_wavelengths, index=df.index)


    if method == 'pchip':
      df.insert(0, '400', df.apply(lambda row: row.min(), axis=1)) # Bound the values for the start of the spectra
      interpolated_values = pchip_interpolate([400] + wavelengths, df.values.T, new_wavelengths).T
      interpolated_df = pd.DataFrame(interpolated_values, index=df.index, columns=new_wavelengths)


    if method == 'combined':
      # First part of spectra with spline, second with pchip
      df.insert(0, '400', df.apply(lambda row: row.min(), axis=1)) # Bound the values for the start of the spectra
      spline_cols = ['400', 'B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A']
      f = interp1d([400] + wavelengths[:-2], df[spline_cols].values, kind='cubic', fill_value="extrapolate")
      interpolated_values_spline = f(new_wavelengths[:865-400])
      interpolated_df_spline = pd.DataFrame(interpolated_values_spline, columns=new_wavelengths[:865-400], index=df[spline_cols].index)
      
      pchip_cols = ['B10', 'B11', 'B12']
      interpolated_values_pchip = pchip_interpolate(wavelengths[-2:], df[pchip_cols].values.T, new_wavelengths[865-400:]).T
      interpolated_df_pchip = pd.DataFrame(interpolated_values_pchip, index=df[pchip_cols].index, columns=new_wavelengths[865-400:])

      interpolated_df = pd.concat([interpolated_df_spline, interpolated_df_pchip], axis=1)

    return interpolated_df



if __name__ == '__main__':


    data_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/DLR_soilsuite')
    bounds_data =  os.path.expanduser('~/mnt/eo-nas1/data/units-bundaries_administrative/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

    metadata = {
        'switzerland': {
            'tiles': ['0040-0026', '0042-0026', '0040-0024', '0042-0024'],
            'valdata': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_S2/s2_val_destructive_licor.pkl'),
            'valsites': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/valsites_CH'),
            'epsg':32632}, #epsg of valdata
        'bulgaria': {
            'tiles': ['0054-0022', '0056-0022'],
            'valdata': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_S2/s2_val_bulgaria.pkl'),
            'valsites': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/Bulgaria_LAI/fields_ALL_34N'),
            'epsg':32634},
        'italy': {
            'tiles': ['0040-0024', '0042-0024', '0044-0024',\
                '0040-0022', '0042-0022', '0044-0022', '0046-0022',\
                '0044-0020', '0046-0020', '0048-0020',\
                '0046-0018', '0048-018', '0050-0018', \
                '0046-0016', '0048-16'],
            'valdata': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_S2/s2_val_italy.pkl'),
            'valsites': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/SENSAGRI_LAIG_data/'),# in epsg 4326
            'epsg': 4326}, 
        'poland': {
            'tiles':  ['0046-0034', '0048-0034', '0050-0034',\
                '0046-0032', '0048-0032', '0050-0032', '0052-0032',\
                '0048-0030', '0050-0030', '0052-0030'],
            'valdata': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/insitu_S2/s2_val_poland.pkl'),
            'valsites': os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/LAI_paper/data/SENSAGRI_LAIG_data/'), # in epsg 4326
            'epsg': 4326}
    }
    
    # Open data and clip to country
    country = 'Italy'
    

    tile_codes = metadata[country.lower()]['tiles']
    data_files = [os.path.join(data_dir, f'soilsuite_{tile}.zarr') for tile in tile_codes] 
    cols = ['MASK', 'SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']
    epsg = metadata[country.lower()]['epsg']

    if not os.path.exists(f'gdf_src_{country.lower()}_arable.pkl'):
        country_bounds = gpd.read_file(bounds_data).to_crs(3035)
        country_bounds = country_bounds[country_bounds['SOVEREIGNT']==country]
        
        src = []
        for f in data_files:
            print(f'Loading data from {f}...')
            ds_tile = xr.open_zarr(f) # EPSG 3035, 20m 
            # Clip to bounds of country
            ds_tile = ds_tile.rio.write_crs(3035).rio.set_spatial_dims(x_dim='x', y_dim='y').rio.clip(country_bounds.geometry, drop=True)
            # Filter for arable land
            df_tile = ds_tile[cols].to_dataframe().reset_index()
            geometry = [Point(xy) for xy in zip(df_tile['x'], df_tile['y'])]
            gdf_tile = gpd.GeoDataFrame(df_tile, geometry=geometry, crs='EPSG:3035')
            gdf_corine = gpd.read_file('U2018_CLC2018_V2020_20u1.gpkg', layer='U2018_CLC2018_V2020_20u1')
            gdf_corine = gdf_corine[gdf_corine['Code_18'].isin(['211','212','213','221','222','223','231','241','242','243','244'])]
            gdf_tile = gpd.sjoin(gdf_tile, gdf_corine, how='inner')
            # Remove Nan
            gdf_tile[gdf_tile==-10000] = np.nan
            gdf_tile[gdf_tile==-10] = np.nan
            gdf_tile = gdf_tile.dropna(subset=cols)
            if country=='Italy':
                gdf_tile.to_pickle(f'SRC_{country.lower()}_{os.path.basename(f).split(".zarr")[0]}.pkl')
            else:
                src.append(gdf_tile)

        if country=='Italy':
            gdf_tile = None # free up memory
            data_files = [f'SRC_{country.lower()}_soilsuite_{tile}.pkl' for tile in tile_codes] 
            for f in data_files:
                df = pd.read_pickle(f)
                src.append(df)
                print('appended', os.path.basename(f))

        df_src = pd.concat(src, ignore_index=True)
        df_src.to_pickle(f'gdf_src_{country.lower()}_arable.pkl')
    
    else:
        df_src = pd.read_pickle(f'gdf_src_{country.lower()}_arable.pkl')
        df_src.drop('index_right', axis=1, inplace=True)
    
    print('loaded df_src')
    # Turn into gdf
    geometry = [Point(xy) for xy in zip(df_src['x'], df_src['y'])]
    gdf_src = gpd.GeoDataFrame(df_src, geometry=geometry, crs='EPSG:3035')

    # Keep only where SCR classified as soil
    gdf_src = gdf_src[gdf_src['MASK']==1]
    
    # Intersect with fields to find bare soil samples in fields
    val_data = pd.read_pickle(metadata[country.lower()]['valdata'])
    geometry = [Point(xy) for xy in zip(val_data['lon'], val_data['lat'])]
    val_data = gpd.GeoDataFrame(val_data, geometry=geometry, crs=epsg)
    
    if country.lower() in ['italy', 'poland', 'switzerland']:
        field_name_col = 'name'
    if country.lower() == 'bulgaria':
        field_name_col = 'OBJECTID'
   
    all_fields = []
    for f in os.listdir(metadata[country.lower()]['valsites']):
        if f.endswith('.shp'):
            val_field = gpd.read_file(os.path.join(metadata[country.lower()]['valsites'], f)).to_crs(epsg)
            # Add a buffer around field edge
            if val_field.crs.is_geographic:
                buffer_dist = -10 / 111_000  
            else:
                buffer_dist = -10
            val_field["geometry"] = val_field.geometry.buffer(buffer_dist)
            all_fields.append(val_field)
            print('added field', f)

    all_fields = pd.concat(all_fields, ignore_index=True)
    val_fields = gpd.sjoin(all_fields, val_data, how='inner', predicate='intersects').drop_duplicates(subset=field_name_col)

    # Filter BS data for validation plots 
    gdf_BS_val = gpd.sjoin(gdf_src.to_crs(epsg), val_fields[['geometry', field_name_col]], how='inner', predicate='intersects')
  
    # Convert to reflectance
    gdf_BS_val[[c for c in gdf_BS_val.columns if 'SRC_B' in c]] /= 10000

    # Rename
    gdf_BS_val.rename(columns=lambda c: c.replace("SRC_", "") if c.startswith("SRC_") else c, inplace=True)
    gdf_BS_val.rename(columns=lambda c: c.replace("B", "B0") if len(c)==2 and c.startswith("B") else c, inplace=True)
    gdf_BS_val['epsg'] = 3035
    
    # Save per field   
    cols = ['y','x','MASK','B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12', 'epsg']
    fields = gdf_BS_val[field_name_col].unique()
    for field in fields:
        print('saving', field)
        field_spectra = gdf_BS_val[gdf_BS_val[field_name_col]==field]
        field_spectra[cols].to_csv(f'{country.lower()}_fields/sampled_soil_spectra_{country}_{field}.csv', index=False)

    # Plot spectra
    wvl = [490,560,665,705,740,783,842,865,1610,2190]
    bands = ['B02', 'B03','B04','B05','B06','B07','B08', 'B8A', 'B11','B12']
    plot_names = gdf_BS_val[field_name_col].unique()
    n_rows = len(plot_names)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5*n_rows))
    for i, plot_name in enumerate(plot_names):
        group = gdf_BS_val[gdf_BS_val[field_name_col] == plot_name]
        for _, row in group.iterrows():
            axes[i].plot(wvl, row[bands].values, alpha=0.5, color='gray')
        mean_spec = group[bands].mean().values
        axes[i].plot(wvl, mean_spec, color='black', lw=2, label='Mean spectrum')
        axes[i].set_xlabel('Band', fontsize=18)
        axes[i].set_ylabel('Reflectance', fontsize=18)
        axes[i].set_ylim(0, 0.5)
        axes[i].set_title(f'Sampled soil spectra: {plot_name}', fontsize=18)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'{country.lower()}_fields/field_soils_{country.lower()}.png')
    

    """
    # Plot locations and spectra
    wvl = [490,560,665,705,740,783,842,865,1610,2190]
    bands = ['B02', 'B03','B04','B05','B06','B07','B08', 'B8A', 'B11','B12']
    plot_names = gdf_BS_val[field_name_col].unique()
    n_rows = len(plot_names)
    n_cols = 2 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 5*n_rows), width_ratios=[1,1.3])

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = np.array([axes])  # shape (1, n_cols)

    for i, plot_name in enumerate(plot_names):
        group = gdf_BS_val[gdf_BS_val[field_name_col] == plot_name]
        colors = ['tab:orange', 'tab:green', 'tab:blue'] #plt.get_cmap('Set2', len(group))
        for j, (_, row) in enumerate(group.iterrows()):
            row_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf_BS_sampled.crs)
            row_gdf.plot(ax=axes[i, 0], color=colors[j], markersize=70)
        axes[i, 0].set_title(f'Soil spectra locations: {plot_name}', fontsize=18)
        axes[i, 0].set_xlabel('Lon', fontsize=18)
        axes[i, 0].set_ylabel('Lat', fontsize=18)
        xmin, ymin, xmax, ymax = group.total_bounds
        buffer = 400 if epsg!=4326 else 0.005
        axes[i, 0].set_xlim(xmin-buffer, xmax+buffer)
        axes[i, 0].set_ylim(ymin-buffer, ymax+buffer)
        axes[i, 0].tick_params(axis='both', which='major', labelsize=16)
        ctx.add_basemap(ax=axes[i, 0], crs=gdf_BS_sampled.crs)

        for j, (_, row) in enumerate(group.iterrows()):
            axes[i, 1].plot(wvl, row[bands], color=colors[j])
        axes[i, 1].set_xlabel('Band', fontsize=18)
        axes[i, 1].set_ylabel('Reflectance', fontsize=18)
        axes[i, 1].set_ylim(0, 0.5)
        axes[i, 1].set_title(f'Sampled soil spectra: {plot_name}', fontsize=18)
        axes[i, 1].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(f'site_soils_{country.lower()}.png')
    """
    
    fields = gdf_BS_val[field_name_col].unique()
    for field in fields:
        # Upsample the spectra to 1nm and plot
        df = pd.read_csv(f'{country.lower()}_fields/sampled_soil_spectra_{country.lower()}_{field}.csv')
        geom = [Point(xy) for xy in zip(df['x'], df['y'])]
        df = gpd.GeoDataFrame(df, geometry=geom, crs=3035) # the x and y cols were extraced from src to_dataframe() before epsg change
        s2_all = [492, 560, 665, 704, 740, 781, 833, 864, 1612, 2194]
        spectra = upsample_spectra(df[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']], s2_all, np.arange(400, 2501, 1), 'pchip')
        spectra.to_csv(f'{country.lower()}_fields/sampled_soil_spectra_{country}_{field}_1nm.csv', index=False)
        spectra.to_pickle(f'{country.lower()}_fields/sampled_soil_spectra_{country.lower()}_{field}_1nm.pkl')

    fig, ax = plt.subplots(figsize=(12,8))
    spectra.T.plot(ax=ax, legend=False, color='grey') #, linewidth=0.5, alpha=0.8)

    mean_spectra = spectra.mean(axis=0)
    mean_spectra_df = mean_spectra.reset_index().rename(columns={0: 'Reflectance'})
    mean_spectra_df.columns = ['nm', 'Reflectance']
    mean_spectra_df.plot(ax=ax, x='nm', y='Reflectance', color='royalblue', linewidth=2.5, label='Mean Reflectance')

    plt.ylim((0,0.5))
    plt.ylabel('Reflectance', fontsize=20)
    plt.xlabel('Wavelength [nm]', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=[handles[-1]], labels=[labels[-1]], loc='upper right', fontsize=18)
    plt.savefig(f'{country.lower()}_fields/hyperspectral_{country.lower()}.png')


    n_rows = len(fields)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5*n_rows))
    for i, field in enumerate(fields):
        spectra = pd.read_csv(f'{country.lower()}_fields/sampled_soil_spectra_{country}_{field}_1nm.csv')
        spectra.T.plot(ax=axes[i], legend=False, color='grey') #, linewidth=0.5, alpha=0.8)
        mean_spectra = spectra.mean(axis=0)
        mean_spectra_df = mean_spectra.reset_index().rename(columns={0: 'Reflectance'})
        mean_spectra_df.columns = ['nm', 'Reflectance']
        mean_spectra_df.plot(ax=axes[i], x='nm', y='Reflectance', color='royalblue', linewidth=2.5, label='Mean Reflectance')
        axes[i].set_xlabel('Band', fontsize=18)
        axes[i].set_ylabel('Reflectance', fontsize=18)
        axes[i].set_ylim(0, 0.5)
        axes[i].set_title(f'Sampled soil spectra: {field}', fontsize=18)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend(handles[-1:], labels[-1:], loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{country.lower()}_fields/hyperspectral_{country.lower()}.png')

    """
    # Plot locations and spectra - HYPERSEPCTRAL
    df = df.merge(spectra, left_index=True, right_index=True)
    wvl = np.arange(400, 2501, 1)
    bands = np.arange(400, 2501, 1)
    plot_names = df[field_name_col].unique()
    n_rows = len(plot_names)
    n_cols = 2 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 5*n_rows), width_ratios=[1,1.3])

    if n_rows == 1:
        axes = np.array([axes])  # shape (1, n_cols)
    for i, plot_name in enumerate(plot_names):
        group = df[df[field_name_col] == plot_name]
        colors = ['tab:orange', 'tab:green', 'tab:blue'] #plt.get_cmap('Set2', len(group))
        for j, (_, row) in enumerate(group.iterrows()):
            row_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=df.crs)
            row_gdf.plot(ax=axes[i, 0], color=colors[j], markersize=70)
        #group.plot(ax=axes[i, 0], color='red', markersize=50)
        axes[i, 0].set_title(f'Soil spectra locations: {plot_name}', fontsize=18)
        axes[i, 0].set_xlabel('Lon', fontsize=18)
        axes[i, 0].set_ylabel('Lat', fontsize=18)
        xmin, ymin, xmax, ymax = group.total_bounds
        axes[i, 0].set_xlim(xmin-400, xmax+400)
        axes[i, 0].set_ylim(ymin-400, ymax+400)
        axes[i, 0].tick_params(axis='both', which='major', labelsize=16)
        ctx.add_basemap(ax=axes[i, 0], crs=df.crs, source=ctx.providers.OpenStreetMap.Mapnik)

        for j, (_, row) in enumerate(group.iterrows()):
            axes[i, 1].plot(wvl, row[bands], color=colors[j])
        axes[i, 1].set_xlabel('Band', fontsize=18)
        axes[i, 1].set_ylabel('Reflectance', fontsize=18)
        axes[i, 1].set_ylim(0, 0.5)
        axes[i, 1].set_title(f'Sampled soil spectra: {plot_name}', fontsize=18)
        axes[i, 1].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(f'site_soils_{country.lower()}_hyperspectral.png')
    """

    """
    # === Save each site seperately too ===
    df_spectra = pd.read_csv(f'sampled_soil_spectra_{country.lower()}.csv')
    df_spectra_hyp = pd.read_csv(f'sampled_soil_spectra_{country}_1nm.csv')

    if country.lower() in ['italy', 'poland', 'switzerland']:
        site_name_col = 'name'
    if country.lower() == 'bulgaria':
        site_name_col = 'OBJECTID'

    os.makedirs(f'{country.lower()}_sites', exist_ok=True)

    sites = df_spectra[site_name_col].unique()
    for s in sites:
        # S2 resolution
        site_spectra = df_spectra[df_spectra[site_name_col]==s]
        site_spectra.to_csv(f'{country.lower()}_sites/sampled_soil_spectra_{country}_{s}.csv', index=False)
        # Hyperspectral 
        idx = site_spectra.index
        site_spectr_hyp = df_spectra_hyp.iloc[idx]
        site_spectr_hyp.to_pickle(f'{country.lower()}_sites/sampled_soil_spectra_{country}_1nm_{s}.pkl')    
    """