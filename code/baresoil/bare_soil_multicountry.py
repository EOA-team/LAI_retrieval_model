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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d, pchip_interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import time
import glob
import contextily as ctx
import calendar
import rasterio
import joblib
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from scipy import stats
import seaborn as sns

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


def find_optimal_clusters(data, max_clusters=7):
    silhouette_scores = []
    wcss = []

    print('Seaching for optimal clusters')
    for i in range(3, max_clusters + 1):
        print(f'Fitting K-means with {i} clusters')
        #kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans = MiniBatchKMeans(
            n_clusters=i,
            random_state=42,
            batch_size=10000,
            n_init='auto'
        )
        kmeans.fit(data)
        labels = kmeans.labels_

        # Compute metrics
        silhouette_scores.append(silhouette_score(data, labels, sample_size=10000))
        wcss.append(kmeans.inertia_)  # WCSS (sum of squared distances to cluster centers)


    print(silhouette_scores)
    print(wcss)
    # Find the optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 3  # Adding 3 due to starting from 3 clusters
    print(f'Optimal number of clusters (silhouette): {optimal_clusters}')
  
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig(f'kmeans_silhouette_countries.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(range(3, max_clusters + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(f'kmeans_elbow_countries.png')

    return optimal_clusters


def sample_spectra_old(data, percentiles, n, save_path):
    if percentiles is None:
        # Sample uniformly n per cluster
        sampled_data = []
        for cluster_id in sorted(data['cluster'].unique()):
            cluster_data = data[data['cluster'] == cluster_id]
            sampled_cluster = cluster_data.sample(n=min(n, len(cluster_data)), random_state=42)
            sampled_data.append(sampled_cluster)
        representative_soils = pd.concat(sampled_data).reset_index(drop=True)
        representative_soils['x'] = representative_soils.geometry.x
        representative_soils['y'] = representative_soils.geometry.y
    
    else:
        # Compute percentiles per band within cluster
        bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
        records = []
        for cluster_id in sorted(data['cluster'].unique()):
            cluster_data = data[data['cluster'] == cluster_id][bands]
            for p in percentiles:
                spectrum = cluster_data.quantile(p / 100.0)
                records.append({
                    'cluster': cluster_id,
                    'percentile': p,
                    **spectrum.to_dict()
                })
        representative_soils = pd.DataFrame(records)

    representative_soils.to_csv(save_path, index=False)

    return 
    

def sample_spectra(gdf, n, k, method, save_path):

    sampled_data = []

    if method == 'proportional':
        # Sample size changes based on cluster size
        for cluster_id, group in gdf.groupby('cluster'):
            n_sample = int(len(group) / len(gdf) * n)
            n_sample = max(1, min(n_sample, len(group)))
            sampled_cluster = group.sample(n=n_sample, random_state=42)
            sampled_data.append(sampled_cluster)
    if method == 'uniform':
        # Sample the same number in each cluster
        n_sample = int(n/len(gdf['cluster'].unique()))
        for cluster_id, group in gdf.groupby('cluster'):
            n_sample_cluster = max(1, min(n_sample, len(group)))
            sampled_cluster = group.sample(n=n_sample_cluster, random_state=42)
            sampled_data.append(sampled_cluster)

    sampled = pd.concat(sampled_data)
    sampled.to_csv(save_path, index=False)

    # Plot these across europe
    fig, ax = plt.subplots(figsize=(10, 10))
    colors =['teal', 'orange', 'purple', 'palevioletred', 'limegreen', 'dodgerblue'][:k] # adapt cmap in function of nbr of clusters
    cluster_labels = sorted(sampled['cluster'].unique())
    cluster_color_map = dict(zip(cluster_labels, colors))
    sampled['color'] = sampled['cluster'].map(cluster_color_map)
    #custom_cmap = ListedColormap(colors)
    #sampled['cluster'] = sampled['cluster'].astype('category')
    sampled.to_crs(4326).plot(ax=ax, markersize=5, color=sampled['color'], legend=True)
    handles = [mpatches.Patch(color=cluster_color_map[c], label=str(c)) for c in cluster_labels]
    ax.legend(handles=handles, title='Soil group', title_fontsize=14, fontsize=14)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ctx.add_basemap(ax, crs=4326)
    plt.savefig(f'sampled_locations_k{k}_n{n}_{method}_countries.png')

    return sampled



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
                '0046-0018', '0048-0018', '0050-0018', \
                 '0048-0016'], #mainland only
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
    countries = ['Switzerland', 'Bulgaria', 'Italy'] #['Poland']
    """
    if not os.path.exists('gdf_src_sampled.pkl'):

        country_file_sizes = {c: os.path.getsize(f"gdf_src_{c.lower()}_arable.pkl") for c in countries}
        total_size = sum(country_file_sizes.values())

        total_samples = 250000  
        sample_sizes = {c: int(total_samples * (country_file_sizes[c] / total_size)) for c in countries}
        print("Sample sizes (based on file size):", sample_sizes)
        
        gdf_src_sampled = []
        for country in countries:
            print('Loading for', country)

            cols = ['MASK', 'SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']
            epsg = metadata[country.lower()]['epsg']

            # Open SRC for arable land
            gdf_src = pd.read_pickle(f'gdf_src_{country.lower()}_arable.pkl')

            # Subsample
            n_samples = min(sample_sizes[country], len(gdf_src))
            gdf_src = gdf_src.sample(n=n_samples, random_state=42)

            geom = [Point(xy) for xy in zip(gdf_src['x'], gdf_src['y'])]
            gdf_src = gpd.GeoDataFrame(gdf_src, geometry=geom, crs='EPSG:3035')
            gdf_src.rename(columns=lambda c: c.replace("SRC_", "") if c.startswith("SRC_") else c, inplace=True)
            gdf_src.rename(columns=lambda c: c.replace("B", "B0") if len(c)==2 and c.startswith("B") else c, inplace=True)

            gdf_src_sampled.append(gdf_src)

        gdf_src_sampled = pd.concat(gdf_src_sampled, ignore_index=True)
        gdf_src_sampled.to_pickle('gdf_src_sampled.pkl')

    else:
        gdf_src_sampled = pd.read_pickle('gdf_src_sampled.pkl')
        geom = [Point(xy) for xy in zip(gdf_src_sampled['x'], gdf_src_sampled['y'])]
        gdf_src_sampled = gpd.GeoDataFrame(gdf_src_sampled, geometry=geom, crs='EPSG:3035')

    # Clean the data
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    mask = (gdf_src_sampled[bands] >= 0).all(axis=1) & (gdf_src_sampled[bands] <= 10000).all(axis=1) # out of range
    gdf_src_sampled = gdf_src_sampled[mask]
    gdf_src_sampled = gdf_src_sampled[gdf_src_sampled[bands].mean(axis=1) > 100]  # drop near-zero spectra
    gdf_src_sampled = gdf_src_sampled[gdf_src_sampled['B08'] > gdf_src_sampled['B04']] # drop unrealistic ratios
    z_scores = stats.zscore(gdf_src_sampled[bands], axis=0)
    gdf_src_sampled = gdf_src_sampled[(z_scores < 3).all(axis=1)]  # remove extreme outliers

    # Convert to reflectance
    gdf_src_sampled[[c for c in gdf_src_sampled.columns if c.startswith('B')]] /= 10000

    # Fit Kmeans at different k (3 to 7)
    #optimal_clusters = find_optimal_clusters(gdf_src_sampled[['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']].values, max_clusters=10)

    # Select K and train final model
    k=5
    if not os.path.exists(f'kmeans_soil_k{k}_countries.pkl'):
        print(f'Fitting K-means with {k} clusters')
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=10000)
        kmeans.fit(gdf_src_sampled[['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']].values)
        joblib.dump(kmeans, f'kmeans_soil_k{k}_countries.pkl')
    else:
        kmeans = joblib.load(f'kmeans_soil_k{k}_countries.pkl')

    # Predict soil group for all data points
    labels = np.empty(len(gdf_src_sampled), dtype=np.int32)
    features = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    chunk_size = 100000
    for start in range(0, len(gdf_src_sampled), chunk_size):
        end = min(start + chunk_size, len(gdf_src_sampled))
        X_chunk = gdf_src_sampled.iloc[start:end][features].values
        labels[start:end] = kmeans.predict(X_chunk)
    gdf_src_sampled['cluster'] = labels

    # Update cluster names (1-5 instead of 0-4)
    gdf_src_sampled['cluster'] = gdf_src_sampled['cluster'].apply(lambda x:x+1)
    """
    # Sample and plot locations
    k=5
    n = 1000
    method = 'uniform'
    #sampled = sample_spectra(gdf_src_sampled, n, k, method=method, save_path=f'soil_spectra_k{k}_n{n}_{method}_countries.csv')
    
    # Plot spectra
    df = pd.read_csv(f'soil_spectra_k{k}_n{n}_{method}_countries.csv').drop('geometry', axis=1)
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    s2_clustering = [492, 560, 665, 704, 740, 781, 833, 864, 1612, 2194]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['teal', 'orange', 'purple', 'palevioletred', 'limegreen', 'dodgerblue'][:k]
    cluster_labels = sorted(df['cluster'].unique())
    cluster_color_map = dict(zip(cluster_labels, colors))
    df['color'] = df['cluster'].map(cluster_color_map)

    for cluster_id, group in df.groupby('cluster'):
        for _, row in group.iterrows():
            ax.plot(s2_clustering, row[bands], label=f'{cluster_id}', color=group['color'].values[0], linewidth=0.5)

    # Make legend unique and clearer
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Soil group')

    #ax.set_title('Sampled soil spectra')
    ax.set_ylabel('Reflectance', fontsize=18)
    ax.set_xlabel('Band', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make legend unique and clearer
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title='Soil group',
        fontsize=16,         # ← legend label font size
        title_fontsize=16    # ← legend title font size
    )
    for line in legend.get_lines():
        line.set_linewidth(2)  # adjust thickness as needed

    plt.tight_layout()
    #plt.savefig(f'sampled_spectra_k{k}_n{n}_{method}_countries.png')


    # Upsample spectra to 1nm resolution
    s2_all = [492, 560, 665, 704, 740, 781, 833, 864, 1612, 2194]
    spectra = upsample_spectra(df[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']], s2_all, np.arange(400, 2501, 1), 'pchip')
    #spectra.to_csv(f'sampled_spectra_k{k}_n{n}_{method}_1nm.csv', index=False)
    #spectra.to_pickle(f'sampled_spectra_k{k}_n{n}_{method}_1nm.pkl')
    
    # Plot
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
    #plt.savefig(f'sampled_spectra_k{k}_n{n}_{method}_1nm.png')

    # Plot hued by soil group
    spectra['cluster'] =  df['cluster']
    spectra['color'] = spectra['cluster'].map(cluster_color_map)
    spectra['spectrum_id'] = spectra.index.astype(str)
    spectra['soil_group'] = spectra['cluster']
    wavelength_cols = [c for c in spectra.columns if isinstance(c, int) or (isinstance(c, str) and c.isdigit())]
    spectra_long = spectra.melt(
        id_vars=['spectrum_id', 'soil_group'],
        value_vars=wavelength_cols,
        var_name='nm',
        value_name='Reflectance'
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=spectra_long, x='nm', y='Reflectance', hue='soil_group', units='spectrum_id', hue_order=[3,2,5,1,4], 
                estimator=None, palette=cluster_color_map, ax=ax, alpha=0.5, linewidth=0.7)
    ax.set_ylim(0, 0.5)
    #leg = ax.legend(title='Soil Group', fontsize=16, title_fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    labels_int = [int(l) for l in labels]
    sorted_pairs = sorted(zip(labels_int, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    leg = ax.legend(
        sorted_handles,
        sorted_labels,
        title='Soil Group',
        fontsize=16,
        title_fontsize=16
    )
    for line in leg.get_lines():
        line.set_linewidth(3)
    ax.set_xlabel('Wavelength [nm]', fontsize=18)
    ax.set_ylabel('Reflectance', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'sampled_spectra_k{k}_n{n}_{method}_1nm_soilgroup.png')
