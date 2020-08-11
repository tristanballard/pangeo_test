#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:51:59 2020

@author: tristanballard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:28:33 2020

@author: tristanballard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:41:19 2020

@author: tristanballard
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gcsfs # Not installed on geoexplorer docker
import xarray as xr # Not installed on geoexplorer docker
import zarr # Not installed on geoexplorer docker (pip install fails, but conda install works)
import intake # Not installed on geoexplorer docker
import time # May be natively installed on geoexplorer docker
import matplotlib.pyplot as plt

## Models used most recently:
# ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2-WACCM']
## Check for models that give 0's near coast. Or ones with abnormal low values. They have something off in units (CIESM I think).
variable_id = 'pr'
table_id = 'Amon' #Amon = Atmosphere monthly 
historical_experiment_id = 'historical'
future_experiment_id = 'ssp585'
start_year = 1980 
end_year = 2100
q_lbd = 0.15 # Quantile bounds for uncertainty analysis
q_ubd = 0.85
max_runs = 1 # Max runs to include per climate model. Most have 5 or fewer.

ENTITY_NAME = 'MRTG.A'
BASE_DATA_FOLDER = '../data/'
#plt.rcParams["figure.figsize"] = (9,4)

#### CUT Lines below on upload to docker
os.getcwd()
os.chdir('/Users/tristanballard/Sources/geoexplorer/jupyter-notebooks/')

gcs = gcsfs.GCSFileSystem(token='anon')

df_asset = pd.read_csv(BASE_DATA_FOLDER + 'sust/'+ ENTITY_NAME + '_combined_locations.csv')
#print(df_asset.head(3))
lats = xr.DataArray(df_asset['lat'], dims = 'z')
lons = xr.DataArray(df_asset['lng'], dims = 'z')

# Read in csv of currently available CMIP6 data on GCP
df_cmip = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_cmip.head()

# Filter out models by variable of interest and CMIP experiment
experiment_id = [historical_experiment_id, future_experiment_id]

df_cmip = df_cmip[(df_cmip['table_id'] == table_id) & 
                (df_cmip['variable_id'] == variable_id) & 
                (df_cmip['experiment_id'].isin(experiment_id)) &
                (df_cmip['grid_label'].isin(['gn', 'gr', 'gr1']))]
print(df_cmip.head(3))


run_counts = df_cmip.groupby(['source_id', 'experiment_id'])['member_id'].size() ; run_counts
print(run_counts)

# Count the number of model members (runs) available for each model and use only those models with both historical and future simulations available
## Filter for only those models with both historical *and* future simulations available
model_names = []
experiment_ids = [historical_experiment_id, future_experiment_id]
for name, group in df_cmip.groupby('source_id'):
    if all([expt in group.experiment_id.values
            for expt in experiment_ids]):
        model_names.append(name)
## In the case of essentially two of the same models from the same center, opt for newer of the two or the one with the higher resolution 
if ('MPI-ESM1-2-HR' in model_names) & ('MPI-ESM1-2-LR' in model_names):
    model_names.remove('MPI-ESM1-2-LR')
if ('CNRM-CM6-1-HR' in model_names) & ('CNRM-CM6-1' in model_names):
    model_names.remove('CNRM-CM6-1')
if ('INM-CM5-0' in model_names) & ('INM-CM4-8' in model_names):
    model_names.remove('INM-CM4-8')
if ('HadGEM3-GC31-LL' in model_names) & ('HadGEM3-GC31-MM' in model_names):
    model_names.remove('HadGEM3-GC31-LL')
if ('NorESM2-LM' in model_names) & ('NorESM2-MM' in model_names):
    model_names.remove('NorESM2-LM')
model_names.remove('CESM2')
model_names.remove('FGOALS-g3')
model_names.remove('MPI-ESM1-2-HR')
model_names.remove('AWI-CM-1-1-MR')
model_names.remove('UKESM1-0-LL')
model_names.remove('MIROC6')
model_names.remove('EC-Earth3-Veg')
n_models = len(model_names); n_models
print('Models available with both future and historical data: ', model_names, sep = '')




# Loop through each model 
#If a model has multiple members (i.e. r1i1p1, r2i2p2, ...) take the annual mean across the model members. 
time_slice = slice(str(start_year), str(end_year)) # Period of interest
pr_all_models_ls = [] # Each list element will be a np array (n_assets x year) for a particular model, converted to np array afterwards
start_time = time.time()
for model_name in model_names[6:11]:
    model_members = []
    ## Loop over individual members available for a SSP. A historical member 
    ## is only included if it has a corresponding SSP member.
    n_runs_include = np.min((np.min((run_counts[model_name, future_experiment_id], run_counts[model_name, historical_experiment_id])), max_runs))
    
    for run_index in range(n_runs_include):
        print(model_name, '.r', run_index+1, 'i1p1', sep = '')
        
        ## Read in historical data
        zstore = df_cmip[(df_cmip['source_id'] == model_name) & (df_cmip['experiment_id'] == historical_experiment_id)].zstore.values[run_index]
        # Create a mutable-mapping-style interface to the store
        mapper = gcs.get_mapper(zstore)
        # Open using xarray and zarr
        ds_historical = xr.open_zarr(mapper, consolidated = True)
        # Subset for time period of interest
        ds_historical = ds_historical.sel(time = time_slice)
        ds_historical = ds_historical.drop(['lat_bnds', 'lon_bnds', 'time_bnds'], errors = 'ignore')
  
        ## Read in future data
        zstore = df_cmip[(df_cmip['source_id'] == model_name) & (df_cmip['experiment_id'] == future_experiment_id)].zstore.values[run_index]
        # Create a mutable-mapping-style interface to the store
        mapper = gcs.get_mapper(zstore)
        # Open it using xarray and zarr
        ds_future = xr.open_zarr(mapper, consolidated = True)
        # Subset for time period of interest
        ds_future = ds_future.sel(time = time_slice)
        ds_future = ds_future.drop(['lat_bnds', 'lon_bnds', 'time_bnds'], errors = 'ignore')
      
        ## Merge the historical and future datasets
        ds_full = xr.concat([ds_historical, ds_future], dim = 'time')
     #   ds_full = ds_full.drop(['lat_bnds', 'lon_bnds', 'time_bnds'], errors = 'ignore')
        ## Rename lat/lon dim names and convert from 0,360 to -180,180 if necessary
        if ('longitude' in ds_full.dims) and ('latitude' in ds_full.dims):
            ds_full = ds_full.rename({'longitude':'lon', 'latitude': 'lat'}) 
        if (np.max(np.array(ds_full.coords['lon']))>300):
            ds_full = ds_full.assign_coords(lon = (((ds_full.lon + 180) % 360) - 180))
            ds_full = ds_full.sortby(ds_full.lon)
        if (np.max(np.array(ds_full.coords['lat']))>300):
            ds_full = ds_full.assign_coords(lat = (((ds_full.lat + 180) % 360) - 180))
            ds_full = ds_full.sortby(ds_full.lat)
              
        ## Convert from monthly to annual total precipitation
        ds_full_annual = ds_full.resample(time = 'AS', skipna = True).sum()
        
        ## Append this model's member data array to a list 
        model_members.append(ds_full_annual)
        del (zstore, mapper, ds_historical, ds_future, ds_full, ds_full_annual) 
        
    print(model_name, ' loading complete. Extracting asset location values.', sep = '')    
    ## Merge individual member data arrays from a model into a single array     
    ds_model_members = xr.concat(model_members, dim = 'member')
    ## Extract heatwave values at each asset's lat/lon coordinates
    # Data is lazily loaded before computation begins
    # Method = 'nearest' finds the nearest pixel to the coordinates
    pr_annual_lazy_load  = ds_model_members[variable_id].sel(
                                        lon = lons, 
                                        lat = lats, 
                                        method = 'nearest')   
    pr_annual = np.array(pr_annual_lazy_load) # Computationally heavy step
    pr_annual = np.transpose(pr_annual, (2,1,0)) # Reorder dimensions
    pr_annual.shape  # dim = n_assets x year x n_members
    
    ## Average the annual precip totals across model members
    pr_annual_mean = np.nanmean(pr_annual, axis = 2) * 86400 # x 86400 converts to mm

    ## Add the model's data to the output list
    pr_all_models_ls.append(pr_annual_mean)
    del (ds_model_members)
    end_time = time.time()
    print('Model (',len(pr_all_models_ls), '/',n_models ,') Complete. Time elapsed: ', round(end_time - start_time), ' seconds ', sep = '')    
end_time = time.time()
print('All ', len(model_names), ' models complete. Time elapsed: ', round(end_time - start_time), ' seconds ', sep = '')    

## Convert list to np array with new dimension corresponding to model
## This stacking could fail if the arrays are different sizes, which can
## happen if the end year is set to 2100 and some models extend to 2100 
## while others, frustratingly, do not (e.g. ending at 12/2099)
pr_all_models = np.transpose(np.stack(pr_all_models_ls, axis = 0), (1,2,0))
pr_all_models.shape  # dim = n_assets x year x n_models
pr_all_models_xr = xr.DataArray(pr_all_models, dims = ['asset_index', 'year', 'model_name'],
                                 coords = {'year': np.array(range(start_year, end_year+1)), 
                                           'model_name': model_names[0:5]})
pr_all_models_xr.to_netcdf(BASE_DATA_FOLDER + 'sust/sustglobal_asset_fwd_annual_precipitation_risk_'+ ENTITY_NAME + '_' + future_experiment_id + '.nc')
# m = 3
# a = pr_all_models_ls[m][:,119][...,None]
# pr_all_models_ls[m] = np.concatenate((pr_all_models_ls[m], a), 1) # CAMS-CSM1-0 ends in 2099
pr_ensemble_average = np.nanmean(pr_all_models, axis = 2)
pr_ensemble_average_lbd = np.nanquantile(pr_all_models, q = q_lbd, axis = 2)
pr_ensemble_average_ubd = np.nanquantile(pr_all_models, q = q_ubd, axis = 2)

pr_out = pd.concat([df_asset, pd.DataFrame(pr_ensemble_average, columns = range(start_year, end_year+1))], axis = 1)
pr_out.to_csv(BASE_DATA_FOLDER + 'sust/sustglobal_asset_fwd_annual_precipitation_risk_'+ ENTITY_NAME + '_' + future_experiment_id + '.csv', index = False)

pr_lbd_out = pd.concat([df_asset, pd.DataFrame(pr_ensemble_average_lbd, columns = range(start_year, end_year+1))], axis = 1)
pr_lbd_out.to_csv(BASE_DATA_FOLDER + 'sust/sustglobal_asset_fwd_annual_precipitation_risk_'+ ENTITY_NAME + '_' + future_experiment_id + '_lbd.csv', index = False)

pr_ubd_out = pd.concat([df_asset, pd.DataFrame(pr_ensemble_average_ubd, columns = range(start_year, end_year+1))], axis = 1)
pr_ubd_out.to_csv(BASE_DATA_FOLDER + 'sust/sustglobal_asset_fwd_annual_precipitation_risk_'+ ENTITY_NAME + '_' + future_experiment_id + '_ubd.csv', index = False)
# asset_idx = 300
# ## TEMP HACK Because there is only 1 model, the figure is less interesting.
# ## In the future there will be more models available (at least 2 not on GCC yet)
# ## So create a duplicate 'model' and add to np array
# colors = ['firebrick', 'red', 'blue', 'green', 'orange', 'yellow', 'steelblue', 'teal']
# for model_i in range(0, hw_all_models.shape[2]):
#     plt.plot(list(range(start_year, end_year+1)), hw_all_models[asset_idx,:,model_i],
# #             color = colors[model_i], label = model_names[model_i], linewidth = 0.5)                                                                                       
#              color = 'firebrick', label = '', linewidth = 0.5)  
# plt.plot(list(range(start_year, end_year+1)), hw_ensemble_average[asset_idx,:],
#          color = 'black', label = 'Ensemble Mean', linewidth = 2)  
# #plt.plot(list(range(start_year, end_year+1)), hw_ensemble_average[asset_idx,:] + 1.68*hw_ensemble_sd[asset_idx,:],
# #         color = 'black', label = '', linewidth = 0.5)  
# #plt.plot(list(range(start_year, end_year+1)), hw_ensemble_average[asset_idx,:] - 1.68*hw_ensemble_sd[asset_idx,:],
# #         color = 'black', label = '', linewidth = 0.5)  
# plt.plot(list(range(start_year, end_year+1)), hw_ensemble_q90[asset_idx,:],
#          color = 'black', label = '', linewidth = 0.5)  
# plt.plot(list(range(start_year, end_year+1)), hw_ensemble_q10[asset_idx,:],
#          color = 'black', label = '', linewidth = 0.5)  
                                                                                   
# plt.legend()
# plt.ylabel('Number of days over ' + str(hw_threshold_temperatureK - 273.15) + 'C')
# plt.title(df_asset.iloc[asset_idx,1])
# #plt.ylabel('Total Annual Burnt Area [%]')


# ## Explore average risk across asset type
# asset_type_hw = pd.DataFrame(hw_ensemble_average).groupby(df_asset['Type']).mean()
# colors = ['firebrick', 'red', 'blue', 'green', 'orange', 'yellow']
# for type_idx in range(len(list(asset_type_hw.index))):
#     type_name = list(asset_type_hw.index)[type_idx]       
#     plt.plot(list(range(start_year, end_year+1)), np.array(asset_type_hw)[type_idx,:],
#              color = colors[type_idx], label = type_name)
# plt.legend()
# plt.ylabel('Number of days over ' + str(hw_threshold_temperatureK - 273.15) + 'C')
# plt.title(ENTITY_NAME + ' (unscaled)')


