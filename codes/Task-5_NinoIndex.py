# import libraries
import sys
import time as time_lib
import numpy as np
import xarray as xr
import pickle
from datetime import datetime
import AllePowerFunctions as apf
import climtools_lib as ctl

print('Script started at ', datetime.now())
#################################################################################################################################################
########################################################## Acquisizione Dataset #################################################################
#################################################################################################################################################
common_Dir = '/nas/'
Dir_hist = 'BOTTINO/CMIP6/LongRunMIP/EC-Earth-Consortium/EC-Earth3/historical/r4i1p1f1/Omon/tos_rg_1850-2014.nc'
Dir_obs = 'reference/HadISST/2020/HadISST_sst.nc'
Dir_pi = 'TIPES/piControl/Omon/tos_Omon_EC-Earth3_piControl_r1i1p1f1_gn_185001-235012.nc'
Dir_output = '/home/montanarini/ELNINO/output/Task-5_NinoIndex/Variables/'
Dir_BOTTINI = 'BOTTINO/irods_data/stabilization-{}/r1i1p1f1/Omon/tos/*.nc'

# Definisco i set di parole per la sostituzione nel path della stringa e nella variabile
bottini_paths = ['hist-1990', 'ssp585-2025', 'ssp585-2050', 'ssp585-2065', 'ssp585-2080', 'ssp585-2100'] #Per il path
dataset_names = ['b990', 'b025', 'b050', 'b065', 'b080', 'b100', 'pi-control', 'hist', 'obs'] #Per la variabile

directories = [(common_Dir + Dir_BOTTINI).format(bottinoP) for bottinoP in bottini_paths]
directories += [common_Dir + Dir_pi, common_Dir + Dir_hist, common_Dir + Dir_obs]

n = len(directories)
# Apertura Dataset.
Dataset_hist = xr.open_dataset(common_Dir+Dir_hist)

# Creo una lista per salvare i tempi impiegati per ogni dataset
timeI = []
timeF = []

for i, (datasetN, directory) in enumerate(zip(dataset_names, directories)):

   timeI.append(datetime.now())

   # Crea una variabile con il nome Dir_{dataset_names}  
   DATASET = f'Dir_{datasetN}'  # 'DATASET' è il nome della variabile, 'datasetN' è il nome della singola parola nella lista 'dataset_names', fer mia d'la confusiouna
   print(DATASET)

   # Assegna la stringa formattata alla variabile con il nome appropriato
   globals()[DATASET] = directory

   print('Inizia lelaborazione di: ', DATASET, '\n')
   print(directory)


   if i==6:
      #######################
      ##### PI-control ######
      #######################
      Dataset_pi = xr.open_dataset(globals()[DATASET], chunks={'time':360}, decode_times=False, use_cftime=True) # Funziona ugualmente sostituendo a 'globals()[DATASET]' --> 'path_formattato'
      # Regridding
      Dataset_pi = ctl.regrid_dataset(Dataset_pi, regrid_to_reference=Dataset_hist)
      # Passaggio al DataArray
      sst = Dataset_pi['tos']
      # Riassegnazione dimensione tempo
      sst['time'] = xr.date_range(start='1850-01-01', end='2351-01-01', periods=None, freq='M', tz=None, normalize=True, name=None, calendar='gregorian', use_cftime=True) #2351-01-01
      # Estrazione coordinate
      time = sst['time']
      lon = sst['lon']
      lat = sst['lat']
   elif i==7:
      #######################
      ######### HIST ########
      #######################
      # Passaggio al DataArray
      sst = Dataset_hist['tos']
      # Estrazione coordinate
      time = sst['time']
      lon = sst['lon']
      lat = sst['lat']      
   elif i==8:
      #######################
      ######### OBS #########
      #######################
      Dataset_obs = xr.open_dataset(globals()[DATASET])
      #Regridding
      Dataset_obs = ctl.regrid_dataset(Dataset_obs, regrid_to_reference=Dataset_hist)
      # Passaggio al DataArray
      sst = Dataset_obs['sst']
      # Estrazione coordinate
      time = sst['time']
      lon = sst['lon']
      lat = sst['lat']
   else:
      #######################
      ####### BOTTINI #######
      #######################
      # # adding 2 seconds time delay
      # time_lib.sleep(1)
      # continue
      Dataset_bxxx = xr.open_mfdataset(globals()[DATASET], chunks={'time':360}, use_cftime=True) #.isel(time=slice(0,120))
      # Regridding
      Dataset_bxxx = ctl.regrid_dataset(Dataset_bxxx, regrid_to_reference=Dataset_hist)
      # Passaggio al DataArray
      sst = Dataset_bxxx['tos']
      # Estrazione coordinate
      time = sst['time']
      lon = sst['lon']
      lat = sst['lat']

   #################################################################################################################################################
   ########################################################## NinoIndex SSP585-2025 ################################################################
   #################################################################################################################################################
   print("\n","Start computing the Index")
   Index34 = apf.nino_index(sst)
   print("\n","End of computing the Index")


   #################################################################################################################################################
   ############################################ Power Spectra (one-dimensional discrete Fourier Transform) #########################################
   #################################################################################################################################################
   print("\n","Start computing the Power Spectra")
   freqs, power_spec = apf.power_spectra(Index34, 0)
   print("\n","End of computing the Power Spectra")

   #################################################################################################################################################
   ########################################################### Standard Deviations##################################################################
   #################################################################################################################################################
   print("\n","Start computing the Standard Deviations")
   std_T, std_window, std_RM =  apf.standard_deviation(Index34, 360, 100, 'time')
   print("\n","End of computing the Standard Deviations")

   #################################################################################################################################################
   ####################################################### Salvataggio Variabili Calcolate #########################################################
   #################################################################################################################################################
   print("\n","Start saving the evaluated variables")
   pickle.dump([Index34, freqs, power_spec,
               std_T, std_window, std_RM], open(Dir_output + f'Task-5_Dataset-{datasetN}.p', 'wb'))
   print("\n","End saving the evaluated variables")

   #################################################################################################################################################
   timeF.append(datetime.now())
   print('Evaluation of ', DATASET, ' completed in {}'.format(timeF[i-1]-timeI[i-1]))
# Recap dei tempi impiegati per ogni dataset
for i, dataset in enumerate(dataset_names):
   print('Dataset ', dataset,' evaluated in {}'.format(timeF[i]-timeI[i]))
print("End of the script")
sys.exit()