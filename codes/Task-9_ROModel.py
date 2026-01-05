"""
Task 9 Recharge Oscillator Model

- Script di calcolo delle _anomalie_ dei campi e le rispettive _derivate_ di 'tos' e 'thetao' per i dataset Bottino + Pi-control,
per poterle utilizzare come serie temporali nell'equazione del modello di Jin del Recharge-Oscillator attraverso la semplificazione del modello lineare di regressione.
Delle variaibili citate, dopo averle isolate nele regioni di definizione presenti nell'articolo di Jin 2021, viene eseguito il detrending, e la condensazione a serie temporali (in cui l'unica dimensione rimanenete è il tempo).
Per 'thetao' viene calcolata la termoclina attraverso 2 funzioni complesse Dask-friendly anche via chunks, la prima interpola sulla dimensione 'lev', la seconda calcola la termoclina.

Nota: Ho cercato di eseguire la miglior ottimizzazione possibile con Dask, attraverso anche la definizione dei 'chunks'.
Tuttavia, visto che lavoro solo su una piccola fetta del mondo (e non su tutto il mondo) i chunk non sono ottimizzati, sono troppo piccoli e mi viene dato un messaggio di WARNING:
"/home/montanarini/miniforge3/envs/ctl4b/lib/python3.9/site-packages/xarray/core/indexing.py:1228: PerformanceWarning: Slicing with an out-of-order index is generating 30 times more chunks"
Ad ogni modo la pipeline di Dask (parallelizzato in modo implicito da multiprocessing) costituisce la base di partenza per altri scripts.
"""
#INSERIRE CODICE PER LEGGERE E DARE IN OUTPUT COME WARNING SUL TERMINALE LA CPU E LA RAM USATA!
# Parallelizing libraries
#import multiprocessing as mp

# Definizione del main: lo script nella sula interezza, eseguito come una funzione
def main():
    #mp.set_start_method("fork", force=True) ---> Emula l'esecuzione come Jupyter notebook, più flessibile

    # import libraries
    ## System libraries
    import sys
    import logging
    ## Parallelizing libraries
    #import multiprocessing as mp
    from dask.distributed import Client, LocalCluster
    ## General data analisys libraries 
    import numpy as np
    import xarray as xr
    ## Time library
    import cftime
    from datetime import datetime
    # Personal library
    import AllePowerFunctions as apf

    # Imposto il cluster Dask
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit="5GB"
    )
    client = Client(cluster)
    print(client.dashboard_link)

    # Configurazione messaggi di logging
    logging.basicConfig(
        level=logging.DEBUG,    # Garantisce che DEBUG passi
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True              # Per Jupyter/ambienti già configurati
    )

    print(__name__, " has been launched on date:", datetime.now(), "\n")

    # Directories
    ## Home
    common_dir_home = '/home/montanarini/ELNINO/'
    dir_output_9 = 'output/Task-9_ROModel/'

    ## Datasets
    common_dir_ds = '/nas/'
    ### pi-control
    dir_pi = 'archive_CMIP6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/piControl/'
    dir_pi_tos = 'ocean/Omon/r1i1p1f1_r25/tos/*.nc'
    ### Bottini
    dir_BOTTINI = 'BOTTINO/irods_data/stabilization-{}/r1i1p1f1/'
    dir_bxxx = '{}/*.nc'

    # Constants
    secondi_mese = 60*60*24*30.43803025425 # Durata media mese: 365.256363051/12

    # Keywords: datasets, variables and path completion
    ## Datsets
    dataset_names = ['b100'] #['pi-control', 'b990','b025', 'b050', 'b065', 'b080', 'b100']
    dataset_names_iteration = len(dataset_names)
    ## Completamento dei path dei bottini
    bottini_paths = ['ssp585-2100'] #['hist-1990','ssp585-2025', 'ssp585-2050', 'ssp585-2065', 'ssp585-2080', 'ssp585-2100']
    ## Definisco le liste di variaibli (Oceaniche)
    var_names = ['tos', 'thetao']
    var_names_iteration = len(var_names)

    # Parametri
    # valori da usare (i) confronto per violin-plot: 500, 360, 500; (ii) confronto per scatter-plot: [[1000, 360, 1000], [100, 360, 100]]
    time_window = 500 # ultimi n anni del Dataset (non per hist e obs), default: 500
    time_rm = 360 # Mesi in cui si applica la RM, default: 360
    print("\nThe following set of parameters was selected:",
        "\n- Last n years of the bottino runs: ", time_window,
        "\n- Running Mean time, in months (for the historical detrending): ", time_rm)

    # Flag per eseguire le cose una volta sola
    flag_continental = True

    # Dizionario dei path per i Datasets per ogni variabile
    directories = {}
    for i, datasetN in enumerate(dataset_names):
        directories[datasetN] = {}
        for var in var_names:
            # Aggiunta dei path non bottino
            if datasetN in ['pi-control']:
                directories[datasetN][var] = [common_dir_ds + dir_pi + 'ocean/Omon/r1i1p1f1_r25/' + var + '/*.nc']
            # Aggiunta dei path bottino
            elif datasetN in ['b990','b025', 'b050', 'b065', 'b080', 'b100']:
                directories[datasetN][var] = [(common_dir_ds + dir_BOTTINI).format(bottini_paths[i-1]) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index(var)])]

    # Liste per salvare i tempi impiegati per ogni dataset
    time_begin_var = [None] * var_names_iteration
    time_end_var = [None] * var_names_iteration
    time_var = [None] * var_names_iteration
    time_begin_ds = [None] * dataset_names_iteration
    time_end_ds = [None] * dataset_names_iteration
        
    # Iterazione sulle Variabili
    for i, var in enumerate(var_names):
        time_begin_var[i] = datetime.now()

        # Seleziona solo la variaibile n ed escludi le altre
        # if var not in ['thetao']:
        #    time.sleep(1)
        #    time_end_var[i] = datetime.now()
        #    time_var[i] = (time_end_var[i]-time_begin_var[i])
        #    continue

        print(f"\n# Inizia l'elaborazione della variabile: {var}")

        # Iterazione sui Dataset
        for j, datasetN in enumerate(dataset_names):
            time_begin_ds[j] = datetime.now()
            
            # Display informazioni sulla directory 
            print(f"\n## Inizia l'elaborazione di: Dataset_{datasetN}\nCorresponding path: ", directories[datasetN][var][0])
            

            print("\nStart of Dataset opening/upload")

            ###################################################################################################################################
            #################################### (a) Discrimination based on the Dataset of origin ############################################

            if datasetN in ['b990','b025', 'b050', 'b065', 'b080', 'b100']:
                # Apertura dataset
                Dataset = xr.open_mfdataset(
                    directories[datasetN][var][0],
                    use_cftime=True,
                    parallel=True,
                    chunks={'time': 100, 'lev': -1, 'lat': 10, 'lon': 50} # NB: le dimensiioni non presenti (lev in tos ad esempio) vengono ignorate
                )
                
            elif datasetN in ['pi-control']:
                # Apertura dataset
                Dataset = xr.open_mfdataset(
                    directories[datasetN][var][0],
                    decode_times=True,
                    use_cftime=True,
                    parallel=True,
                    chunks={'time': 100, 'lev': -1, 'lat': 10, 'lon': 50}
                )
                                
            # Riordinamento coordinate per evitare frammentazione dei chunks
            #Dataset = apf.sort_dataset_coords(Dataset)#.rechunk({'time': 10, 'lev': -1, 'lat': 10, 'lon': 50})
            
            # Discriminazione del range temporale:     
            max_year = Dataset['time'].max()
            min_year = Dataset['time'].min()
            ## Estremi per i Bottini
            sup_year_bott = cftime.DatetimeProlepticGregorian(int(max_year.dt.year), 1, 1)
            inf_year_bott = cftime.DatetimeProlepticGregorian(int(max_year.dt.year-time_window), 1, 1)
            date_format = '%Y-%m-%d'
            ## Limitazione del range temporale
            Dataset = Dataset.sel(time=slice(inf_year_bott, sup_year_bott))

            print("\nStart of time and space variable selection")

            if var in ['tos']:
                # Passaggio al DataArray
                campo = Dataset['tos'].sel(lat=slice(-5, 5), lon=slice(210, 270)) # Nino3 zone for SSTA: eastern equatorial Pacific SSTA (T_E)

                # Detrending del campo e calcolo dell'anomalia
                campo_detrend, anomaly_detrend, trend_hist, seasonal_cycle = apf.detrending(campo, time_rm)
                
                # Maschera continentale
                if flag_continental == True:
                    continental_mask = np.isnan(campo)[0,...]
                    flag_continental = False
                
                # Calcolo la derivata dell'anomalia
                anomaly_detrend = anomaly_detrend.chunk({'time': 100, 'lat': 10, 'lon': 50})  # Rechunking del campo dopo la funzione di detrending(). {'time': -1} se vuoi un unico blocco
                Danom_Dt = anomaly_detrend.differentiate('time') * secondi_mese

                # Condensazione ad array 1D, discriminazione sul tipo di variaibile: zonale (time, lat, lon) o verticale (time, lat, lon, lev)
                #campo_detrend = apf.global_mean(campo_detrend)
                anomaly_detrend = apf.global_mean(anomaly_detrend)
                Danom_Dt = apf.global_mean(Danom_Dt)

                # Salvataggio dei campi
                print("\nStart saving the evaluated variables. Dataset: ", datasetN, ", Variable :", var)
                ds = xr.Dataset(
                    data_vars=dict(
                        anomaly = anomaly_detrend,
                        Danom_Dt = Danom_Dt,
                    )
                ).to_netcdf(common_dir_home + dir_output_9 + f'Task-9_{datasetN}-{var}.nc', compute=True)

            elif var in ['thetao']:
                # Passaggio al DataArray
                campo = Dataset['thetao'].sel(lev=slice(0,300), lat=slice(-5, 5), lon=slice(140, 205)) # Warm Pool region for Thermocline Depth Anomaly (hw) (Original: [120, 205])), limitation to the Mixing Layer.
                
                # Controllo dei chunks
                ## Informazioni sui chunk
                chunks = campo.data.chunks  # tuple di tuple: una per dimensione
                dtype_size = campo.data.dtype.itemsize  # byte per elemento
                ## Calcola il numero di elementi per chunk (in genere prendiamo il primo per stimare)
                chunk_shape = tuple(dim[0] for dim in chunks)
                n_elements = np.prod(chunk_shape)
                bytes_per_chunk = n_elements * dtype_size
                print(f"Chunk shape: {chunk_shape}")
                print(f"Bytes per chunk: {bytes_per_chunk / 1e6:.2f} MB")

                # Detrending del campo e calcolo dell'anomalia
                campo_detrend, anomaly_detrend, trend_hist, seasonal_cycle = apf.detrending(campo, time_rm)
                
                # Preparazione al calcolo della termoclina
                ## Coordinate
                lev = campo_detrend['lev']
                lon = campo_detrend['lon']
                ## Definizione dimensioni estese per interpolazioni
                lev_ext = xr.DataArray(np.linspace(lev.min().values, lev.max().values, 500), dims=['lev'])
                lon_ext = np.linspace(lon.min(), lon.max(), 500)
                lon_ext_coord = xr.DataArray(lon_ext, dims=['lon_ext']) # Crea un DataArray per allineare le coordinat
                
                ## Media latitudinale
                campo_tofunc = apf.meridional_mean(campo_detrend)

                ## Applico l'interpolazione alla coordinata di profondità (z) -'lev'- per ogni punto di longitudine e ricalcolo il valore del campo nei nuovi punti
                # Ci sono più modi per interpolare:
                #  - Modo 1, interpolazione semplice: DataArrray.interp()
                #  - campo_tofunc_interp = campo_tofunc.interp(coords={'lev': lev_ext}, method="linear")  # Dask-friendly e vettorizzato.
                #  - Modo 2: interpolazione spline, più precisa e accurata. Si può ottenere solo con funzioni non Dask friendly di Scipy.interpolate: interp1d(), splrep() + splev(); CubicSpline()
                #  che sono da applicare via 'apply_ufunc()' o 'map_blocks()' più flessibile specialmente per datasets chunked. Sono riuscito a eseguire una spline solo con interp1d(), con le altre funzioni ho fallito.
                campo_tofunc_interp = apf.spline_interpolation_along_dim(
                    campo_tofunc,
                    lev_ext,
                    "lev",
                    "cubic"
                )#.transpose('time', 'lon', 'lev').chunk({'time': 100, 'lon': 50}) # Riordino le dimensioni in modo che ci siano per prima le non core ('time', 'lat') e poi le core ('lon', 'lev') e forzo la coerenza con un chunking
                
                # Calcolod della Termoclina 
                interpol = False # Flag per abilitare o meno l'interpolazione
                termocline, depth_max_grad, temp_max_grad = apf.thermocline(campo_tofunc_interp, interpol) # La termoclina restituitita ha valori positivi sull'asse 'lev' , quindi sono da invertire moltiplicando *(-1)
                
                ## Anomalia
                termocline_anom = termocline - termocline.mean('time')
                
                ## Calcolo della derivata della termoclina
                #termocline_anom = termocline_anom.chunk({'time': -1})  # Rechunking del campo dopo la funzione di detrending()
                Dtermocline_anom_Dt = termocline_anom.differentiate('time')
                
                ## Riduzione dimensionale ad array 1D
                termocline_anom = termocline_anom.mean('lon')*(-1)
                Dtermocline_anom_Dt = Dtermocline_anom_Dt.mean('lon')*secondi_mese*(-1)

                # Salvataggio dei campi
                print("\nStart saving the evaluated variables. Dataset: ", datasetN, ", Variable :", var)
                    
                ds = xr.Dataset(
                    data_vars=dict(
                        termocline_anom = termocline_anom,
                        Dtermocline_anom_Dt = Dtermocline_anom_Dt,
                    )
                ).to_netcdf(common_dir_home + dir_output_9 + f'Task-9_{datasetN}-{var}.nc', compute=True)
            
            ### Uscita dal loop sui Dataset ###
        
            # Dataset time recap
            time_end_ds[j] = datetime.now()
            print("\nEnd dataset elaboration")
            print("Dataset {} took {}".format(datasetN, time_end_ds[j]-time_begin_ds[j]))

        ### Uscita dal loop sulle variabili ###
    
        # Variable time recap
        time_end_var[i] = datetime.now()
        time_var[i] = (time_end_var[i]-time_begin_var[i])
        print("\nEnd of variable evaluation")
        print("Variable {} took {}".format(var_names[i], time_var[i]))

    # Time recap
    print("\n## Timing recap")
    for i, var in enumerate(var_names):
        print("\nOverall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))

    print("\nOverall evaluation of the script was completed in {}".format(np.sum(time_var)))
    print("\nEnd of the script")

# Esecuzione del main
if __name__ == "__main__":
    main()