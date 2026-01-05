"""
Task 6 Regressioni, 7 Termoclina, 8 Composite:


- Calcolo delle Regressioni sull'indice Nino 3.4 per gli ultimi 100 anni di ogni serie temporale per i campi di: 
   SST;
   T2m;
   U850;
   Prec;
   GeoPot500 (nei mesi DJF);
   Salinity; --> da implementare
   
   _l'indice è recuperato dal risultato di Task 5_
   
   NB: all'indice per ogni Dataset è stato rimosso il trend storico con il metodo della Running Mean e anche il ciclo stagionale con la media sul tempo,
    Tuttavia a differenza delle regressioni in Task 1,2,3 NON è stato rimosso il trend storico ai vari campi su cui si va ad eseguire la regressione.
    Ciò non ha conseguenze su tutti i Dataset dei bottini, essendo le regressioni calcolate sugli ultimi 100 anni quando il transiente è già finito e 
    si è giunti ad una stabilizzazione, tuttavia ha una certa incidenza (ignota) sui Dataset di Storico e Osservato

- Calcolo della termoclina negli ultimi 50 anni dei vari dataset presenti (Osservato escluso in quanto tale dato è mancante)

- Calcolo della Composite Nino e Nina. (fatto solo nella seconda parte del codice, ma estendibile anche alla prima parte del codice.)

Nota: 
Lo script è suddiviso in due macro-rami a seconda della scelta dell'utente se regriddare o meno i dataset.
- Nel primo caso tutti i dataset vengono regriddati sul dataset prodotto da Ec-Earth di Temperatura osservata superficiale degli oceani del Dataset_hist_tos, dunque in una griglia 180x360.
Questa operazione è particolarmente lunga da svolgere da un punto di vista computazionale, è meglio eseguirla fuori dallo script python, con altri strumenti come CDO o altre librerie in grado di parallelizzare la cosa.
- Nel secondo caso non viene eseguita alcuna regriddazione, è necessario assicurarsi che tutti i dataset condividano la stessa griglia lat-lon per evitare crash.
- Il 1° e il 2° caso adottano due metodi diversi per la costruzione dei dizionari dei path dei datasets:
il 1° è un metodo esplicito e leggibile e debuggabile, si vede a colpo d'occhio ogni path. Il 2° è un metodo implicito molto più compatto ma di conseguenza meno leggibile e pi difficile da debuggare in caso di problemi.
"""

""" ROBA DA FARE:
- TOGLIERE IL TREND STORICO AL CALCOLO DELLA REG PER TUTTI I CAMPI DI HIST E OBS. PER I BOTTINI NON è NECESSARIO.
- IMPLEMENTARE IL CALCOLO DELLA TERMOCLINA TRAMITE DEFINIZIONE (GRADIENTE MASSIMO DELLA TEMPERATURA)

-- HO MESSO A POSTO THETHAO, MA CONTROLLARE, FINIRE LE ALTRE VARIAIBILI.
"""

### import libraries ###

# General data analisys libraries 
import sys
import numpy as np
import xarray as xr
import pickle
# Librari for treads managements
import threading
# Time library
import time
from datetime import datetime
# My and Fede functions libraries
import AllePowerFunctions as apf
import climtools_lib as ctl

print("Task-6_RegCompTermo.py has been launched on date:", datetime.now(), "\n")
print("Choose which script to run:",
      "\n1 := Enabled regression of the Datasets over the Historical Dataset wich has a space grid divided in 180x360 gridpoints",
      "\n2 := Disabled regression. Warning! The Datasets must have the same space grid subdivision, better if it is 180x360 or lower",
      )
   
# Avvia il thread per ottenere l'input dell'utente
user_input_thread = threading.Thread(target=apf.get_user_input)
user_input_thread.start()

# Attendi per un massimo di n secondi
user_input_thread.join(timeout=10)

# Se il thread è ancora attivo, significa che l'utente non ha inserito nulla
if user_input_thread.is_alive():
    user_input = 2  # Imposta il valore di default
    print("Time over. Default value of, ", user_input, " will be used.")

print(f"Hai selezionato il metodo {user_input}")

if (user_input==1):
    
    #################################################################################################################################################
    ########################################################## Acquisizione Dataset #################################################################
    #################################################################################################################################################

    ###############################################################
    # 1: (a) Get Paths;
    #    (b) Variables assegnation;
    #    (c) Directories building: directories's dictionary creation.
    #    (d) Regression building: regression's dictionary creation.
    ###############################################################

    ###################################################################################################################################
    ########################################################## (a) Get paths ##########################################################

    # Common dirs
    common_Dir = '/nas/'
    common_Dir_Output6 = '/home/montanarini/output/Task-6/'
    common_Dir_Output7 = '/home/montanarini/output/Task-7/'
    Dir_Output_task5 = '/home/montanarini/output/Task-5/Variables/'

    # Directories multi-variables Datasets
    Dir_hist = 'BOTTINO/CMIP6/LongRunMIP/EC-Earth-Consortium/EC-Earth3/historical/r4i1p1f1/'
    Dir_obs = 'reference/HadISST/2020/'
    Dir_pi = 'archive_CMIP6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/piControl/' #'TIPES/piControl/'
    Dir_BOTTINI = 'BOTTINO/irods_data/stabilization-{}/r1i1p1f1/'
    ######################################################### Fine punto 1(a) #########################################################
    ###################################################################################################################################

    ###################################################################################################################################
    ############################################ (b) Dataset's single Variables assegnation ###########################################

    ### singles's variabiles directories ---->questa parte è necessaria solamente per il 1° metodo di riempimento del dizionario, mentre è inutile nel 2°
    
    # Bottini
    dir_bxxx_tos = 'Omon/tos/*.nc'
    #dir_b_sos = 'Omon/sos/*.nc'
    dir_bxxx_thetao = 'Omon/thetao/*.nc'
    dir_bxxx_tas = 'Amon/tas/*.nc'
    dir_bxxx_ua = 'Amon/ua/*.nc'
    dir_bxxx_pr = 'Amon/pr/*.nc'
    dir_bxxx_zg = 'Amon/zg/*.nc'
    dir_bxxx = '{}/*.nc'

    # Pi-Control (no salinity)
    dir_pi_tos = 'ocean/Omon/r1i1p1f1/tos/*.nc'
    dir_pi_thetao = 'ocean/Omon/r1i1p1f1/thetao/*.nc'
    dir_pi_tas = 'atmos/Amon/r1i1p1f1/tas/*.nc'
    dir_pi_ua = 'atmos/Amon/r1i1p1f1/ua/*.nc' 
    dir_pi_pr = 'atmos/Amon/r1i1p1f1/pr/*.nc'
    dir_pi_zg = 'atmos/Amon/r1i1p1f1/zg/*.nc'

    # Historic (no salinity)
    dir_hist_tos = 'Omon/tos/*.nc'
    dir_hist_thetao = 'Omon/thetao/*.nc'
    dir_hist_tas = 'Amon/tas/*.nc'
    dir_hist_ua = 'Amon/ua/*.nc'
    dir_hist_pr = 'Amon/pr/*.nc'
    dir_hist_zg = 'Amon/zg/*.nc'

    # Observed (only tos/sst)
    dir_obs_tos = 'HadISST_sst.nc'



    ### Definisco i set di parole per la sostituzione nel path e nella variabile della stringa

    # Datasets
    dataset_names = ['b990', 'b025', 'b050', 'b065', 'b080', 'b100', 'pi-Control', 'hist', 'obs']
    dataset_names_iteration = len(dataset_names)

    # Completamento dei path dei bottini
    bottini_paths = ['hist-1990', 'ssp585-2025', 'ssp585-2050', 'ssp585-2065', 'ssp585-2080', 'ssp585-2100'] 

    # Definisco le liste di variaibli Oceaniche e Atmosferiche
    var_ocean = ['tos', 'thetao']
    var_atm = ['tas', 'ua', 'pr', 'zg']
    var_omon = ['Omon/' + var for var in var_ocean] # Aggiungo la stringa "Omon/" ad ogni elemento delle liste
    var_amon = ['Amon/' + var for var in var_atm]

    var_names = var_ocean + var_atm
    var_names_folder = var_omon + var_amon
    ######################################################### Fine punto 1(b) #########################################################
    ###################################################################################################################################

    ###################################################################################################################################
    ######################################## (c) Directories building: dictionary creation ############################################

    ### Definisco le liste di path dei vari Datasets per ogni variabile ---> ogni lista conterrà i path di tutti i Datasets per la specifica variabile

    ## 1° metodo ##
    # --> Metodo esplicito, formalmente più corretto, utilizza una struttura più organizzata, intuitiva e facilmente aggiornabile; necessita tuttavia di più righe di codice (52-84) ed è più ripetitivo
    directories = {
    'tos': [(common_Dir + Dir_BOTTINI).format(bottinoP) + dir_bxxx.format(var_names_folder[var_names.index('tos')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_tos] + [common_Dir + Dir_hist + dir_hist_tos] + [common_Dir + Dir_obs + dir_obs_tos],
    'thetao': [(common_Dir + Dir_BOTTINI).format(bottinoP) + dir_bxxx.format(var_names_folder[var_names.index('thetao')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_thetao] + [common_Dir + Dir_hist + dir_hist_thetao],
    'tas': [(common_Dir + Dir_BOTTINI).format(bottinoP) + dir_bxxx.format(var_names_folder[var_names.index('tas')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_tas] + [common_Dir + Dir_hist + dir_hist_tas],
    'ua': [(common_Dir + Dir_BOTTINI).format(bottinoP) + dir_bxxx.format(var_names_folder[var_names.index('ua')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_ua] + [common_Dir + Dir_hist + dir_hist_ua],
    'pr': [(common_Dir + Dir_BOTTINI).format(bottinoP) + dir_bxxx.format(var_names_folder[var_names.index('pr')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_pr] + [common_Dir + Dir_hist + dir_hist_pr],
    'zg': [(common_Dir + Dir_BOTTINI).format(bottinoP) + dir_bxxx.format(var_names_folder[var_names.index('zg')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_zg] + [common_Dir + Dir_hist + dir_hist_zg]
    }

    ## 2° metodo ##
    # --> Metodo implicito, più compatto (quindi meno dispersivo), è meno intuibile e non mostra immediatamente le chiavi del dizionario
    directories = {}

    for var in var_names:
        if var in var_ocean:  # Se la variabile è oceanica
            directories[var] = [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Omon/' + dir_bxxx.format(var_names[var_names.index(var)]) for bottinoP in bottini_paths]
            # Aggiunta dei path non bottino
            if var == 'tos':
                    directories[var] += [common_Dir + Dir_pi + dir_pi_tos, common_Dir + Dir_hist + dir_hist_tos, common_Dir + Dir_obs + dir_obs_tos]
            else:
                    directories[var] += [common_Dir + Dir_pi + 'ocean/Omon/r1i1p1f1/' + var + '/*.nc', common_Dir + Dir_hist + 'Omon/' + var + '/*.nc']
        else:
            # Se la variabile è atmosferica
            directories[var] = [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon/' + dir_bxxx.format(var_names[var_names.index(var)]) for bottinoP in bottini_paths]
            # Aggiunta dei path non bottino
            directories[var] += [common_Dir + Dir_pi + 'atmos/Amon/r1i1p1f1/' + var + '/*.nc', common_Dir + Dir_hist + 'Amon/' + var + '/*.nc']
    ######################################################### Fine punto 1(c) #########################################################
    ###################################################################################################################################


    ###################################################################################################################################
    ############################### (d) Regression array building: regressions's dictionary creation ##################################
    ### Creazione di un dizionario contenente un dizionario annidato per archiviare le liste che conterranno i risultati della regressione (suddivisi in 5 diversi prodotti)
    regression_product_dict = {}

    regression_products_list = ['trend', 'intercept', 'trend_err', 'intercept_err', 'pval'] # Nomi dei prodotti generati dalla funzione che calcola la regressione
    var_names_sel = ['tos', 'tas', 'ua850', 'ua200', 'pr', 'zg500'] # Nome delle variabili dopo la selezione dei livelli ('thetao' non figura perchè su di essa non calcoliamo la reg.)

    for product in regression_products_list:
        regression_product_dict[product] = {}
        for var in var_names:
            if var == 'tos':
                regression_product_dict[product][var] = [None] * dataset_names_iteration
            else:
                regression_product_dict[product][var] = [None] * (dataset_names_iteration - 1)

    var_names_sel.insert(1, 'thetao') # lista su cui vado ad eseguire il main loop


    # Creo delle liste per salvare i tempi impiegati per ogni dataset ('_a' è per il punto 2 (a), '_b' è per il punto 2 (b))
    time_begin_a = [None] * dataset_names_iteration
    time_end_a = [None] * dataset_names_iteration
    time_begin_b = [None] * dataset_names_iteration
    time_end_b = [None] * dataset_names_iteration
    time_var = [None] * len(var_names)
    ######################################################### Fine punto 1(d) #########################################################
    ###################################################################################################################################

    #################################################################################################################################################
    ############################################################### Main Loop #######################################################################
    #################################################################################################################################################

    ###############################################################
    # 2: main loop over dictionary keys thus on the Dataset's variables names; second nested loop over the Dataset names and dataset's values
    #    (a) Discrimination based on the Dataset of origin;
    #    (b) Discrimination based on the Dictionary' keys (Dataset's variables) and Regression calculation.
    #    (c) Variable saving.
    ###############################################################

    """
        Nota:      
        metodo "globals()", restituisce il dizionario dei simboli globali.
        Accedo a variabili il cui nome era costruito dinamicamente concatenando una stringa (dir_pi_ o dir_hist_) con il valore della variabile var.
        Utilizzando globals(), puoi ottenere il valore della variabile usando il suo nome come chiave nel dizionario restituito da globals().
        globals(): Restituisce un dizionario che rappresenta lo spazio dei nomi globale. In altre parole, contiene tutte le variabili globali
            definite nel tuo programma. Le chiavi del dizionario sono i nomi delle variabili, e i valori sono i corrispondenti oggetti.
        locals(): Restituisce un dizionario che rappresenta lo spazio dei nomi locale. Questo spazio dei nomi è specifico di una funzione o di un blocco di codice.
            Le chiavi del dizionario sono i nomi delle variabili locali, e i valori sono i corrispondenti oggetti.
    """
    # Apertura Dataset 'hist_tos'
    # Dataset_hist_tos = xr.open_mfdataset(common_Dir + Dir_hist + dir_hist_tos) ----> problea successivo nel regridding
    Dir_hist_tos = '/nas/BOTTINO/CMIP6/LongRunMIP/EC-Earth-Consortium/EC-Earth3/historical/r4i1p1f1/Omon/tos_rg_1850-2014.nc'
    Dataset_hist_tos = xr.open_dataset(Dir_hist_tos)

    nino_index = [None]*dataset_names_iteration  # Lista per memorizzare gli indici dei vari Dataset
    nino_index_resampled = [None]*(dataset_names_iteration-1)

    thetao = [None]*(dataset_names_iteration-1) # Lista per memorizzare i termoclini dei vari Dataset

    time_window = 1 # ultimi n anni del Dataset (non per hist e obs)
    print("\nThe following set of parameters was selected:\nLast n years of the bottino runs: ", time_window)



    ### Iterazione sulla chiave del dizionario (=sulle variabili contenute nei Dataset)
    for i, var in enumerate(var_names):

        ### Iterazione sul valore del dizionario (=sulle directories delle variabili contenute nei Dataset) e sui Datasets stessi: l'indice 'j' indica il Dataset
        #for j, (datasetN, directoryTOS) in enumerate(zip(dataset_names, directories[var])):
        for j, datasetN in enumerate(dataset_names):
            
            if (var != 'tos' and j == dataset_names_iteration -1): # skip each iteration for Dataset obs except for 'tos'
                time_begin_a[j] = datetime.now()
                time.sleep(1)
                time_end_b[j] = datetime.now()
                continue

            Dataset = f'Dir_{datasetN}' # ora 'Dataset' è una stringa, dopo diventerà un xarray.Dataset
            # Assegna la stringa formattata alla variabile con il nome appropriato
            globals()[Dataset] = directories[var][j]   # Funziona ugualmente sostituendo a 'globals()[Dataset]' --> 'path_formattato'

            print("\nInizia lelaborazione di: Dataset_{}, \nCorresponding path: {}".format(datasetN).format(directories[var][j]))

            print("\nStart of Dataset opening/upload and regridding")
            time_begin_a[j] = datetime.now()

            ###################################################################################################################################
            #################################### (a) Discrimination based on the Dataset of origin ############################################

            if datasetN in ['b990','b025', 'b050', 'b065', 'b080', 'b100']: #j < 6:
                ####### BOTTINI #######

                ### Apertura dataset

                Dataset = xr.open_mfdataset(globals()[Dataset], use_cftime=True) # globals()[Dataset] <----> directories[var][j]

                
            elif datasetN in ['pi-control']: #j==6:
                ##### Pi-Control ######

                ### Apertura dataset
                
                Dataset = xr.open_mfdataset(globals()[Dataset], decode_times=True, use_cftime=True)
                
                # Riassegnazione dimensione tempo
                Dataset['time'] = xr.date_range(start='1850-01-01', end='2351-01-01', periods=None, freq='M', tz=None, normalize=True, name=None, calendar='gregorian', use_cftime=True)
                

            elif datasetN in ['hist']: # j==7:
                ######### HIST ########
                
                ### Apertura dataset

                Dataset = xr.open_mfdataset(globals()[Dataset])
                

            else: #datasetN in ['obs']: or j==8
                ######### OBS #########

                ### Apertura dataset

                Dataset = xr.open_dataset(globals()[Dataset])
                
                # Riassegnazione nome campo 
                Dataset = Dataset.rename({'sst': 'tos'})
            
            
            
            ### Limitazione del range temporale: DA SETTARE NELLA VERSIONE FINALE
            max_year_dtgr = Dataset['time'].max()
            sup_year = str(int(max_year_dtgr.dt.year))+'-01-01'
            inf_year = str(int(max_year_dtgr.dt.year)-time_window)+'-01-01'

            Dataset = Dataset.sel(time=slice(inf_year, sup_year)).compute()


            ### Regridding
            Dataset = ctl.regrid_dataset(Dataset, regrid_to_reference=Dataset_hist_tos)
            
            ### Recupero dell' Indice Nino 3.4

            filename = 'Task-5_Dataset-{}.p'.format(datasetN)
            nino_index[j] = pickle.load(open(Dir_Output_task5 + filename, 'rb'))[0]
            # Seleziono il periodo desiderato dell'indice
            nino_index[j] = nino_index[j].sel(time=slice(inf_year, sup_year)).values
            with open(Dir_Output_task5 + f'Task-5_Dataset-{datasetN}-Subset.p', 'ab') as file:
                pickle.dump([nino_index[j]],
                            file)
                file.close()            
            
            
            print("End of Dataset elaboration")
            time_end_a[j] = datetime.now()
            print("Elaboration of Dataset_{} completed in {}".format(datasetN).format(time_end_a[j]-time_begin_a[j]))

            ######################################################### Fine punto 2(a) #########################################################
            ###################################################################################################################################
        
            ###################################################################################################################################
            ################ (b) Discrimination based on the Dictionary' keys (Dataset's variables) and Recgression calculation ###############

            print("\nStart of time and space variable selection and Regression evaluation")
            time_begin_b[j] = datetime.now()

            if var == 'tos':
                # Isola le singole variabili
                tos = Dataset['tos'].sel(time=slice(inf_year, sup_year), lat=slice(-5, 5), lon=slice(190, 240)).compute() #lat=slice(-85, 85)
                # Maschera continentale
                continental_mask = np.isnan(tos)[0,...]
                # Calcolo della Regressione
                (regression_product_dict['trend'][var][j], 
                    regression_product_dict['intercept'][var][j], 
                    regression_product_dict['trend_err'][var][j], 
                    regression_product_dict['intercept_err'][var][j], 
                    regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index[j], tos, var_units = None)

            elif var == 'thetao':
                window_thetao = 50
                thetao[j] = Dataset['thetao'].sel(time=slice(str(int(max_year_dtgr.dt.year)-time_window+window_thetao)+'-01-01', sup_year), lat=slice(-5, 5), lon=slice(125, 280)).compute() # 125E - 80W
                # Eseguo una media spaziale     
                thetao[j] = thetao[j].mean('time').mean('lat')
                ###################################################################################################################################
                #################################################### (c) Variables saving #########################################################            
                with open(common_Dir_Output7 + 'Variables/' + f'Task-7-{datasetN}.p', 'ab') as file:
                    pickle.dump([thetao[j]],
                                file)
                    file.close()
                continue # In questo modo non andrà a salvare le variaibili (inenistenti) in fondo a questo 'if'

            elif var == 'tas':
                tas = Dataset['tas'].sel(time=slice(inf_year, sup_year), lat=slice(-5, 5), lon=slice(190, 240)).compute()
                # Calcolo della Regressione
                (regression_product_dict['trend'][var][j], 
                    regression_product_dict['intercept'][var][j], 
                    regression_product_dict['trend_err'][var][j], 
                    regression_product_dict['intercept_err'][var][j], 
                    regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index[j], tas, var_units = None)
                
            elif var == 'ua':
                ua850 = Dataset['ua'].sel(time=slice(inf_year, sup_year), lat=slice(-5, 5), lon=slice(190, 240), plev=85000).compute()
                ua200 = Dataset['ua'].sel(time=slice(inf_year, sup_year), lat=slice(-5, 5), lon=slice(190, 240), plev=20000).compute()
                ua = [ua850, ua200]
                ua_names = ["ua850", "ua200"]
                # Calcolo della Regressione
                for k in range(len(ua)):
                    (regression_product_dict['trend'][var][j], 
                        regression_product_dict['intercept'][var][j], 
                        regression_product_dict['trend_err'][var][j], 
                        regression_product_dict['intercept_err'][var][j], 
                        regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index[j], ua[k], var_units = None)
                    
                    print("\nStart saving the evaluated variables: ", ua_names[k])
                    with open(common_Dir_Output6 + 'Variables/' + f'Task-6_' + ua_names[k] + f'-Products-{datasetN}.p', 'ab') as file:
                        pickle.dump([regression_product_dict['trend'][var][j], 
                                    regression_product_dict['intercept'][var][j], 
                                    regression_product_dict['trend_err'][var][j], 
                                    regression_product_dict['intercept_err'][var][j], 
                                    regression_product_dict['pval'][var][j]],
                                    file)
                        file.close()
                    print("\n","End saving the evaluated variables")

                print("End of time and space variable selection and Regression evaluation")
                time_end_b[j] = datetime.now()
                print("time and space ", var, " (ua850 + ua200) selection and Regression for Dataset_{}, completed in {}".format(datasetN).format(time_end_b[j]-time_begin_b[j]))
                continue


            elif var == 'pr':
                pr = Dataset['pr'].sel(time=slice(inf_year, sup_year), lat=slice(-5, 5), lon=slice(190, 240)).compute()
                # Calcolo della Regressione
                (regression_product_dict['trend'][var][j], 
                    regression_product_dict['intercept'][var][j], 
                    regression_product_dict['trend_err'][var][j], 
                    regression_product_dict['intercept_err'][var][j], 
                    regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index[j], pr, var_units = None)

            elif var == 'zg':
                # Seleziono la stagione: 'DJF'
                Dataset_seasonal = Dataset.sel(time=Dataset.time.dt.month.isin([12, 1, 2])).groupby('time.year').mean('time')
                zg500 = Dataset_seasonal['zg'].sel(lat=slice(-5, 5), lon=slice(190, 240), plev=50000).compute()
                # Resampling annuale dell'indice e calcolo della media
                nino_index_resampled[j] = nino_index[j].resample(time="Y").mean()
                # Calcolo della Regressione
                (regression_product_dict['trend'][var][j], 
                    regression_product_dict['intercept'][var][j], 
                    regression_product_dict['trend_err'][var][j], 
                    regression_product_dict['intercept_err'][var][j], 
                    regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index_resampled[j], zg500, var_units = None)
            
            print("End of time and space variable selection and Regression evaluation")
            time_end_b[j] = datetime.now()
            print("time and space ", var, " selection and Regression for Dataset_{}, completed in {}".format(datasetN).format(time_end_b[j]-time_begin_b[j]))
            
            ######################################################### Fine punto 2(b) #########################################################
            ###################################################################################################################################
            
            ###################################################################################################################################
            #################################################### (c) Variables saving #########################################################


            print("\nStart saving the evaluated variables")
            with open(common_Dir_Output6 + 'Variables/' + f'Task-6_{var}-Products-{datasetN}.p', 'ab') as file:
                pickle.dump([regression_product_dict['trend'][var][j], 
                            regression_product_dict['intercept'][var][j], 
                            regression_product_dict['trend_err'][var][j], 
                            regression_product_dict['intercept_err'][var][j], 
                            regression_product_dict['pval'][var][j]],
                            file)
                file.close()

            print("\n","End saving the evaluated variables")

            ######################################################### Fine punto 2(c) #########################################################
            ###################################################################################################################################
            
            print("\nEvaluation of ", var, f"for Dataset_{datasetN}", " took {}".format(time_end_b[j]-time_begin_a[j]))
        time_var[i] = (time_end_b[dataset_names_iteration-1]-time_begin_a[0])
        print("\nOverall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))
    
    ######################################################## Fine del Main Loop ###############################################################

    ### Recap dei tempi impiegati
    print("\nTiming recap\n")
    for i, var in enumerate(var_names):
        print("Overall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))
    
    print("\nOverall evaluation of the script was completed in {}".format(np.sum(time_var)))






elif (user_input==2):
    
    #################################################################################################################################################
    ########################################################## Acquisizione Dataset #################################################################
    #################################################################################################################################################

    ###############################################################
    # 1: (a) Get Paths;
    #    (b) Variables assegnation;
    #    (c) Directories building: directories's dictionary creation.
    #    (d) Regression building: regression's dictionary creation.
    ###############################################################

    ###################################################################################################################################
    ########################################################## (a) Get paths ##########################################################

    # Common dirs
    common_Dir = '/nas/'
    common_Dir_Output6 = '/home/montanarini/output/Task-6/'
    common_Dir_Output7 = '/home/montanarini/output/Task-7/'
    common_Dir_Output8 = '/home/montanarini/output/Task-8/'
    Dir_Output_task5 = '/home/montanarini/output/Task-5/Variables/'

    # Directories multi-variables Datasets
    Dir_hist = 'BOTTINO/CMIP6/LongRunMIP/EC-Earth-Consortium/EC-Earth3/historical/r4i1p1f1/'
    Dir_obs = '/home/fabiano/'
    Dir_pi = 'archive_CMIP6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/piControl/'
    Dir_BOTTINI = 'BOTTINO/irods_data/stabilization-{}/r1i1p1f1/'
    ######################################################### Fine punto 1(a) #########################################################
    ###################################################################################################################################

    ###################################################################################################################################
    ############################################ (b) Dataset's single Variables assegnation ###########################################

    ### singles's variabiles directories ----> Questa parte è necessaria solamente per il 1° metodo di riempimento del dizionario, mentre è inutile nel 2°

    # Bottini
    dir_bxxx_tos = 'Omon_r25/tos/*.nc'
    #dir_b_sos = 'Omon/sos/*.nc'
    dir_bxxx_thetao = 'Omon_r25/thetao/*.nc'
    dir_bxxx_tas = 'Amon_r25/tas/*.nc'
    dir_bxxx_ua = 'Amon_r25/ua/*.nc'
    dir_bxxx_pr = 'Amon_r25/pr/*.nc'
    dir_bxxx_zg = 'Amon_r25/zg/*.nc'

    dir_bxxx = '{}/*.nc'

    # Pi-Control (no salinity)
    dir_pi_tos = 'ocean/Omon/r1i1p1f1_r25/tos/*.nc'
    dir_pi_thetao = 'ocean/Omon/r1i1p1f1_r25/thetao/*.nc'
    dir_pi_tas = 'atmos/Amon/r1i1p1f1_r25/tas/*.nc'
    dir_pi_ua = 'atmos/Amon/r1i1p1f1_r25/ua/*.nc' 
    dir_pi_pr = 'atmos/Amon/r1i1p1f1_r25/pr/*.nc'
    dir_pi_zg = 'atmos/Amon/r1i1p1f1_r25/zg500/*.nc'

    # Historic (no salinity)
    dir_hist_tos = 'Omon_r25/tos/*.nc'
    dir_hist_thetao = 'Omon_r25/thetao/*.nc'
    dir_hist_tas = 'Amon_r25/tas/*.nc'
    dir_hist_ua = 'Amon_r25/ua/*.nc'
    dir_hist_pr = 'Amon_r25/pr/*.nc'
    dir_hist_zg = 'Amon_r25/zg500/*.nc'

    # Observed (only tos/sst)
    dir_obs_tos = 'HadISST_sst_25.nc'



    ### Definisco i set di parole per la sostituzione nel path e nella variabile della stringa

    # Datasets
    dataset_names = ['b990','b025', 'b050', 'b065', 'b080', 'b100', 'pi-control', 'hist', 'obs']
    dataset_names_iteration = len(dataset_names)

    # Completamento dei path dei bottini
    bottini_paths = ['hist-1990','ssp585-2025', 'ssp585-2050', 'ssp585-2065', 'ssp585-2080', 'ssp585-2100']

    # Definisco le liste di variaibli Oceaniche e Atmosferiche
    var_ocean = ['tos', 'thetao']
    var_atm = ['tas', 'ua', 'pr', 'zg']

    var_names = var_ocean + var_atm
    ######################################################### Fine punto 1(b) #########################################################
    ###################################################################################################################################

    ###################################################################################################################################
    ######################################## (c) Directories building: dictionary creation ############################################

    ### Definisco le liste di path dei vari Datasets per ogni variabile ---> ogni lista conterrà i path di tutti i Datasets per la specifica variabile

    ## 1° metodo ##
    # --> Metodo esplicito, formalmente più corretto, utilizza una struttura più organizzata, intuitiva e facilmente aggiornabile; necessita tuttavia di più righe di codice (52-84) ed è più ripetitivo
    directories = {
    'tos': [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index('tos')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_tos] + [common_Dir + Dir_hist + dir_hist_tos] + [Dir_obs + dir_obs_tos],
    'thetao': [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index('thetao')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_thetao] + [common_Dir + Dir_hist + dir_hist_thetao],
    'tas': [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('tas')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_tas] + [common_Dir + Dir_hist + dir_hist_tas],
    'ua': [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('ua')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_ua] + [common_Dir + Dir_hist + dir_hist_ua],
    'pr': [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('pr')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_pr] + [common_Dir + Dir_hist + dir_hist_pr],
    'zg': [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('zg')]) for bottinoP in bottini_paths]
                + [common_Dir + Dir_pi + dir_pi_zg] + [common_Dir + Dir_hist + dir_hist_zg]
    }
    ## 2° metodo ##
    # --> Metodo implicito, più compatto (quindi meno dispersivo), è meno intuibile e non mostra immediatamente le chiavi del dizionario
    directories = {}

    for var in var_names:
        if var in var_ocean:  # Se la variabile è oceanica
            directories[var] = [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index(var)]) for bottinoP in bottini_paths]
            # Aggiunta dei path non bottino
            if var == 'tos':
                    directories[var] += [common_Dir + Dir_pi + dir_pi_tos, common_Dir + Dir_hist + dir_hist_tos, Dir_obs + dir_obs_tos]
            else:
                    directories[var] += [common_Dir + Dir_pi + 'ocean/Omon/r1i1p1f1_r25/' + var + '/*.nc', common_Dir + Dir_hist + 'Omon_r25/' + var + '/*.nc']
        else:
            # Se la variabile è atmosferica
            directories[var] = [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index(var)]) for bottinoP in bottini_paths]
            # Aggiunta dei path non bottino
            if var =='zg':
                directories[var] += [common_Dir + Dir_pi + 'atmos/Amon/r1i1p1f1_r25/' + var + '500/*.nc', common_Dir + Dir_hist + 'Amon_r25/' + var + '/*.nc']
            else:
                directories[var] += [common_Dir + Dir_pi + 'atmos/Amon/r1i1p1f1_r25/' + var + '/*.nc', common_Dir + Dir_hist + 'Amon_r25/' + var + '/*.nc']
    
    ######################################################### Fine punto 1(c) #########################################################
    ###################################################################################################################################


    ###################################################################################################################################
    ############################### (d) Regression array building: regressions's dictionary creation ##################################
    ### Creazione di un dizionario contenente un dizionario annidato per archiviare le liste che conterranno i risultati della regressione (suddivisi in 5 diversi prodotti)
    regression_product_dict = {}

    regression_products_list = ['trend', 'intercept', 'trend_err', 'intercept_err', 'pval'] # Nomi dei prodotti generati dalla funzione che calcola la regressione
    var_names_sel = ['tos', 'tas', 'ua850', 'ua200', 'pr', 'zg500'] # Nome delle variabili dopo la selezione dei livelli ('thetao' non figura perchè su di essa non calcoliamo la reg.)

    for product in regression_products_list:
        regression_product_dict[product] = {}
        for var in var_names:
            if var == 'tos':
                regression_product_dict[product][var] = [None] * dataset_names_iteration
            else:
                regression_product_dict[product][var] = [None] * (dataset_names_iteration - 1)

    var_names_sel.insert(1, 'thetao') # lista su cui vado ad eseguire il main loop

    # Composite dictionary declaration
    composite = {}
    oscilaltion = ['nino', 'nina']
    for phase in oscilaltion:
        composite[phase] = {}
        for var in var_names:
            if var == 'tos':
                composite[phase][var] = [None] * dataset_names_iteration
            else:
                composite[phase][var] = [None] * (dataset_names_iteration - 1)


    # Creo delle liste per salvare i tempi impiegati per ogni dataset ('_a' è per il punto 2 (a), '_b' è per il punto 2 (b))
    time_begin_a = [None] * dataset_names_iteration
    time_end_a = [None] * dataset_names_iteration
    time_begin_b = [None] * dataset_names_iteration
    time_end_b = [None] * dataset_names_iteration
    time_var = [None] * len(var_names)
    ######################################################### Fine punto 1(d) #########################################################
    ###################################################################################################################################

    #################################################################################################################################################
    ############################################################### Main Loop #######################################################################
    #################################################################################################################################################

    ###############################################################
    # 2: main loop over dictionary keys thus on the Dataset's variables names; second nested loop over the Dataset names and dataset's values
    #    (a) Discrimination based on the Dataset of origin;
    #    (b) Discrimination based on the Dictionary' keys (Dataset's variables) and Regression calculation.
    #    (c) Variable saving.
    ###############################################################

    """
        Nota:      
        metodo "globals()", restituisce il dizionario dei simboli globali.
        Accedo a variabili il cui nome era costruito dinamicamente concatenando una stringa (dir_pi_ o dir_hist_) con il valore della variabile var.
        Utilizzando globals(), puoi ottenere il valore della variabile usando il suo nome come chiave nel dizionario restituito da globals().
        globals(): Restituisce un dizionario che rappresenta lo spazio dei nomi globale. In altre parole, contiene tutte le variabili globali
            definite nel tuo programma. Le chiavi del dizionario sono i nomi delle variabili, e i valori sono i corrispondenti oggetti.
        locals(): Restituisce un dizionario che rappresenta lo spazio dei nomi locale. Questo spazio dei nomi è specifico di una funzione o di un blocco di codice.
            Le chiavi del dizionario sono i nomi delle variabili locali, e i valori sono i corrispondenti oggetti.
    """

    nino_index = [None]*dataset_names_iteration # Lista per memorizzare gli indici dei vari Dataset
    nino_index_nino = [None]*dataset_names_iteration
    nino_index_nina = [None]*dataset_names_iteration
    nino_index_resampled = [None]*(dataset_names_iteration-1)

    #thetao = [None]*(dataset_names_iteration-1) # Lista per memorizzare i termoclini dei vari Dataset, non necessaria

    time_window = 100 # ultimi n anni del Dataset (non per hist e obs)
    time_rm = 360 # Mesi in cui si applica la RM
    window_thetao = 50 # Ultimi n anni in cui si calcola la termoclina
    print("\nThe following set of parameters was selected:",
          "\nLast n years of the bottino runs: ", time_window,
          "\nRunning Mean time, in months (for the historical detrending): ", time_rm,
          "\nLast n years for the evaluation of the thermocline", window_thetao)

    flag = bool = True # Flag per eseguire le cose una volta sola



    ### Iterazione sulla chiave del dizionario (=sulle variabili contenute nei Dataset)
    for i, var in enumerate(var_names):
        
        # Seleziona solo la variaibile n ed escludi le altre
        # if var != 'thetao':
        #     continue

        ### Iterazione sul valore del dizionario (=sulle directories delle variabili contenute nei Dataset) e sui Datasets stessi: l'indice 'i' indica il Dataset

        for j, datasetN in enumerate(dataset_names):


            if (var != 'tos' and j == dataset_names_iteration -1): # skip each iteration for Dataset 'obs' except for 'tos'
                time_begin_a[j] = datetime.now()
                time.sleep(1)
                time_end_b[j] = datetime.now()
                continue

            Dataset = f'Dir_{datasetN}' # ora 'Dataset' è una stringa, dopo diventerà un xarray.Dataset
            # Assegna la stringa formattata alla variabile con il nome appropriato
            globals()[Dataset] = directories[var][j]   # Funziona ugualmente sostituendo a 'globals()[Dataset]' --> 'path_formattato'

            print(f"\nInizia lelaborazione di: Dataset_{datasetN}\nCorresponding path: ", directories[var][j])
            

            print("\nStart of Dataset opening/upload")
            time_begin_a[j] = datetime.now()

            ###################################################################################################################################
            #################################### (a) Discrimination based on the Dataset of origin ############################################

            if datasetN in ['b990','b025', 'b050', 'b065', 'b080', 'b100']: #j < 6
                ####### BOTTINI #######

                ### Apertura dataset

                Dataset = xr.open_mfdataset(globals()[Dataset], use_cftime=True) # globals()[Dataset] <----> directories[var][j]


                
            elif datasetN in ['pi-control']: #j==6
                ##### Pi-Control ######

                ### Apertura dataset
                
                Dataset = xr.open_mfdataset(globals()[Dataset], decode_times=True, use_cftime=True)
                                
                # Riassegnazione dimensione tempo
                if var in ['thetao']:
                    Dataset['time'] = xr.date_range(start='1850-01-01', end='2349-03-01', periods=None, freq='M', tz=None, normalize=True, name=None, calendar='gregorian', use_cftime=True)
                else:
                    Dataset['time'] = xr.date_range(start='1850-01-01', end='2351-01-01', periods=None, freq='M', tz=None, normalize=True, name=None, calendar='gregorian', use_cftime=True)

                

            elif datasetN in ['hist']: #j==7
                ######### HIST ########
                
                ### Apertura dataset

                Dataset = xr.open_mfdataset(globals()[Dataset])

                

            else: #datasetN in ['obs'] or :j==8
                ######### OBS #########

                ### Apertura dataset

                Dataset = xr.open_dataset(globals()[Dataset])
                
                # Riassegnazione nome campo 
                Dataset = Dataset.rename({'sst': 'tos'})


            

            ### Discriminazione del range temporale:
                
            max_year_dtgr = Dataset['time'].max()
            min_year_dtgr = Dataset['time'].min()
            # sup_year_rm = str(int(max_year_dtgr.dt.year)-half_rm_window)+'-01-01'
            # inf_year_rm = str(int(min_year_dtgr.dt.year)+half_rm_window)+'-01-01'
            sup_year = str(int(max_year_dtgr.dt.year))+'-01-01'
            inf_year = str(int(min_year_dtgr.dt.year))+'-01-01' # Estremo solo per Hist e Obs
            inf_year_bott = str(int(max_year_dtgr.dt.year)-time_window)+'-01-01'

        

            ### Recupero dell' Indice Nino 3.4

            filename = 'Task-5_Dataset-{}.p'.format(datasetN)
            nino_index[j] = pickle.load(open(Dir_Output_task5 + filename, 'rb'))[0]
            # with open(Dir_Output_task5 + f'Task-5_Dataset-{datasetN}-Subset.p', 'ab') as file:
            #     pickle.dump([nino_index[j]],
            #                 file)
            #     file.close()

            
            ### Limitazione del range temporale per il Dataset(non per Hist e Obs)

            if datasetN not in ['hist', 'obs']:
                Dataset = Dataset.sel(time=slice(inf_year_bott, sup_year)).compute()
                # Seleziono il periodo desiderato dell'indice
                nino_index[j] = nino_index[j].sel(time=slice(inf_year_bott, sup_year)).compute()
            else:
                Dataset = Dataset.sel(time=slice(inf_year, sup_year)).compute()
                # Seleziono il periodo desiderato dell'indice
                nino_index[j] = nino_index[j].sel(time=slice(inf_year, sup_year)).compute()

            # Indici per la Composite
            nino_index_nino[j] = nino_index[j].where(nino_index[j].rolling(time=6, center=True).mean().dropna("time")>=0.4)
            nino_index_nina[j] = nino_index[j].where(nino_index[j].rolling(time=6, center=True).mean().dropna("time")<=-0.4)

    
            print("End of Dataset elaboration")
            time_end_a[j] = datetime.now()
            print("Elaboration of Dataset_{} completed in {}".format(datasetN).format(time_end_a[j]-time_begin_a[j]))

            ######################################################### Fine punto 2(a) #########################################################
            ###################################################################################################################################
        
            ###################################################################################################################################
            ################ (b) Discrimination based on the Dictionary' keys (Dataset's variables) and Recgression calculation ###############

            print("\nStart of time and space variable selection and Regression evaluation")
            time_begin_b[j] = datetime.now()

            if var == 'tos':
                # Isola le singole variabili
                campo = Dataset['tos'].sel(lat=slice(-85, 85)).compute() #lat=slice(-5, 5), lon=slice(190, 240)
                # Maschera continentale
                if flag == True:
                    continental_mask = np.isnan(campo)[0,...]
                    flag = False

            elif var == 'thetao':
                campo = Dataset['thetao'].sel(time=slice(str(int(max_year_dtgr.dt.year)-time_window+window_thetao)+'-01-01', sup_year), lat=slice(-5, 5), lon=slice(125, 280)).compute() # 125E - 80W
                nino_index_nino[j] = nino_index_nino[j].sel(time=slice(str(int(max_year_dtgr.dt.year)-time_window+window_thetao)+'-01-01', sup_year)).compute()
                nino_index_nina[j] = nino_index_nina[j].sel(time=slice(str(int(max_year_dtgr.dt.year)-time_window+window_thetao)+'-01-01', sup_year)).compute()
                
                # Detrending del campo
                #---------------------------------------------------------newer
                # Trend storico
                running_mean = campo.sel(lat=slice(-85,85)).rolling(time=time_rm, min_periods=1, center=True).mean().dropna("time") # Running Mean
                rm_trend = ctl.global_mean(running_mean.groupby('time.month')-running_mean.groupby('time.month').mean('time')) # Trend storico dato dalla RM
                campo_detrended = campo - rm_trend
                # Seasonal cycle
                anomaly_detrended_decycled = (campo_detrended.groupby('time.month') - 
                                campo_detrended.groupby('time.month').mean('time'))
                campo_detrended_decycled = anomaly_detrended_decycled
                #--------------------------------------------------------------older
                # running_mean = campo.sel(lat=slice(-85,85)).rolling(time=time_rm, center=True, min_periods=1).mean() # Running Mean
                # RMTrend = (running_mean.groupby('time.month')-running_mean.groupby('time.month').mean('time')) # Trend storico
                # campo_detrended = campo-RMTrend
                # anomaly_detrended_decycled = (campo_detrended.groupby('time.month') - campo_detrended.groupby('time.month').mean('time')) # Rimoz. seasonal cycle
                # campo_detrended_decycled = campo_detrended.mean('time') + anomaly_detrended_decycled 
                #----------------------------------------------------------------

                # Equalizzazione del range temporale con l'indice
                campo_detrended_decycled = campo_detrended_decycled.sel(time=slice(str(int(max_year_dtgr.dt.year)-window_thetao)+'-01-01', sup_year)).compute()
                #anomaly_detrended_decycled = anomaly_detrended_decycled.sel(time=slice(str(int(max_year_dtgr.dt.year)-window_thetao)+'-01-01', sup_year)).compute()
                
                # Eseguo una media spaziale
                campo_detrended_decycled = campo_detrended_decycled.mean('time').mean('lat') # Campo + Anomalia
                #anomaly_detrended_decycled = anomaly_detrended_decycled.mean('lat') # Anomalia
                
                # Calcolo delle Composite
                for phase in oscilaltion:
                    if phase == 'nino':
                        composite[phase]['thetao'][j] = (xr.where(nino_index_nino[j], campo_detrended_decycled, np.nan)).mean('time')
                    elif phase == 'nina':
                        composite[phase]['thetao'][j] = (xr.where(nino_index_nina[j], campo_detrended_decycled, np.nan)).mean('time')
                
               
                ###################################################################################################################################
                #################################################### (c) Variables saving #########################################################            
                with open(common_Dir_Output7 + 'Variables/' + f'Task-7-{datasetN}.p', 'ab') as file:
                    pickle.dump([campo_detrended_decycled,
                                 composite['nino']['thetao'][j],
                                 composite['nina']['thetao'][j]],
                                file)
                    file.close()
                continue # In questo modo non andrà a salvare le variaibili (inesistenti) in fondo a questo 'if'

            elif var == 'tas':
                campo = Dataset['tas'].sel(lat=slice(-85, 85)).compute()
                
            elif var == 'ua':
                ua850 = Dataset['ua'].sel(lat=slice(-85, 85), plev=85000).compute()
                ua200 = Dataset['ua'].sel(lat=slice(-85, 85), plev=20000).compute()
                ua = [ua850, ua200]
                ua_names = ["ua850", "ua200"]
                ## Calcolo della Regressione
                for k in range(len(ua)):
                    # Detrending del campo
                    running_mean = ua[k].sel(lat=slice(-85,85)).rolling(time=time_rm, center=True, min_periods=1).mean() # Running Mean
                    RMTrend = (running_mean.groupby('time.month')-running_mean.groupby('time.month').mean('time'))
                    campo_detrended = ua[k]-RMTrend
                    # Rimoz. seasonal cycle
                    anomaly_detrended_decycled = (campo_detrended.groupby('time.month') - 
                        campo_detrended.groupby('time.month').mean('time'))
                    campo_detrended_decycled = campo_detrended.mean('time') + anomaly_detrended_decycled
                    # Equalizzazione del range temporale con l'indice
                    if datasetN not in ['hist', 'obs']:
                        anomaly_detrended_decycled = anomaly_detrended_decycled.sel(time=slice(inf_year_bott, sup_year_rm)).compute()
                        campo_detrended_decycled = campo_detrended_decycled.sel(time=slice(inf_year_bott, sup_year_rm)).compute()
                    else:
                        anomaly_detrended_decycled = anomaly_detrended_decycled.sel(time=slice(inf_year_rm, sup_year_rm)).compute()      
                        campo_detrended_decycled = campo_detrended_decycled.sel(time=slice(inf_year_rm, sup_year_rm)).compute()      
                    # Regressione              
                    (regression_product_dict['trend'][var][j], 
                        regression_product_dict['intercept'][var][j], 
                        regression_product_dict['trend_err'][var][j], 
                        regression_product_dict['intercept_err'][var][j], 
                        regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index[j], anomaly_detrended_decycled, var_units = None)
                    # Calcolo delle Composite
                    for phase in oscilaltion:
                        print(f"Processing phase: {phase}, variable: {var}")
                        if phase == 'nino':
                            composite[phase][var][j] = (xr.where(nino_index[j]>=0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                        elif phase == 'nina':
                            composite[phase][var][j] = (xr.where(nino_index[j]<=-0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                    
                    print("\nStart saving the evaluated variables: ", ua_names[k])
                    with open(common_Dir_Output6 + 'Variables/' + f'Task-6_' + ua_names[k] + f'-Products-{datasetN}.p', 'ab') as file:
                        pickle.dump([regression_product_dict['trend'][var][j], 
                                    regression_product_dict['intercept'][var][j], 
                                    regression_product_dict['trend_err'][var][j], 
                                    regression_product_dict['intercept_err'][var][j], 
                                    regression_product_dict['pval'][var][j]],
                                    file)
                        file.close()
                    with open(common_Dir_Output8 + 'Variables/' + f'Task-8_' + ua_names[k] + f'-Composite-{datasetN}.p', 'ab') as file:
                        pickle.dump([composite['nino'][var][j], 
                                    composite['nina'][var][j]], 
                                    file)
                        file.close()
                    print("\nEnd saving the evaluated variables")

                print("End of time and space variable selection and Regression evaluation")
                time_end_b[j] = datetime.now()
                print("time and space ", var, " (ua850 + ua200) selection and Regression for Dataset_{} completed in {}".format(datasetN).format(time_end_b[j]-time_begin_b[j]))
                continue


            elif var == 'pr':
                campo = Dataset['pr'].sel(lat=slice(-85, 85)).compute()

            elif var == 'zg':
                campo = Dataset['zg'].sel(lat=slice(-85, 85), plev=50000).compute()
                
                # Detrending storico del campo
                running_mean = campo.sel(lat=slice(-85,85)).rolling(time=time_rm, center=True, min_periods=1).mean() # Running Mean
                RMTrend = (running_mean.groupby('time.month')-running_mean.groupby('time.month').mean('time')) # Trend storico
                campo_detrended = campo-RMTrend
                # Equalizzazione del range temporale con l'indice
                if datasetN not in ['hist', 'obs']:
                    campo_detrended = campo_detrended.sel(time=slice(inf_year_bott, sup_year_rm)).compute()# campo_detrended.sel(time=slice(str(int(max_year_dtgr.dt.year)-time_window), str((int(max_year_dtgr.dt.year))))).compute() # inf_year_bott; sup_year_rm
                else:
                    campo_detrended = campo_detrended.sel(time=slice(inf_year_rm, sup_year_rm)).compute()# campo_detrended.sel(time=slice(str(int(min_year_dtgr.dt.year)), str((int(max_year_dtgr.dt.year))))).compute() # inf_year_rm; sup_year_rm
                
                # Seleziono la stagione: 'DJF'
                campo_detrended_decycled = campo_detrended.sel(time=campo_detrended.time.dt.month.isin([12, 1, 2])).groupby('time.year').mean('time') # Si rimuove il Ciclo stagionale
                anomaly_detrended_decycled = campo_detrended_decycled - campo_detrended.mean('time')
                #campo = Dataset_seasonal['zg'].sel(lat=slice(-85, 85), plev=50000).compute()
                # Resampling annuale dell'indice e calcolo della media
                nino_index[j] = nino_index[j].resample(time="Y").mean()

                # Calcolo delle Composite
                for phase in oscilaltion:
                    if phase == 'nino':
                        composite[phase][var][j] = (xr.where(nino_index[j]>=0.4, anomaly_detrended_decycled, np.nan)).mean()
                    elif phase == 'nina':
                        composite[phase][var][j] = (xr.where(nino_index[j]<=-0.4, anomaly_detrended_decycled, np.nan)).mean()
            #--------------------------------------------------------------------------------------------------------------------------------------#
                           
            
            if var != 'zg':
                ### Detrending storico del campo
                running_mean = campo.sel(lat=slice(-85,85)).rolling(time=time_rm, center=True, min_periods=1).mean() # Running Mean
                RMTrend = (running_mean.groupby('time.month')-running_mean.groupby('time.month').mean('time')) # Trend storico
                campo_detrended = campo-RMTrend
                ### Rimoz. seasonal cycle
                anomaly_detrended_decycled = (campo_detrended.groupby('time.month') - # quella che ritrovo è un'anomalia
                                                campo_detrended.groupby('time.month').mean('time'))
                campo_detrended_decycled = campo_detrended.mean('time') + anomaly_detrended_decycled # Campo senza trend storico e senza seasonal cycle.
                                                                                                        # Viene calcolato ma non viene salvato
                
                ### Equalizzazione del range temporale con l'indice
                if datasetN not in ['hist', 'obs']:
                    campo_detrended_decycled = campo_detrended_decycled.sel(time=slice(inf_year_bott, sup_year_rm)).compute()
                    anomaly_detrended_decycled = anomaly_detrended_decycled.sel(time=slice(inf_year_bott, sup_year_rm)).compute()
                else:
                    anomaly_detrended_decycled = anomaly_detrended_decycled.sel(time=slice(inf_year_rm, sup_year_rm)).compute()
                    campo_detrended_decycled = campo_detrended_decycled.sel(time=slice(inf_year_rm, sup_year_rm)).compute()
                ### Calcolo delle Composite
                for phase in oscilaltion:
                    print(f"Processing phase: {phase}, variable: {var}")
                    if phase == 'nino':
                        composite[phase][var][j] = (xr.where(nino_index[j]>=0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                    elif phase == 'nina':
                        composite[phase][var][j] = (xr.where(nino_index[j]<=-0.4, anomaly_detrended_decycled, np.nan)).mean('time')
            
            ### Calcolo della Regressione
            (regression_product_dict['trend'][var][j], 
                regression_product_dict['intercept'][var][j], 
                regression_product_dict['trend_err'][var][j], 
                regression_product_dict['intercept_err'][var][j], 
                regression_product_dict['pval'][var][j]) = ctl.calc_trend_climatevar(nino_index[j], anomaly_detrended_decycled, var_units = None)
            
            
            print("End of time and space variable selection and Regression evaluation")
            time_end_b[j] = datetime.now()
            print("time and space ", var, " selection and Regression for Dataset_{} completed in {}".format(datasetN).format(time_end_b[j]-time_begin_b[j]))
            
            ######################################################### Fine punto 2(b) #########################################################
            ###################################################################################################################################
            
            ###################################################################################################################################
            #################################################### (c) Variables saving #########################################################


            print("\n","Start saving the evaluated variables")
            with open(common_Dir_Output6 + 'Variables/' + f'Task-6_{var}-Products-{datasetN}.p', 'ab') as file:
                pickle.dump([regression_product_dict['trend'][var][j], 
                            regression_product_dict['intercept'][var][j], 
                            regression_product_dict['trend_err'][var][j], 
                            regression_product_dict['intercept_err'][var][j], 
                            regression_product_dict['pval'][var][j]],
                            file)
                file.close()
            with open(common_Dir_Output8 + 'Variables/' + f'Task-8_{var}-Composite-{datasetN}.p', 'ab') as file:
                pickle.dump([composite['nino'][var][j], 
                            composite['nina'][var][j]], 
                            file)
                file.close()

            print("\n","End saving the evaluated variables")

            ######################################################### Fine punto 2(c) #########################################################
            ###################################################################################################################################
            
            print("\nEvaluation of ", var, f"for Dataset_{datasetN}", " took {}".format(time_end_b[j]-time_begin_a[j]))
        time_var[i] = (time_end_b[dataset_names_iteration-1]-time_begin_a[0])
        print("\nOverall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))

    ######################################################## Fine del Main Loop ###############################################################

    ### Recap dei tempi impiegati
    print("\nTiming recap")
    for i, var in enumerate(var_names):
        print("\nOverall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))
    
    print("\nOverall evaluation of the script was completed in {}".format(np.sum(time_var)))

else:
    raise ValueError('Given input is not allowed')

print("\nEnd of the script")
sys.exit()