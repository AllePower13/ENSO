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
   
   NB: Sia all'Indice che ai Datasets è stato rimosso il trend storico con il metodo della Running Mean e anche il ciclo stagionale con la media sul tempo.

- Calcolo della termoclina negli ultimi 50 anni dei vari dataset presenti (Osservato escluso in quanto tale dato è mancante)

- Calcolo della Composite Nino e Nina.

Nota: 
Lo script era suddiviso in due macro-rami, ora ridotto ad uno solo, l'altro è stato eliminato perchè superato.
Viene proposto il regrid di ogni singolo dataset sul dataset dello storico prodotto da Ec-Earth di Temperatura osservata superficiale degli oceani (Dataset_hist_tos),
dunque in una griglia 180x360, per omologare le griglie di tutti i dataset. Questo perchè è necessario assicurarsi che tutti i dataset condividano la stessa griglia lat-lon per evitare crash.
Questa operazione è particolarmente lunga da svolgere da un punto di vista computazionale, è meglio eseguirla fuori dallo script python, con altri strumenti come CDO o altre librerie in grado di parallelizzare la cosa.
"""
### Import libraries
# System libraries
import sys
import os
# General data analisys libraries 
import numpy as np
import xarray as xr
import pickle
# Librari for treads managements
import threading
# Time library
import time
import cftime
from datetime import datetime
# My and Fede functions libraries
import AllePowerFunctions as  apf
from climtools import climtools_lib as ctl

nome_script = os.path.basename(__file__)
print(nome_script, " has been launched on date:", datetime.now(), "\n")
print("Choose which branch to run:",
      "\n1 := Obsolete.",
      "\n2 := Currently most developed and omnicomrehensive branch.",
      )
   
# Flag per eseguire le cose una volta sola
bool = True

# Prepara una lista per il primo input
input_branch = []
input_thread_branch = threading.Thread(target=apf.get_user_input, args=(input_branch,))
input_thread_branch.start()

# Aspetta che il thread finisca di ottenere l'input
input_thread_branch.join(timeout=10)

# Se il thread è ancora attivo, significa che l'utente non ha inserito nulla
if input_thread_branch.is_alive():
    input_branch.append(2)  # Imposta il valore di default
    print("Time over. Default value of: ", input_branch[0], " will be used.")

# Verifica imput utente
if ((input_branch[0] != 1) and (input_branch[0] != 2)):
    raise ValueError("Given input", input_branch[0], "is not allowed")

# Accedo al valore del primo input dalla lista
if input_branch: # Controlla se la lista non è vuota
    metodo_selezionato_branch = input_branch[0]
    print(f"Hai selezionato il branch: {metodo_selezionato_branch}")
else:
    metodo_selezionato_branch = None
    print("Nessun input valido ricevuto per il branch.")

while bool == True:

    if (metodo_selezionato_branch==1):
        
        print("\nThis method became obsolete.")
        metodo_selezionato_branch = 2
        print("\nYour input has been changed automatically to: ", metodo_selezionato_branch)
        
    elif (metodo_selezionato_branch==2):

        print("Choose wheter to enable or not the detrending:",
        "\n0 := Default, Disabled regridding. Warning! The Datasets must have the same space grid subdivision, better if it is 180x360 or lower",
        "\n1 := Enabled regridding of the Datasets over the Historical Dataset wich has a space grid divided in 180x360 gridpoints",
        )
        # Prepara una lista per il primo input
        input_regrid = []
        input_thread_regrid = threading.Thread(target=apf.get_user_input, args=(input_regrid,))
        input_thread_regrid.start()

        # Aspetta che il thread finisca di ottenere l'input
        input_thread_regrid.join(timeout=10)
                
        # Se il thread è ancora attivo, significa che l'utente non ha inserito nulla
        if input_thread_regrid.is_alive():
            input_regrid.append(0) # Imposta il valore di default
            print("\nTime over. Default value of: ", input_regrid[0], " will be used.")
        
        # Verifica imput utente
        if ((input_regrid[0] != 0) and (input_regrid[0] != 1)):
            raise ValueError("Given input", input_regrid[0], "is not allowed")

        # Accedo al valore del primo input dalla lista
        if input_regrid: # Controlla se la lista non è vuota
            metodo_selezionato_regrid = input_regrid[0]
            print(f"Hai selezionato il regrid: {metodo_selezionato_regrid}")
        else:
            metodo_selezionato_regrid = None
            print("Nessun input valido ricevuto per il metodo.")

        print(f"\nRecap of choose settings:\nHai selezionato il branch: {metodo_selezionato_branch}\nHai selezionato il regrid: {metodo_selezionato_regrid}")

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
        dir_output_task5 = '/home/montanarini/ELNINO/output/Task-5_NinoIndex/Variables/'
        common_Dir_Output6 = '/home/montanarini/ELNINO/output/Task-6_Regressions/'
        common_Dir_Output7 = '/home/montanarini/ELNINO/output/Task-7_Thermocline/'
        common_Dir_Output8 = '/home/montanarini/ELNINO/output/Task-8_Composites/'

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
        dataset_names = ['obs', 'hist', 'pi-control', 'b990','b025', 'b050', 'b065', 'b080', 'b100']
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

        ### Dizionario contenente i path dei vari Datasets per ogni variabile

        ## 1° metodo ##
        # Metodo esplicito, formalmente più corretto, utilizza una struttura più organizzata e compatta, intuitiva e facilmente aggiornabile;
        # unico difetto: per selezionare il Dataset di riferimento si deve ricorrere all'indicizzazione [j], quindi può confondere
        directories = {
        'tos': [Dir_obs + dir_obs_tos] + [common_Dir + Dir_hist + dir_hist_tos] + [common_Dir + Dir_pi + dir_pi_tos] + 
                [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index('tos')]) for bottinoP in bottini_paths],
        'thetao': [common_Dir + Dir_hist + dir_hist_thetao] + [common_Dir + Dir_pi + dir_pi_thetao] + 
                [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index('thetao')]) for bottinoP in bottini_paths],
        'tas': [common_Dir + Dir_hist + dir_hist_tas] + [common_Dir + Dir_pi + dir_pi_tas] + 
                [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('tas')]) for bottinoP in bottini_paths],
        'ua': [common_Dir + Dir_hist + dir_hist_ua] + [common_Dir + Dir_pi + dir_pi_ua] + 
                [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('ua')]) for bottinoP in bottini_paths],
        'pr': [common_Dir + Dir_hist + dir_hist_pr] + [common_Dir + Dir_pi + dir_pi_pr] + 
                [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('pr')]) for bottinoP in bottini_paths],
        'zg': [common_Dir + Dir_hist + dir_hist_zg] + [common_Dir + Dir_pi + dir_pi_zg] + 
                [(common_Dir + Dir_BOTTINI).format(bottinoP) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index('zg')]) for bottinoP in bottini_paths]
        }

        ## 2° metodo ##
        # Metodo implicito, meno compatto, è meno intuitivo, tratta le eccezioni con delle condizioni e non mostra immediatamente la forma del dizionario;
        # permette di di chiamare la directory richiesta con le chiavi del dizionario relative al dataset e la variaibile voluta, senza fare ricorso all'indicizzazione, anche se ritorna una lista che va poi scompattata.
        directories = {}

        for i, datasetN in enumerate(dataset_names):
            directories[datasetN] = {}
            for var in var_names:
                
                if var in var_ocean: # Se la variabile è oceanica
                    # Aggiunta dei path non bottino
                    if var == 'tos':
                        if datasetN in ['obs']:
                            directories[datasetN][var] = [Dir_obs + dir_obs_tos]
                        elif datasetN in ['hist']:
                            directories[datasetN][var] = [common_Dir + Dir_hist + dir_hist_tos]
                        elif datasetN in ['pi-control']:
                            directories[datasetN][var] = [common_Dir + Dir_pi + dir_pi_tos]
                    else:
                        if datasetN in ['obs']:
                            continue
                        elif datasetN in ['hist']:
                            directories[datasetN][var] = [common_Dir + Dir_hist + 'Omon_r25/' + var + '/*.nc']
                        elif datasetN in ['pi-control']:
                            directories[datasetN][var] = [common_Dir + Dir_pi + 'ocean/Omon/r1i1p1f1_r25/' + var + '/*.nc']
                    # Aggiunta dei path bottino
                    if datasetN in ['b990','b025', 'b050', 'b065', 'b080', 'b100']:
                        directories[datasetN][var] = [(common_Dir + Dir_BOTTINI).format(bottini_paths[i-3]) + 'Omon_r25/' + dir_bxxx.format(var_names[var_names.index(var)])]                
                else: # Se la variabile è atmosferica
                    # Aggiunta dei path non bottino
                    if datasetN in ['obs']:
                        continue                    
                    if ((datasetN in ['pi-control']) and (var =='zg')):
                        directories[datasetN][var] = [common_Dir + Dir_pi + 'atmos/Amon/r1i1p1f1_r25/' + var + '500/*.nc']
                    else:
                        if datasetN in ['hist']:
                            directories[datasetN][var] = [common_Dir + Dir_hist + 'Amon_r25/' + var + '/*.nc']
                        elif datasetN in ['pi-control']:
                            directories[datasetN][var] = [common_Dir + Dir_pi + 'atmos/Amon/r1i1p1f1_r25/' + var + '/*.nc']
                        else:
                            # Aggiunta dei path bottino
                            directories[datasetN][var] = [(common_Dir + Dir_BOTTINI).format(bottini_paths[i-3]) + 'Amon_r25/' + dir_bxxx.format(var_names[var_names.index(var)])]
        
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
                regression_product_dict[product][var] = {}
                for datasetN in dataset_names:
                    regression_product_dict[product][var][datasetN] = [None]
                # if var == 'tos':
                #     regression_product_dict[product][var] = [None] * dataset_names_iteration
                # else:
                #     regression_product_dict[product][var] = [None] * (dataset_names_iteration - 1)

        var_names_sel.insert(1, 'thetao') # lista su cui vado ad eseguire il main loop

        # Composite dictionary declaration
        composite = {}
        oscillation = ['nino', 'nina']
        for phase in oscillation:
            composite[phase] = {}
            for var in var_names:
                composite[phase][var] = {}
                for datasetN in dataset_names:
                    composite[phase][var][datasetN] = [None]
                # if var == 'tos':
                #     composite[phase][var] = [None] * dataset_names_iteration
                # else:
                #     composite[phase][var] = [None] * (dataset_names_iteration - 1)


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
        nino_index = {} #[None]*dataset_names_iteration # Lista per memorizzare gli indici dei vari Dataset
        nino_index_nino = {} #[None]*dataset_names_iteration
        nino_index_nina = {} #[None]*dataset_names_iteration
        #nino_index_resampled = [None]*(dataset_names_iteration-1)

        #thetao = [None]*(dataset_names_iteration-1) # Lista per memorizzare i termoclini dei vari Dataset, non necessaria

        # valori da usare 100; 360; 50
        time_window = 100 # ultimi n anni del Dataset (non per hist e obs), default: 100
        time_rm = 360 # Mesi in cui si applica la RM, default: 360
        window_thetao = 50 # Ultimi n anni in cui si calcola la termoclina, default: 50
        print("\nThe following set of parameters was selected:",
            "\n- Last n years of the bottino runs: ", time_window,
            "\n- Running Mean time, in months (for the historical detrending): ", time_rm,
            "\n- Last n years for the evaluation of the thermocline:", window_thetao)

        # Flag per eseguire le cose una volta sola
        flag_continental = True



        ### Iterazione sulle Variabili
        for i, var in enumerate(var_names):
            
            # Seleziona solo la variaibile n ed escludi le altre
            # if var not in ['thetao']:
            #     continue

            print(f"\n# Inizia l'elaborazione della variabile: {var}")

            ### Iterazione sui Dataset

            for j, datasetN in enumerate(dataset_names):

                # Skip each iteration for Dataset 'obs' except for 'tos'
                if ((datasetN == 'obs') and (var != 'tos')):
                    time_begin_a[j] = datetime.now()
                    time.sleep(1)
                    time_end_b[j] = datetime.now()
                    continue

                Dataset = f'DS_{datasetN}' # ora 'Dataset' è una stringa, dopo diventerà un xarray.Dataset
                # Assegna la stringa formattata alla variabile con il nome appropriato
                globals()[Dataset], = directories[datasetN][var]   # Funziona ugualmente sostituendo a 'globals()[Dataset]' --> 'path_formattato'
                print(f"\n## Inizia l'elaborazione di: Dataset_{datasetN}\nCorresponding path: ", globals()[f'DS_{datasetN}'])
                

                print("\nStart of Dataset opening/upload")
                time_begin_a[j] = datetime.now()

                ###################################################################################################################################
                #################################### (a) Discrimination based on the Dataset of origin ############################################

                if datasetN in ['b990','b025', 'b050', 'b065', 'b080', 'b100']: #j > 2
                    ####### BOTTINI #######

                    ### Apertura dataset

                    Dataset = xr.open_mfdataset(globals()[f'DS_{datasetN}'], use_cftime=True) # globals()[Dataset] <----> directories[datasetN][var]


                    
                elif datasetN in ['pi-control']: #j==2
                    ##### Pi-Control ######

                    ### Apertura dataset
                    
                    Dataset = xr.open_mfdataset(globals()[f'DS_{datasetN}'], decode_times=True, use_cftime=True)
                                    
                    # Riassegnazione dimensione tempo
                    if var in ['thetao']:
                        Dataset['time'] = xr.date_range(start='1850-01-01', end='2349-03-01', periods=None, freq='M', tz=None, normalize=True, name=None, calendar='gregorian', use_cftime=True)
                    else:
                        Dataset['time'] = xr.date_range(start='1850-01-01', end='2351-01-01', periods=None, freq='M', tz=None, normalize=True, name=None, calendar='gregorian', use_cftime=True)

                    

                elif datasetN in ['hist']: #j==1
                    ######### HIST ########
                    
                    ### Apertura dataset
                    Dataset = xr.open_mfdataset(globals()[f'DS_{datasetN}'])

                    

                elif datasetN in ['obs']: # or :j==0
                    ######### OBS #########

                    ### Apertura dataset

                    Dataset = xr.open_dataset(globals()[f'DS_{datasetN}'])
                    
                    # Riassegnazione nome campo 
                    Dataset = Dataset.rename({'sst': 'tos'})


                ### Regridding
                if metodo_selezionato_regrid == 1:
                    Dir_hist_tos = '/nas/BOTTINO/CMIP6/LongRunMIP/EC-Earth-Consortium/EC-Earth3/historical/r4i1p1f1/Omon/tos_rg_1850-2014.nc'
                    Dataset_hist_tos = xr.open_dataset(Dir_hist_tos) # Seve se si vuole fare il regridding
                    Dataset = ctl.regrid_dataset(Dataset, regrid_to_reference=Dataset_hist_tos)

                ### Recupero dell' Indice Nino 3.4

                filename = 'Task-5_Dataset-{}.p'.format(datasetN)
                nino_index[datasetN] = pickle.load(open(dir_output_task5 + filename, 'rb'))[0]

                ### Discriminazione del range temporale:     
                max_year_dtgr = Dataset['time'].max()
                min_year_dtgr = Dataset['time'].min()
                # Estremi solo per Hist e Obs
                sup_year = str(int(max_year_dtgr.dt.year))+'-01-01'
                inf_year = str(int(min_year_dtgr.dt.year))+'-01-01'
                # Estremi per i Bottini
                sup_year_bott = cftime.DatetimeProlepticGregorian(int(max_year_dtgr.dt.year), 1, 1)
                inf_year_bott = cftime.DatetimeProlepticGregorian(int(max_year_dtgr.dt.year-time_window), 1, 1) # str(int(max_year_dtgr.dt.year)-time_window)+'-01-01'
                # sup_year_bott = cftime.DatetimeProlepticGregorian(int(max_year_dtgr.dt.year), 1, 1)
                # inf_year_bott = cftime.DatetimeProlepticGregorian(int(max_year_dtgr.dt.year)-time_window, 1, 1)
                date_format = '%Y-%m-%d'

                
                ### Limitazione del range temporale

                if datasetN not in ['hist', 'obs']:
                    Dataset = Dataset.sel(time=slice(inf_year_bott, sup_year_bott))
                    # Seleziono il periodo desiderato dell'indice
                    nino_index[datasetN] = nino_index[datasetN].sel(time=slice(inf_year_bott, sup_year_bott)).compute()
                else:
                    Dataset = Dataset.sel(time=slice(inf_year, sup_year))
                    # Seleziono il periodo desiderato dell'indice
                    nino_index[datasetN] = nino_index[datasetN].sel(time=slice(inf_year, sup_year)).compute()
                    # Arrotondo all'orario più vicino (necessario per Obs)
                    nino_index[datasetN]['time'] = nino_index[datasetN]['time'].dt.round('H')

                # Indici per la Composite
                nino_index_nino[datasetN] = nino_index[datasetN].where(nino_index[datasetN].rolling(time=6, min_periods=1, center=True).mean()>=0.4).dropna("time")
                nino_index_nina[datasetN] = nino_index[datasetN].where(nino_index[datasetN].rolling(time=6, min_periods=1, center=True).mean()<=-0.4).dropna("time")
        
                print("End of Dataset elaboration")
                time_end_a[j] = datetime.now()
                print("Elaboration of Dataset_{} completed in {}".format(datasetN, time_end_a[j]-time_begin_a[j]))

                ######################################################### Fine punto 2(a) #########################################################
                ###################################################################################################################################
            
                ###################################################################################################################################
                ################ (b) Discrimination based on the Dictionary' keys (Dataset's variables) and Recgression calculation ###############

                print("\nStart of time and space variable selection")
                time_begin_b[j] = datetime.now()

                if var == 'tos':
                    # Isola le singole variabili
                    campo = Dataset['tos'].sel(lat=slice(-85, 85)).compute() #lat=slice(-5, 5), lon=slice(190, 240)
                    # Maschera continentale
                    if flag_continental == True:
                        continental_mask = np.isnan(campo)[0,...]
                        flag_continental = False
                    if datasetN == 'obs':
                        campo['time'] = campo['time'].dt.round('H') # Arrotondo all'orario più vicino (necessario per Obs)


                elif var == 'thetao':
                    # Limito il range temporale di Thetao agli ultimi anni togliendo window_thetao tranne che per Hist
                    if datasetN in ['hist']:
                        campo = Dataset['thetao'].sel(time=slice(inf_year, sup_year), lat=slice(-5, 5), lon=slice(130, 280)).compute() # 130E - 80W
                        nino_index_nino[datasetN] = nino_index_nino[datasetN].sel(time=slice(inf_year, sup_year)).compute()
                        nino_index_nina[datasetN] = nino_index_nina[datasetN].sel(time=slice(inf_year, sup_year)).compute()
                    else:
                        campo = Dataset['thetao'].sel(time=slice(str(int(max_year_dtgr.dt.year) - window_thetao)+'-01-01', sup_year), lat=slice(-5, 5), lon=slice(130, 280)).compute() # 130E - 80W
                        nino_index_nino[datasetN] = nino_index_nino[datasetN].sel(time=slice(cftime.DatetimeProlepticGregorian((sup_year_bott.year - window_thetao), 1, 1), sup_year_bott)).compute() # .sel(time=slice(str(int(max_year_dtgr.dt.year)-time_window+window_thetao)+'-01-01', sup_year)).compute()
                        nino_index_nina[datasetN] = nino_index_nina[datasetN].sel(time=slice(cftime.DatetimeProlepticGregorian((sup_year_bott.year - window_thetao), 1, 1), sup_year_bott)).compute()
                    
                    # Detrending del campo
                    campo_detrended_decycled, anomaly_detrended_decycled, trend_hist, seasonal_cycle = apf.detrending(campo, time_rm//2)

                    campo_detrended_decycled = campo_detrended_decycled.mean('time').mean('lat') # Condensiamo al transetto longitudinale
                    
                    # Intersezione tra le coordinate temporali discontinue dell'Indice
                    common_times_nino = np.intersect1d(nino_index_nino[datasetN]["time"], anomaly_detrended_decycled["time"])
                    common_times_nina = np.intersect1d(nino_index_nina[datasetN]["time"], anomaly_detrended_decycled["time"])
                    # Selezione dal campo solo i punti temporali  condivisi con l'indice e riduzione dimensionale
                    anomaly_detrended_decycled_nino = anomaly_detrended_decycled.sel(time=common_times_nino).mean('lat').mean('time')
                    anomaly_detrended_decycled_nina = anomaly_detrended_decycled.sel(time=common_times_nina).mean('lat').mean('time')

                    # Calcolo delle Composite
                    for phase in oscillation:
                        if phase == 'nino':
                            composite[phase][var][datasetN] = anomaly_detrended_decycled_nino#.copy(deep=True)
                        else:
                            composite[phase][var][datasetN] = anomaly_detrended_decycled_nina#.copy(deep=True)
                    ###################################################################################################################################
                    #################################################### (c) Variables saving #########################################################
                    print("\nStart saving the evaluated variables: ", var)

                    with open(common_Dir_Output7 + 'Variables/' + f'Task-7-{datasetN}.pkl', 'ab') as file:
                        pickle.dump([campo_detrended_decycled,
                                    composite['nino'][var][datasetN],
                                    composite['nina'][var][datasetN]],
                                    file)
                        file.close()
                    print("End saving the evaluated variables")
                    continue # In questo modo non andrà a salvare le variaibili (inesistenti) in fondo a questo 'if'

                elif var == 'tas':
                    campo = Dataset['tas'].sel(lat=slice(-85, 85)).compute()
                    
                elif var == 'ua':
                    ua850 = Dataset['ua'].sel(lat=slice(-85, 85), plev=85000).compute()
                    ua200 = Dataset['ua'].sel(lat=slice(-85, 85), plev=20000).compute()
                    ua = [ua850, ua200]
                    ua_names = ['ua850', 'ua200']
                    ## Calcolo della Regressione
                    for k in range(len(ua)):
                        # Detrending del campo
                        campo_detrended_decycled, anomaly_detrended_decycled, trend_hist, seasonal_cycle = apf.detrending(ua[k], time_rm)
                        # Regressione
                        print("\nRegression of: ", ua_names[k])
                        (regression_product_dict['trend'][var][datasetN], 
                            regression_product_dict['intercept'][var][datasetN], 
                            regression_product_dict['trend_err'][var][datasetN], 
                            regression_product_dict['intercept_err'][var][datasetN], 
                            regression_product_dict['pval'][var][datasetN]) = ctl.calc_trend_climatevar(nino_index[datasetN].compute(), anomaly_detrended_decycled.compute(), var_units = None)
                        # Calcolo delle Composite
                        for phase in oscillation:
                            print(f"Processing phase: {phase}, variable: {var}")
                            if phase == 'nino':
                                composite[phase][var][datasetN] = (xr.where(nino_index[datasetN]>=0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                            elif phase == 'nina':
                                composite[phase][var][datasetN] = (xr.where(nino_index[datasetN]<=-0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                        
                        print("\nStart saving the evaluated variables: ", ua_names[k])
                        with open(common_Dir_Output6 + 'Variables/' + 'Task-6_' + ua_names[k] + f'-Products-{datasetN}.pkl', 'ab') as file:
                            pickle.dump([regression_product_dict['trend'][var][datasetN], 
                                        regression_product_dict['intercept'][var][datasetN], 
                                        regression_product_dict['trend_err'][var][datasetN], 
                                        regression_product_dict['intercept_err'][var][datasetN], 
                                        regression_product_dict['pval'][var][datasetN]],
                                        file)
                            file.close()
                        with open(common_Dir_Output8 + 'Variables/' + 'Task-8_' + ua_names[k] + f'-Composite-{datasetN}.pkl', 'ab') as file:
                            pickle.dump([composite['nino'][var][datasetN], 
                                        composite['nina'][var][datasetN]], 
                                        file)
                            file.close()
                        print("End saving the evaluated variables")

                    print("\nEnd of time and space variable selection and Regression evaluation")
                    time_end_b[j] = datetime.now()
                    print("Time and space ", var, " (ua850 + ua200) selection and Regression for Dataset_{} completed in {}".format(datasetN, time_end_b[j]-time_begin_b[j]))
                    continue


                elif var == 'pr':
                    campo = Dataset['pr'].sel(lat=slice(-85, 85)).compute()

                elif var == 'zg':
                    campo = Dataset['zg'].sel(lat=slice(-85, 85), plev=50000).compute()
                #--------------------------------------------------------------------------------------------------------------------------------------#
        
                ### Detrending
                campo_detrended_decycled, anomaly_detrended_decycled, trend_hist, seasonal_cycle = apf.detrending(campo, time_rm)

                if var == 'zg':
                    # Seleziono la stagione: 'DJF'
                    anomaly_detrended_decycled = anomaly_detrended_decycled.sel(time=anomaly_detrended_decycled.time.dt.month.isin([12, 1, 2])).groupby('time.year').mean('time') # Si rimuove il Ciclo stagionale (già rimosso)
                    # Resampling annuale dell'indice e calcolo della media
                    nino_index[datasetN] = nino_index[datasetN].resample(time="Y").mean()

                ### Calcolo delle Composite
                for phase in oscillation:
                    print(f"Processing phase: {phase}, variable: {var}")
                    if phase == 'nino':
                        composite[phase][var][datasetN] = (xr.where(nino_index[datasetN]>=0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                    elif phase == 'nina':
                        composite[phase][var][datasetN] = (xr.where(nino_index[datasetN]<=-0.4, anomaly_detrended_decycled, np.nan)).mean('time')
                
                ### Calcolo della Regressione
                print("\nRegression of : ", var)
                (regression_product_dict['trend'][var][datasetN], 
                    regression_product_dict['intercept'][var][datasetN], 
                    regression_product_dict['trend_err'][var][datasetN], 
                    regression_product_dict['intercept_err'][var][datasetN], 
                    regression_product_dict['pval'][var][datasetN]) = ctl.calc_trend_climatevar(nino_index[datasetN].compute(), anomaly_detrended_decycled.compute(), var_units = None)
                
                print("\nEnd of time and space variable selection and Regression evaluation")
                time_end_b[j] = datetime.now()
                print("Time and space ", var, " selection and Regression for Dataset_{} completed in {}".format(datasetN, time_end_b[j]-time_begin_b[j]))
                
                ######################################################### Fine punto 2(b) #########################################################
                ###################################################################################################################################
                
                ###################################################################################################################################
                #################################################### (c) Variables saving #########################################################


                print("\nStart saving the evaluated variables", var)
                with open(common_Dir_Output6 + 'Variables/' + f'Task-6_{var}-Products-{datasetN}.pkl', 'ab') as file:
                    pickle.dump([regression_product_dict['trend'][var][datasetN], 
                                regression_product_dict['intercept'][var][datasetN], 
                                regression_product_dict['trend_err'][var][datasetN], 
                                regression_product_dict['intercept_err'][var][datasetN], 
                                regression_product_dict['pval'][var][datasetN]],
                                file)
                    file.close()
                with open(common_Dir_Output8 + 'Variables/' + f'Task-8_{var}-Composite-{datasetN}.pkl', 'ab') as file:
                    pickle.dump([composite['nino'][var][datasetN], 
                                composite['nina'][var][datasetN]], 
                                file)
                    file.close()

                print("End saving the evaluated variables")

                ######################################################### Fine punto 2(c) #########################################################
                ###################################################################################################################################
                
                print("\nEvaluation of ", var, f"for Dataset_{datasetN}", " took {}".format(time_end_b[j]-time_begin_a[j]))
            time_var[i] = (time_end_b[dataset_names_iteration-1]-time_begin_a[0])
            print("\nOverall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))

        ######################################################## Fine del Main Loop ###############################################################

        ### Recap dei tempi impiegati
        print("\n## Timing recap")
        for i, var in enumerate(var_names):
            print("\nOverall evaluation of ", var, " for all the Datasets took {}".format(time_var[i]))
        
        print("\nOverall evaluation of the script was completed in {}".format(np.sum(time_var)))
         
        bool = False # Per il while

    else:
        raise ValueError('Given input is not allowed')

print("\nEnd of the script")
sys.exit()