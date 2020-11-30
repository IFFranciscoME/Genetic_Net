
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Trading System with Genetic Programming for Feature Engineering, Multilayer Perceptron     -- #
# -- -------  Neural Network Predictive Model and Genetic Algorithms for Hyperparameter Optimization     -- #
# -- file: main.py : main functions structure                                                            -- #
# -- author: IFFranciscoME - franciscome@iteso.mx                                                        -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/Genetic_Net                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import visualizations as vs
import data as dt
import functions as fn
import warnings
import pickle
warnings.filterwarnings("ignore")

# ------------------------------------------------------------- PLOT 1: MXN/USD Historical Future Prices -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# plot 1 : time series candlesticks OHLC historical prices
plot_1 = vs.g_ohlc(p_ohlc=dt.ohlc_data, p_theme=dt.theme_plot_1, p_vlines=None)

# show plot in explorer
# plot_1.show()

# generate the plot online with chart studio
# py.plot(plot_1)

# ------------------------------------------------------------------- TABLE 1: Exploratory Data Analysis -- #
# ------------------------------------------------------------------- ---------------------------------- -- #

# data description
table_1 = dt.ohlc_data.describe()

# --------------------------------------------------------- Timeseries T-Folds Blocks Without Filtration -- #
# --------------------------------------------------------- -------------------------------------------- -- #

# in quarters obtain 4 folds for each year
t_folds = fn.t_folds(p_data=dt.ohlc_data, p_period='quarter')
# drop the last quarter because it is incomplete until december 31
t_folds.pop('q_04_2020', None)

# -- ----------------------------------------------------------------- PLOT 2: Time Series Block T-Folds -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# construccion de fechas para lineas verticales de division de cada fold
dates_folds = []
for fold in t_folds:
    dates_folds.append(t_folds[fold]['timestamp'].iloc[0])
    dates_folds.append(t_folds[fold]['timestamp'].iloc[-1])

# grafica OHLC
plot_2 = vs.g_ohlc(p_ohlc=dt.ohlc_data,
                   p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# mostrar grafica
# plot_2.show()

# generar grafica en chart studio
# py.plot(plot_2)

# ----------------------------------------------------------------- Features - Train/Optimization - Test -- #
# ----------------------------------------------------------------- ------------------------------------ -- #

# -- --------------------------------------------------------------- Run Process (WARNING - Takes Hours) -- #
# ml_models = list(dt.models.keys())
# global_cases = fn.global_evaluations(p_data_folds=t_folds, p_models=ml_models,
#                                      p_saving=True, p_file_name='Genetic_Net_Data_Folds.dat')

# -- ------------------------------------------------------------------------- Load Data for offline use -- #
global_cases = dt.data_save_load(p_data_objects=None, p_data_action='load',
                                 p_data_file='Genetic_Net_Quarters.dat')

# -- -------------------------------------------------------------------- RESULTS: AUC Min and Max cases -- #
# -- --------------------------------------------------------------------------- ----------------------- -- #
# -- Funcion de casos representativos

# diccionario para almacenar resultados de busqueda
auc_cases = {j: {i: {'data': {}}
                 for i in ['auc_min', 'auc_max', 'hof_metrics']} for j in list(dt.models.keys())}

# ciclo para busqueda de auc_min y auc_max
for model in list(dt.models.keys()):
    auc_min = 1
    auc_max = 0
    auc_max_params = {}
    auc_min_params = {}
    for period in t_folds:
        auc_cases[model]['hof_metrics']['data'][period] = {}
        auc_s = []
        for i in range(0, 10):
            auc_s.append(global_cases[model][period]['e_hof'][i]['metrics']['test']['auc'])

            # -- caso 1
            # El individuo de todos los HOF de todos los periodos que produjo la minima AUC
            if global_cases[model][period]['e_hof'][i]['metrics']['test']['auc'] < auc_min:
                auc_min = global_cases[model][period]['e_hof'][i]['metrics']['test']['auc']
                auc_cases[model]['auc_min']['data'] = global_cases[model][period]['e_hof'][i]
                auc_min_params = global_cases[model][period]['e_hof'][i]['params']

            # -- caso 2
            # El individuo de todos los HOF de todos los periodos que produjo la maxima AUC
            elif global_cases[model][period]['e_hof'][i]['metrics']['test']['auc'] > auc_max:
                auc_max = global_cases[model][period]['e_hof'][i]['metrics']['test']['auc']
                auc_cases[model]['auc_max']['data'] = global_cases[model][period]['e_hof'][i]
                auc_max_params = global_cases[model][period]['e_hof'][i]['params']

        # Guardar info por periodo
        auc_cases[model]['hof_metrics']['data'][period]['auc_s'] = auc_s
        auc_cases[model]['hof_metrics']['data'][period]['auc_max'] = auc_max
        auc_cases[model]['hof_metrics']['data'][period]['auc_max_params'] = auc_max_params
        auc_cases[model]['hof_metrics']['data'][period]['auc_min'] = auc_min
        auc_cases[model]['hof_metrics']['data'][period]['auc_min_params'] = auc_min_params
