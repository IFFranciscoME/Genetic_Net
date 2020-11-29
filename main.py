
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


# --------------------------------------------------------------- PLOT 1: PRECIOS OHLC DE FUTURO USD/MXN -- #
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

# ----------------------------------------------------------------- Features - Train/Optimization - Test -- #
# ----------------------------------------------------------------- ------------------------------------ -- #

# main data structure for calculations
memory_palace = {j: {i: {'pop': [], 'logs': [], 'hof': [], 'e_hof': []}
                     for i in t_folds} for j in list(dt.models.keys())}

for model in list(dt.models.keys()):
    for period in t_folds:

        print('\n')
        print('----------------------------')
        print('modelo: ', model)
        print('periodo: ', period)
        print('----------------------------')
        print('\n')
        print('----------------------- Ingenieria de Variables por Periodo ------------------------')
        print('----------------------- ----------------------------------- ------------------------')

        # generacion de features
        m_features = fn.genetic_programed_features(p_data=t_folds[period], p_memory=7)

        # resultados de optimizacion
        print('\n')
        print('--------------------- Optimizacion de hiperparametros por Periodo ------------------')
        print('--------------------- ------------------------------------------- ------------------')

        hof_model = fn.genetic_algo_optimisation(p_data=m_features, p_model=dt.models[model])
        # -- evaluacion de modelo para cada modelo y cada periodo de todos los Hall of Fame
        for i in range(0, len(list(hof_model['hof']))):
            # evaluar modelo
            hof_eval = fn.model_evaluations(p_features=m_features,
                                            p_model=dt.models[model],
                                            p_optim_data=hof_model['hof'])
            # guardar evaluaciones de todos los individuos del Hall of Fame
            memory_palace[model][period]['e_hof'].append(hof_eval)


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
            auc_s.append(memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'])

            # -- caso 1
            # El individuo de todos los HOF de todos los periodos que produjo la minima AUC
            if memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'] < auc_min:
                auc_min = memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc']
                auc_cases[model]['auc_min']['data'] = memory_palace[model][period]['e_hof'][i]
                auc_min_params = memory_palace[model][period]['e_hof'][i]['params']

            # -- caso 2
            # El individuo de todos los HOF de todos los periodos que produjo la maxima AUC
            elif memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'] > auc_max:
                auc_max = memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc']
                auc_cases[model]['auc_max']['data'] = memory_palace[model][period]['e_hof'][i]
                auc_max_params = memory_palace[model][period]['e_hof'][i]['params']

        # Guardar info por periodo
        auc_cases[model]['hof_metrics']['data'][period]['auc_s'] = auc_s
        auc_cases[model]['hof_metrics']['data'][period]['auc_max'] = auc_max
        auc_cases[model]['hof_metrics']['data'][period]['auc_max_params'] = auc_max_params
        auc_cases[model]['hof_metrics']['data'][period]['auc_min'] = auc_min
        auc_cases[model]['hof_metrics']['data'][period]['auc_min_params'] = auc_min_params
