
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
from datetime import datetime
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# General dataframe with all the data for the project
general_data = dt.ohlc_data.copy()

# ------------------------------------------------------------- PLOT 1: MXN/USD Historical Future Prices -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# Override plot main title
dt.theme_plot_1['p_labels']['title'] = 'Gráfica 1: <b> Precios Históricos OHLC </b>'

# plot 1 : time series candlesticks OHLC historical prices
plot_1 = vs.g_ohlc(p_ohlc=general_data, p_theme=dt.theme_plot_1, p_vlines=None)

# show plot in explorer
# plot_1.show()

# generate the plot online with chart studio
# py.plot(plot_1)

# ------------------------------------------------------------------- TABLE 1: Exploratory Data Analysis -- #
# ------------------------------------------------------------------- ---------------------------------- -- #

# data description
table_1 = general_data.describe()

# --------------------------------------------------------- Timeseries T-Folds Blocks Without Filtration -- #
# --------------------------------------------------------- -------------------------------------------- -- #

# # in quarters obtain 4 folds for each year
t_folds = fn.t_folds(p_data=general_data.copy(), p_period='Quarter')
# drop the last quarter because it is incomplete until december 31
t_folds.pop('q_04_2020', None)

# -- ----------------------------------------------------------------- PLOT 2: Time Series Block T-Folds -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# construccion de fechas para lineas verticales de division de cada fold
dates_folds = []
for fold in list(t_folds.keys()):
    dates_folds.append(t_folds[fold]['timestamp'].iloc[0])
    dates_folds.append(t_folds[fold]['timestamp'].iloc[-1])

# grafica OHLC
plot_2 = vs.g_ohlc(p_ohlc=general_data,
                   p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# mostrar grafica
# plot_2.show()

# generar grafica en chart studio
# py.plot(plot_2)

# ----------------------------------------------------------------- Features - Train/Optimization - Test -- #
# ----------------------------------------------------------------- ------------------------------------ -- #

# initialization of code running
# print(datetime.now())

# list with the names of the models
ml_models = list(dt.models.keys())

# -- --------------------------------------------------------------- Run Process (WARNING - Takes Hours) -- #
# global_evaluations = fn.folds_evaluations(p_data_folds=t_folds, p_models=ml_models,
#                                           p_saving=False, p_file_name='Genetic_Net_Year.dat')
#
# dt.data_save_load(p_data_objects=global_evaluations,
#                   p_data_action='save', p_data_file='Genetic_Net_Year.dat')
#
# # ending of code running
# print(datetime.now())

# -- ------------------------------------------------------------------------- Load Data for offline use -- #
memory_palace = dt.data_save_load(p_data_objects=None, p_data_action='load',
                                  p_data_file='files/pickle_rick/Genetic_Net_Quarter.dat')

# -- ----------------------------------------------------------------------------- AUC Min and Max cases -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# min and max AUC cases for the models
auc_cases = fn.models_auc(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=t_folds)

# -- -------------------------------------------------------------------------- Model Global Performance -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# # model performance for all models, with the min and max AUC parameters
# global_evaluations = fn.model_evaluation(p_data=general_data, p_memory=7, p_global_cases=memory_palace,
#                                          p_models=ml_models, p_cases=auc_cases)

# -- ------------------------------------------------------------- PLOT 3: Classification Global Results -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# pick case
case = 'max'

# pick model to generate the plot
auc_model = 'ann-mlp'

# generate title
auc_title = 'max AUC for: ' + auc_model + ' found in period: ' + auc_cases[auc_model]['auc_max']['period']

# get data from auc_cases
train_y = auc_cases[auc_model]['auc' + '_' + case]['data']['results']['data']['train']
test_y = auc_cases[auc_model]['auc' + '_' + case]['data']['results']['data']['test']

# get data for prices and predictions
ohlc_prices = t_folds[auc_cases[auc_model]['auc' + '_' + case]['period']]
ohlc_class = {'train_y': train_y['y_train'], 'train_y_pred': train_y['y_train_pred'],
              'test_y': test_y['y_test'], 'test_y_pred': test_y['y_test_pred']}

# make plot
plot_3 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class,
                         p_vlines=None)

# visualize plot
plot_3.show()

# -- ------------------------------------------------------------- PLOT 4: Classification Global Results -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# Timeseries of the AUCs
plot_4_folds = vs.g_roc_auc(p_cases=auc_cases, p_type='test', p_models=ml_models, p_theme=dt.theme_plot_4)

# offline plot
# plot_4_folds.show()

# online plot

# -- ----------------------------------------------------------------- PLOT 5: Timeseries AUC for Models -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

minmax_auc_test = {i: {'x_period': [], 'y_mins': [], 'y_maxs': []} for i in ml_models}

# get the cases where auc was min and max in all the periods
for model in ml_models:
    minmax_auc_test[model]['x_period'] = list(auc_cases[model]['hof_metrics']['data'].keys())
    minmax_auc_test[model]['y_mins'] = [auc_cases[model]['hof_metrics']['data'][periodo]['auc_min']
                                        for periodo in list(auc_cases[model]['hof_metrics']['data'].keys())]
    minmax_auc_test[model]['y_maxs'] = [auc_cases[model]['hof_metrics']['data'][periodo]['auc_max']
                                        for periodo in list(auc_cases[model]['hof_metrics']['data'].keys())]

# produce plot
plot_5 = vs.g_timeseries_auc(p_data_auc=minmax_auc_test, p_theme=dt.theme_plot_5)

# offline plot
# plot_5.show()

# online plot

# -- ----------------------------------------------- Table 2: Model Parameter for AUC min and max values -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #
# -- for every model, for every period, for AUC Min and AUC Max cases

# stable data
data_stables = {model: {'df_auc_max': {period: {} for period in t_folds},
                        'df_auc_min': {period: {} for period in t_folds}} for model in ml_models}

# periods
period_max_auc = {model: {period: {} for period in t_folds} for model in ml_models}
period_min_auc = {model: {period: {} for period in t_folds} for model in ml_models}

# cycle for getting the parameters of every model
for model in ml_models:
    for period in list(t_folds.keys()):
        period_max_auc[model][period] = auc_cases[model]['hof_metrics']['data'][period]['auc_max_params']
        period_min_auc[model][period] = auc_cases[model]['hof_metrics']['data'][period]['auc_min_params']

# Table 2: Model parameters
table_2 = {'model_1': {'max': pd.DataFrame(period_max_auc['logistic-elasticnet']).T,
                       'min': pd.DataFrame(period_min_auc['logistic-elasticnet']).T},
           'model_2': {'max': pd.DataFrame(period_max_auc['ls-svm']).T,
                       'min': pd.DataFrame(period_min_auc['ls-svm']).T},
           'model_3': {'max': pd.DataFrame(period_max_auc['ann-mlp']).T,
                       'min': pd.DataFrame(period_min_auc['ann-mlp']).T}}

# evolution of parameters for every  model
t_model_1 = table_2['model_1']['max']
t_model_2 = table_2['model_2']['max']
t_model_3 = table_2['model_3']['max']
t_model_3['hidden_layers'] = [str(i) for i in list(t_model_3['hidden_layers'])]


# -- ------------------------------------------------------------------------------- Complexity Analysis -- #
# -- ------------------------------------------------------------------------------- ------------------- -- #

# object for time keeping
times = {'Quarter': [], 'Semester': [], 'Year': []}

for size in times.keys():
    # load data
    memory_palace = dt.data_save_load(p_data_objects=None, p_data_action='load',
                                      p_data_file='files/pickle_rick/Genetic_Net_' + size + '.dat')

    # # in quarters obtain 4 folds for each year
    t_folds = fn.t_folds(p_data=general_data.copy(), p_period=size)
    # drop the last quarter because it is incomplete until december 31
    t_folds.pop('q_04_2020', None)
    # list with the names of the models
    ml_models = list(dt.models.keys())
    period_times = list()
    # iterate to have all the times for each period for each model
    for model in ml_models:
        for period in list(t_folds.keys()):
            period_times.append(memory_palace[model][period]['time'].seconds)

    times[size] = np.mean(period_times)
