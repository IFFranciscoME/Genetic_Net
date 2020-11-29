
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Trading System with Genetic Programming for Feature Engineering, Multilayer Perceptron     -- #
# -- -------  Neural Network Predictive Model and Genetic Algorithms for Hyperparameter Optimization     -- #
# -- file: data.py : input and output data functions for the project                                     -- #
# -- author: IFFranciscoME - franciscome@iteso.mx                                                        -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/Genetic_Net                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
from os import listdir, path
from os.path import isfile, join


# -------------------------------------------------------------------- Historical Minute Prices Grouping -- #
# -------------------------------------------------------------------- --------------------------------- -- #

def group_daily():
    main_path_g = 'files/daily_prices/'
    abspath = path.abspath(main_path_g)
    p_years_list = ['2007', '2008', '2009']
    r_data = {}
    files = sorted([f for f in listdir(abspath) if isfile(join(abspath, f))])
    column_names = ["timestamp", "open", "high", "low", "close", "volume"]

    for file in files:
        data = pd.read_csv(main_path_g + file,
                           names=column_names, parse_dates=["timestamp"], index_col=["timestamp"])

        data.columns = [i.lower() for i in list(data.columns)]
        data = data.resample("T").agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min',
                                         'volume': 'sum'})
        data = data.dropna()

        years = set([str(datadate.year) for datadate in list(data.index)])
        [years.discard(i) for i in p_years_list]

        years = sorted(list(years))

        for year in years:
            data_temp = data.groupby(pd.Grouper(freq='1Y')).get_group(year + '-12-31')
            # data_temp.to_csv('Archivos/' + 'MP_MIN_' + year + '.csv')
            r_data['MP_D_' + year] = data_temp

    return r_data


# ---------------------------------------------------------------------------- Historical Prices Reading -- #
# ---------------------------------------------------------------------------- ------------------------- -- #

# path in order to read files
main_path = 'files/daily_prices/'
abspath_f = path.abspath(main_path)
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

# iterative data reading
for file_f in files_f:
    data_f = pd.read_csv(main_path + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])
    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data['MP_D_' + year_f] = data_f

# whole data sets integrated
ohlc_data = pd.concat([price_data[list(price_data.keys())[5]], price_data[list(price_data.keys())[6]],
                       price_data[list(price_data.keys())[7]], price_data[list(price_data.keys())[8]],
                       price_data[list(price_data.keys())[9]], price_data[list(price_data.keys())[10]]])

# reset index
ohlc_data.reset_index(inplace=True, drop=True)

# ----------------------------------------------------------------------- Hyperparameters for the Models -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# data dictionary for models and their respective hyperparameter value candidates
models = {
    'logistic-elasticnet': {
        'label': 'logistic-elasticnet',
        'params': {'ratio': [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                   'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5]}},
    'ls-svm': {
        'label': 'ls-svm',
        'params': {'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5],
                   'kernel': ['linear', 'linear', 'linear', 'linear', 'linear',
                              'rbf', 'rbf', 'rbf', 'rbf', 'rbf'],
                   'gamma': ['scale', 'scale', 'scale', 'scale', 'scale',
                             'auto', 'auto', 'auto', 'auto', 'auto']}},
    'ann-mlp': {
        'label': 'ann-mlp',
        'params': {'hidden_layers': [(10, ), (20, ), (5, 5), (20, 20), (50, ),
                                     (10, ), (10, ), (5, 5), (10, 10), (20, )],
                   'activation': ['relu', 'relu', 'relu', 'relu', 'relu',
                                  'logistic', 'logistic', 'logistic', 'logistic', 'logistic'],
                   'alpha': [0.2, 0.1, 0.01, 0.001, 0.0001, 0.2, 0.1, 0.01, 0.001, 0.0001],
                   'learning_r': ['constant', 'constant', 'constant', 'constant', 'constant',
                                  'adaptive', 'adaptive', 'adaptive', 'adaptive', 'adaptive'],
                   'learning_r_init': [0.2, 0.1, 0.01, 0.001, 0.0001,
                                       0.2, 0.1, 0.01, 0.001, 0.0001]}}}

# ------------------------------------------------------------------------------------- Themes for plots -- #
# ------------------------------------------------------------------------------------- ---------------- -- #

# Plot_1 : Original Historical OHLC prices
theme_plot_1 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 12, 'font_axis': 12, 'font_ticks': 12},
                    p_dims={'width': 800, 'height': 400},
                    p_labels={'title': 'Precios OHLC',
                              'x_title': 'Fechas', 'y_title': 'Futuros USD/MXN'})

# Plot_2 : Timeseries T-Folds blocks without filtration
theme_plot_2 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 12, 'font_axis': 12, 'font_ticks': 12},
                    p_dims={'width': 800, 'height': 400},
                    p_labels={'title': 'T-Folds por Bloques Sin Filtraciones',
                              'x_title': 'Fechas', 'y_title': 'Futuros USD/MXN'})

# Plot_3 Observed Class vs Predicted Class
theme_plot_3 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 12, 'font_axis': 12, 'font_ticks': 12},
                    p_dims={'width': 800, 'height': 400},
                    p_labels={'title': 'Clasificaciones',
                              'x_title': 'Fechas', 'y_title': 'Clasificacion'})

# Plot_4 ROC of models
theme_plot_4 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 12, 'font_axis': 12, 'font_ticks': 12},
                    p_dims={'width': 800, 'height': 400},
                    p_labels={'title': 'ROC (Test Data)',
                              'x_title': 'FPR', 'y_title': 'TPR'})

# Plot_5 AUC Timeseries of models
theme_plot_5 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 12, 'font_axis': 12, 'font_ticks': 12},
                    p_dims={'width': 800, 'height': 400},
                    p_labels={'title': 'AUC por periodo (Test Data)',
                              'x_title': 'Periodos', 'y_title': 'AUC'})
