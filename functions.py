
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Trading System with Genetic Programming for Feature Engineering, Multilayer Perceptron     -- #
# -- -------  Neural Network Predictive Model and Genetic Algorithms for Hyperparameter Optimization     -- #
# -- file: functions.py : Data processing and models                                                     -- #
# -- author: IFFranciscoME - franciscome@iteso.mx                                                        -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/Genetic_Net                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
import random
import data as dt

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from gplearn.genetic import SymbolicTransformer
from deap import base, creator, tools, algorithms

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------------------------------------------------------- Data Scaling/Transformation -- #
# -------------------------------------------------------------------------- --------------------------- -- #

def data_scaler(p_data, p_trans):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada

    Parameters
    ----------
    p_trans: str
        Standard: Para estandarizacion (restar media y dividir entre desviacion estandar)
        Robust: Para estandarizacion robusta (restar mediana y dividir entre rango intercuartilico)

    p_data: pd.DataFrame
        Con datos numericos de entrada

    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados

    """

    if p_trans == 'Standard':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        # armar objeto de salida
        p_data[list(p_data.columns[1:])] = StandardScaler().fit_transform(lista)

    elif p_trans == 'Robust':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        # armar objeto de salida
        p_data[list(p_data.columns[1:])] = RobustScaler().fit_transform(lista)

    elif p_trans == 'Scale':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        p_data[list(p_data.columns[1:])] = MaxAbsScaler().fit_transform(lista)

    return p_data


# --------------------------------------------------------------------------- Divide the data in T-Folds -- #
# --------------------------------------------------------------------------- ----------------------------- #

def t_folds(p_data, p_period):
    """
    Function to separate in T-Folds the data, considering not having filtrations (Month and Quarter)

    Parameters
    ----------
    p_data : pd.DataFrame
        DataFrame with data

    p_period : str
        'month': monthly data division
        'quarter' quarterly data division

    Returns
    -------
    m_data or q_data : 'period_'

    References
    ----------
    https://web.stanford.edu/~hastie/ElemStatLearn/

    """

    # data scaling by standarization
    # p_data.iloc[:, 1:] = data_scaler(p_data=p_data.copy(), p_trans='Standard')

    # For quarterly separation of the data
    if p_period == 'quarter':
        # List of quarters in the dataset
        quarters = list(set(time.quarter for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        q_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            q_data.update({'q_' + str('0') + str(i) + '_' + str(y) if i <= 9 else str(i) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      (pd.to_datetime(p_data['timestamp']).dt.quarter == i)]
                           for i in quarters})
        return q_data

    # For quarterly separation of the data
    elif p_period == 'semester':
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        s_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            # y = sorted(list(years))[0]
            s_data.update({'s_' + str('0') + str(1) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      ((pd.to_datetime(p_data['timestamp']).dt.quarter == 1) |
                                      (pd.to_datetime(p_data['timestamp']).dt.quarter == 2))]})

            s_data.update({'s_' + str('0') + str(2) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      ((pd.to_datetime(p_data['timestamp']).dt.quarter == 3) |
                                       (pd.to_datetime(p_data['timestamp']).dt.quarter == 4))]})

        return s_data

        # For quarterly separation of the data
    elif p_period == 'year':
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        y_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            # y = sorted(list(years))[0]
            y_data.update({'y_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y)]})
        return y_data

    # In the case a different label has been receieved
    return 'Error: verify parameters'


# ------------------------------------------------------------------------------ Autoregressive Features -- #
# --------------------------------------------------------------------------------------------------------- #

def autoregressive_features(p_data, p_nmax):
    """
    Generation of autoregressive features (lags, moving average, differences)

    Parameters
    ----------
    p_data: pd.DataFrame
        With columns OHLCV to build features

    p_nmax: int
        Memory parameter to consider in the calculations with the historical prices

    Returns
    -------
    r_features: pd.DataFrame
        With the calculated features

    """

    # multiplication factor
    pip_mult = 10000
    # make a copy of the data
    data = p_data.copy()
    # discounted pips in the close
    data['co'] = (data['close'] - data['open']) * pip_mult
    # discounted pips in uptrend
    data['ho'] = (data['high'] - data['open']) * pip_mult
    # discounted pips in downtrend
    data['ol'] = (data['open'] - data['low']) * pip_mult
    # discounted pips in volatility
    data['hl'] = (data['high'] - data['low']) * pip_mult
    # binary class to predict
    data['co_d'] = [1 if i > 0 else 0 for i in list(data['co'])]

    # iterations to calculate the N features
    for n in range(0, p_nmax):

        # Lag n with Open Interest
        data['lag_vol_' + str(n + 1)] = data['volume'].shift(n + 1)
        # Lag n with Open - Low
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
        # Lag n with High - Open
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
        # Lag n with High - Low
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)
        # moving average with volume with n window
        data['ma_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).mean()
        # moving average with open-low with n window
        data['ma_ol_' + str(n + 1)] = data['ol'].rolling(n + 1).mean()
        # moving average with high-open with n window
        data['ma_ho_' + str(n + 1)] = data['ho'].rolling(n + 1).mean()
        # moving average with high-low with n window
        data['ma_hl_' + str(n + 1)] = data['hl'].rolling(n + 1).mean()

    # timestamp as index
    data.index = pd.to_datetime(data.index)
    # drop columns
    r_features = data.drop(['open', 'high', 'low', 'close', 'hl', 'ol', 'ho', 'volume'], axis=1)
    r_features = r_features.dropna(axis='columns', how='all')
    r_features = r_features.dropna(axis='rows')
    # convert to float
    r_features.iloc[:, 2:] = r_features.iloc[:, 2:].astype(float)
    # binary column
    r_features['co_d'] = [0 if i <= 0 else 1 for i in r_features['co_d']]
    # reset index
    r_features.reset_index(inplace=True, drop=True)

    return r_features


# ------------------------------------------------------------------------------------ Hadamard Features -- #
# --------------------------------------------------------------------------------------------------------- #

def hadamard_features(p_data, p_nmax):
    """
    Hadamard product for feature variables generation

    Parameters
    ----------
    p_data: pd.DataFrame
        With columns OHLCV to build features

    p_nmax: int
        Memory parameter to consider in the calculations with the historical prices

    Returns
    -------
    r_features: pd.DataFrame
        With the calculated features

    """

    # sequential combination of variables
    for n in range(p_nmax):

        # previously generated features columns
        list_hadamard = ['lag_vol_' + str(n + 1),
                         'lag_ol_' + str(n + 1),
                         'lag_ho_' + str(n + 1),
                         'lag_hl_' + str(n + 1)]

        # Hadamard product with previous features
        for x in list_hadamard:
            p_data['h_' + x + '_' + 'ma_ol_' + str(n + 1)] = p_data[x] * p_data['ma_ol_' + str(n + 1)]
            p_data['h_' + x + '_' + 'ma_ho_' + str(n + 1)] = p_data[x] * p_data['ma_ho_' + str(n + 1)]
            p_data['h_' + x + '_' + 'ma_hl_' + str(n + 1)] = p_data[x] * p_data['ma_hl_' + str(n + 1)]

    return p_data


# ------------------------------------------------------------------------------------ Symbolic Features -- #
# --------------------------------------------------------------------------------------------------------- #

def symbolic_features(p_x, p_y):
    """
    Symbolic features generation with genetic programming

    Parameters
    ----------
    p_x: pd.DataFrame
        with regressors or predictor variables

    p_y: pd.DataFrame
        with variable to predict

    Returns
    -------
    score_gp: float
        error of prediction

    """

    # funcion de generacion de variables simbolicas
    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log', 'sqrt'],
                                population_size=10000, hall_of_fame=40, n_components=30,
                                generations=80, tournament_size=20,  stopping_criteria=.60,
                                const_range=None, init_method='half and half', init_depth=(4, 12),
                                metric='pearson', parsimony_coefficient=0.01,
                                p_crossover=0.4, p_subtree_mutation=0.3, p_hoist_mutation=0.1,
                                p_point_mutation=0.2, p_point_replace=.05,
                                verbose=1, random_state=None, n_jobs=7, feature_names=p_x.columns,
                                warm_start=True)

    # result of fit with the SymbolicTransformer function
    model_fit = model.fit_transform(p_x, p_y)

    # dataframe with parameters
    data = pd.DataFrame(model_fit)

    # parameters of the model
    model_params = model.get_params()

    # results
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data}

    return results


# -------------------------------------------------------------------------------- Feature Concatenation -- #
# -------------------------------------------------------------------------------- ------------------------ #

def genetic_programed_features(p_data, p_memory):
    """
    Autoregressive, Hadamard product and Genetic programming tools to generate timeseries endogenous
    features.

    Parameters
    ----------
    p_data: pd.DataFrame
        Data to generate features, OHLC

    p_memory: int
        Memory parameter to consider in the calculations with the historical prices

    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'test_x': pd.DataFrame, 'test_y': pd.DataFrame}

    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """

    # ------------------------------------------------- Feature Engineering for Autoregressive processes -- #
    # ------------------------------------------------- ------------------------------------------------ -- #

    # function to generate autoregressive features
    data_arf = autoregressive_features(p_data=p_data, p_nmax=p_memory)

    # dependent (target) variable separation from the data set in order to avoid filtration
    data_y = data_arf['co_d'].copy()

    # independent (explanatory) candidate variables separation
    data_arf = data_arf.drop(['timestamp', 'co', 'co_d'], axis=1, inplace=False)

    # --------------------------------------------------------- Feature Engineering for Hadamard Product -- #
    # --------------------------------------------------------- ---------------------------------------- -- #

    # function to generate hadamard product features
    data_had = hadamard_features(p_data=data_arf, p_nmax=p_memory)

    # -------------------------------------------------------- Feature Engineering for Symbolic Features -- #
    # -------------------------------------------------------- ----------------------------------------- -- #

    # Ejemplo de ecuacion:
    # ma_hl_5 * (lag_ol_5 + ma_ho_6) * (lag_ol_6 + 0.074 + ma_hl_3 * (lag_ol_4 - ma_ol_3) / lag_ho_4)

    # Symbolic features generation
    fun_sym = symbolic_features(p_x=data_had, p_y=data_y)

    # variables
    data_sym = fun_sym['data']
    data_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]

    # equations for the symbolic features
    equations = [i.__str__() for i in list(fun_sym['model'])]

    # concatenated data of the 3 types of features
    data_model = pd.concat([data_arf.copy(), data_had.copy(), data_sym.copy()], axis=1)
    model_data = {}

    # -- Data vision in train and test according to a proportion 70% Train and 30% test
    xtrain, xtest, ytrain, ytest = train_test_split(data_model, data_y, test_size=.3, shuffle=False)

    # Data division between explanatory variables (x) and target variable (y)
    model_data['train_x'] = xtrain
    model_data['train_y'] = ytrain
    model_data['test_x'] = xtest
    model_data['test_y'] = ytest
    model_data['features_eq'] = equations

    return model_data


# ---------------------------------------------------------------------------- Model Performance Metrics -- #
# --------------------------------------------------------------------------------------------------------- #

def model_metrics():

    return {}


# ------------------------------------------- MODEL: Logistic Regression with ELASTIC NET regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def logistic_net(p_data, p_params):
    """
    Funcion para ajustar varios modelos lineales

    Parameters
    ----------
    p_data: dict
        Diccionario con datos de entrada como los siguientes:

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]

    p_params: dict
        Diccionario con parametros de entrada para modelos, como los siguientes

        p_alpha: float
                alpha for the models
                p_alpha = 0.1

        p_ratio: float
            elastic net ratio between L1 and L2 regularization coefficients
            p_ratio = 0.1

        p_iterations: int
            Number of iterations until stop the model fit process
            p_iter = 200

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    References
    ----------
    ElasticNet
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    """

    # Datos de entrenamiento
    x_train = p_data['train_x']
    y_train = p_data['train_y']

    # Datos de prueba
    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # alpha, l1_ratio,

    # computations parameters
    # fit_intercept, normalize, precompute, copy_X, tol, warm_start, positive, selection

    # Fit model
    en_model = LogisticRegression(l1_ratio=p_params['ratio'], C=p_params['c'], tol=1e-3,
                                  penalty='elasticnet', solver='saga', multi_class='ovr', n_jobs=7,
                                  max_iter=5000, fit_intercept=False)

    # model fit
    en_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = en_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})

    # Confussion matrix
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = en_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = en_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = en_model.predict_proba(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': en_model, 'intercept': en_model.intercept_, 'coef': en_model.coef_,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

    return r_models


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #
@ignore_warnings(category=ConvergenceWarning)
def ls_svm(p_data, p_params):
    """
    Least Squares Support Vector Machines

    Parameters
    ----------
    p_data: dict
        Diccionario con datos de entrada como los siguientes:

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]

    p_params: dict
        Diccionario con parametros de entrada para modelos, como los siguientes

        p_kernel: str
                kernel de LS_SVM
                p_alpha = ['linear']

        p_c: float
            Valor de coeficiente C
            p_ratio = 0.1

        p_gamma: int
            Valor de coeficiente gamma
            p_iter = 0.1

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    References
    ----------
    https://scikit-learn.org/stable/modules/svm.html#

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # C, kernel, degree (if kernel = poly), gamma (if kernel = {rbf, poly, sigmoid},
    # coef0 (if kernel = {poly, sigmoid})

    # computations parameters
    # shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape,
    # break_ties, random_state

    # model function
    svm_model = SVC(C=p_params['c'], kernel=p_params['kernel'], gamma=p_params['gamma'],

                    shrinking=True, probability=True, tol=1e-5, cache_size=4000,
                    class_weight=None, verbose=False, max_iter=100000, decision_function_shape='ovr',
                    break_ties=False, random_state=None)

    # model fit
    svm_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = svm_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = svm_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = svm_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = svm_model.predict_proba(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': svm_model,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

    return r_models


# --------------------------------------------------- MODEL: Artificial Neural Net Multilayer Perceptron -- #
# --------------------------------------------------------------------------------------------------------- #

def ann_mlp(p_data, p_params):
    """
    Artificial Neural Network, particularly, a MultiLayer Perceptron for Supervised Classification

    Parameters
    ----------
    p_data: dict
        Diccionario con datos de entrada como los siguientes:

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]

    p_params: dict

        Diccionario con parametros de entrada para modelos, como los siguientes

        hidden_layers: ()
        activation: float
        alpha: int
        learning_r: int
        learning_r_init: int

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    References
    ----------
    https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # hidden_layer_sizes, activation, solver, alpha, learning_rate,

    # batch_size, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose,
    # warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction

    # model function
    mlp_model = MLPClassifier(hidden_layer_sizes=p_params['hidden_layers'],
                              activation=p_params['activation'], alpha=p_params['alpha'],
                              learning_rate=p_params['learning_r'],
                              learning_rate_init=p_params['learning_r_init'],

                              batch_size='auto', solver='sgd', power_t=0.5, max_iter=10000, shuffle=False,
                              random_state=None, tol=1e-7, verbose=False, warm_start=True, momentum=0.8,
                              nesterovs_momentum=True, early_stopping=False, validation_fraction=0.2,
                              n_iter_no_change=100)

    # model fit
    mlp_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = mlp_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = mlp_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds

    x_probs = np.isnan(probs_train)
    # replacing NaN values with 0
    probs_train[x_probs] = 0

    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = mlp_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = mlp_model.predict_proba(x_test)

    y_probs = np.isnan(probs_test)
    # replacing NaN values with 0
    probs_test[y_probs] = 0

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for test data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': mlp_model,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

    return r_models


# -------------------------------------------------------------------------- FUNCTION: Genetic Algorithm -- #
# ------------------------------------------------------- ------------------------------------------------- #

def genetic_algo_optimisation(p_data, p_model):
    """
    El uso de algoritmos geneticos para optimizacion de hiperparametros de varios modelos

    Parameters
    ----------
    p_model: dict
        'label' con etiqueta del modelo, 'params' llaves con parametros y listas de sus valores a optimizar

    p_data: pd.DataFrame
        data frame con datos del m_fold

    Returns
    -------
    r_model_ols_elasticnet: dict
        resultados de modelo OLS con regularizacion elastic net

    r_model_ls_svm: dict
        resultados de modelo Least Squares Support Vector Machine

    r_model_ann_mlp: dict
        resultados de modelo Red Neuronal Artificial tipo perceptron multicapa

    References
    ----------
    https://github.com/deap/deap

    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """

    # -- ------------------------------------------------------- OLS con regularizacion tipo Elastic Net -- #
    # ----------------------------------------------------------------------------------------------------- #
    if p_model['label'] == 'logistic-elasticnet':

        # borrar clases previas si existen
        try:
            del creator.FitnessMax_en
            del creator.Individual_en
        except AttributeError:
            pass

        # inicializar ga
        creator.create("FitnessMax_en", base.Fitness, weights=(1.0,))
        creator.create("Individual_en", list, fitness=creator.FitnessMax_en)
        toolbox_en = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_en.register("attr_ratio", random.choice, p_model['params']['ratio'])
        toolbox_en.register("attr_c", random.choice, p_model['params']['c'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_en.register("Individual_en", tools.initCycle, creator.Individual_en,
                            (toolbox_en.attr_ratio, toolbox_en.attr_c), n=1)

        # population definition
        toolbox_en.register("population", tools.initRepeat, list, toolbox_en.Individual_en)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_en(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['ratio'])
            elif gene == 1:
                individual[1] = random.choice(p_model['params']['c'])

            return individual,

        # --------------------------------------------------- funcion de evaluacion para OLS Elastic Net -- #
        def evaluate_en(eva_individual):

            # output of genetic algorithm
            chromosome = {'ratio': eva_individual[0], 'c': eva_individual[1]}

            # model results
            model = logistic_net(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,

        toolbox_en.register("mate", tools.cxOnePoint)
        toolbox_en.register("mutate", mutate_en)
        toolbox_en.register("select", tools.selTournament, tournsize=10)
        toolbox_en.register("evaluate", evaluate_en)

        population_size = 60
        crossover_probability = 0.8
        mutation_probability = 0.1
        number_of_generations = 1

        en_pop = toolbox_en.population(n=population_size)
        en_hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic Algorithm Implementation
        en_pop, en_log = algorithms.eaSimple(population=en_pop, toolbox=toolbox_en, stats=stats,
                                             cxpb=crossover_probability, mutpb=mutation_probability,
                                             ngen=number_of_generations,
                                             halloffame=en_hof, verbose=True)

        return {'pop': en_pop, 'logs': en_log, 'hof': en_hof}

    # -- --------------------------------------------------------- Least Squares Support Vector Machines -- #
    # ----------------------------------------------------------------------------------------------------- #

    elif p_model['label'] == 'ls-svm':

        # borrar clases previas si existen
        try:
            del creator.FitnessMax_svm
            del creator.Individual_svm
        except AttributeError:
            pass

        # inicializar ga
        creator.create("FitnessMax_svm", base.Fitness, weights=(1.0, ))
        creator.create("Individual_svm", list, fitness=creator.FitnessMax_svm)
        toolbox_svm = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_svm.register("attr_c", random.choice, p_model['params']['c'])
        toolbox_svm.register("attr_kernel", random.choice, p_model['params']['kernel'])
        toolbox_svm.register("attr_gamma", random.choice, p_model['params']['gamma'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_svm.register("Individual_svm", tools.initCycle, creator.Individual_svm,
                             (toolbox_svm.attr_c, toolbox_svm.attr_kernel, toolbox_svm.attr_gamma), n=1)

        # population definition
        toolbox_svm.register("population", tools.initRepeat, list, toolbox_svm.Individual_svm)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_svm(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['c'])
            elif gene == 1:
                if individual[1] == 'linear':
                    individual[1] = 'rbf'
                else:
                    individual[1] = 'linear'
            elif gene == 2:
                if individual[2] == 'scale':
                    individual[2] = 'auto'
                else:
                    individual[2] = 'scale'
            return individual,

        # ------------------------------------------------------------ funcion de evaluacion para LS SVM -- #
        def evaluate_svm(eval_individual):

            # output of genetic algorithm
            chromosome = {'c': eval_individual[0], 'kernel': eval_individual[1], 'gamma': eval_individual[2]}

            # model results
            model = ls_svm(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,

        toolbox_svm.register("mate", tools.cxOnePoint)
        toolbox_svm.register("mutate", mutate_svm)
        toolbox_svm.register("select", tools.selTournament, tournsize=10)
        toolbox_svm.register("evaluate", evaluate_svm)

        population_size = 60
        crossover_probability = 0.8
        mutation_probability = 0.1
        number_of_generations = 1

        svm_pop = toolbox_svm.population(n=population_size)
        svm_hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic Algortihm implementation
        svm_pop, svm_log = algorithms.eaSimple(population=svm_pop, toolbox=toolbox_svm, stats=stats,
                                               cxpb=crossover_probability, mutpb=mutation_probability,
                                               ngen=number_of_generations,
                                               halloffame=svm_hof, verbose=True)

        return {'pop': svm_pop, 'logs': svm_log, 'hof': svm_hof}

    # -- ----------------------------------------------- Artificial Neural Network MultiLayer Perceptron -- #
    # ----------------------------------------------------------------------------------------------------- #

    elif p_model['label'] == 'ann-mlp':

        # borrar clases previas si existen
        try:
            del creator.FitnessMax_mlp
            del creator.Individual_mlp
        except AttributeError:
            pass

        # inicializar ga
        creator.create("FitnessMax_mlp", base.Fitness, weights=(1.0,))
        creator.create("Individual_mlp", list, fitness=creator.FitnessMax_mlp)
        toolbox_mlp = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_mlp.register("attr_hidden_layers", random.choice, p_model['params']['hidden_layers'])
        toolbox_mlp.register("attr_activation", random.choice, p_model['params']['activation'])
        toolbox_mlp.register("attr_alpha", random.choice, p_model['params']['alpha'])
        toolbox_mlp.register("attr_learning_r", random.choice, p_model['params']['learning_r'])
        toolbox_mlp.register("attr_learning_r_init", random.choice, p_model['params']['learning_r_init'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_mlp.register("Individual_mlp", tools.initCycle, creator.Individual_mlp,
                             (toolbox_mlp.attr_hidden_layers,
                              toolbox_mlp.attr_activation,
                              toolbox_mlp.attr_alpha,
                              toolbox_mlp.attr_learning_r,
                              toolbox_mlp.attr_learning_r_init), n=1)

        # population definition
        toolbox_mlp.register("population", tools.initRepeat, list, toolbox_mlp.Individual_mlp)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_mlp(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['hidden_layers'])
            elif gene == 1:
                individual[1] = random.choice(p_model['params']['activation'])
            elif gene == 2:
                individual[2] = random.choice(p_model['params']['alpha'])
            elif gene == 3:
                individual[3] = random.choice(p_model['params']['learning_r'])
            elif gene == 4:
                individual[4] = random.choice(p_model['params']['learning_r_init'])
            return individual,

        # ------------------------------------------------------------ funcion de evaluacion para LS SVM -- #
        def evaluate_mlp(eval_individual):

            # output of genetic algorithm
            chromosome = {'hidden_layers': eval_individual[0], 'activation': eval_individual[1],
                          'alpha': eval_individual[2], 'learning_r': eval_individual[3],
                          'learning_r_init': eval_individual[4]}

            # model results
            model = ann_mlp(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,

        toolbox_mlp.register("mate", tools.cxOnePoint)
        toolbox_mlp.register("mutate", mutate_mlp)
        toolbox_mlp.register("select", tools.selTournament, tournsize=10)
        toolbox_mlp.register("evaluate", evaluate_mlp)

        population_size = 60
        crossover_probability = 0.8
        mutation_probability = 0.1
        number_of_generations = 1

        mlp_pop = toolbox_mlp.population(n=population_size)
        mlp_hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # ga algorithjm
        mlp_pop, mlp_log = algorithms.eaSimple(population=mlp_pop, toolbox=toolbox_mlp, stats=stats,
                                               cxpb=crossover_probability, mutpb=mutation_probability,
                                               ngen=number_of_generations,
                                               halloffame=mlp_hof, verbose=True)

        return {'pop': mlp_pop, 'logs': mlp_log, 'hof': mlp_hof}

    return 'error, sin modelo seleccionado'


# -------------------------------------------------------------------------- Model Evaluations by period -- #
# --------------------------------------------------------------------------------------------------------- #

def model_evaluations(p_features, p_optim_data, p_model):

    for params in list(p_optim_data):

        if p_model == 'logistic-elasticnet':
            parameters = {'ratio': params[0], 'c': params[1]}

            return logistic_net(p_data=p_features, p_params=parameters)

        elif p_model == 'ls-svm':
            parameters = {'c': params[0], 'kernel': params[1], 'gamma': params[2]}

            return ls_svm(p_data=p_features, p_params=parameters)

        elif p_model == 'ann-mlp':
            parameters = {'hidden_layers': params[0], 'activation': params[1], 'alpha': params[2],
                          'learning_r': params[3], 'learning_r_init': params[4]}

            return ann_mlp(p_data=p_features, p_params=parameters)


# -------------------------------------------------------------------------- Model Evaluations by period -- #
# --------------------------------------------------------------------------------------------------------- #

def folds_evaluations(p_data_folds, p_models, p_saving, p_file_name):
    """
    Global evaluations for specified data folds for specified models

    Parameters
    ----------
    p_data_folds: dict

    p_models: list

    p_saving: bool

    p_file_name: str

    Returns
    -------
    memory_palace: dict

    """

    # main data structure for calculations
    memory_palace = {j: {i: {'e_hof': [], 'time': [], 'features': {}}
                         for i in p_data_folds} for j in list(dt.models.keys())}

    # cycle to iterate all periods for all models
    for period in p_data_folds:
        for model in p_models:

            # time measurement
            init = datetime.now()

            print('\n')
            print('----------------------------')
            print('modelo: ', model)
            print('periodo: ', period)
            print('----------------------------')
            print('\n')
            print('----------------------- Ingenieria de Variables por Periodo ------------------------')
            print('----------------------- ----------------------------------- ------------------------')

            # scale data of the corresponding fold to evaluate
            data_folds = data_scaler(p_data=p_data_folds[period].copy(), p_trans='Standard')

            # Feature engineering (Autoregressive, Hadamard, Symbolic)
            m_features = genetic_programed_features(p_data=data_folds, p_memory=7)

            # Save features used in the evaluation in memory_palace
            memory_palace[model][period]['features'] = m_features

            # Optimization
            print('\n')
            print('--------------------- Optimizacion de hiperparametros por Periodo ------------------')
            print('--------------------- ------------------------------------------- ------------------')

            # -- model optimization and evaluation for every element in the Hall of Fame for every period
            # optimization process
            hof_model = genetic_algo_optimisation(p_data=m_features, p_model=dt.models[model])

            # Save logs information of the optimisation process
            # memory_palace[model][period]['logs'] = hof_model['logs']

            # Save Hall of Fame information of the optimisation process
            # memory_palace[model][period]['hof'] = hof_model['hof']

            # evaluation process
            for i in range(0, len(list(hof_model['hof']))):
                hof_eval = model_evaluations(p_features=m_features, p_model=model,
                                                p_optim_data=hof_model['hof'])
                # save evaluation in memory_palace
                memory_palace[model][period]['e_hof'].append(hof_eval)

            # time measurement
            end = datetime.now()
            print("\nElapsed Time =", end - init)
            memory_palace[model][period]['time'] = end - init

    # -- --------------------------------------------------------------------- Save Data for offline use -- #

    if p_saving:
        # objects to be saved
        pickle_rick = {'data': dt.ohlc_data, 'models': p_models,
                       't_folds': p_data_folds, 'memory_palace': memory_palace}

        # pickle format function
        dt.data_save_load(p_data_objects=pickle_rick,
                          p_data_file=p_file_name,
                          p_data_action='save')

    return memory_palace


# -------------------------------------------------------------------------- Model AUC Min and Max Cases -- #
# --------------------------------------------------------------------------------------------------------- #

def models_auc(p_models, p_global_cases, p_data_folds):
    """
    AUC min and max cases for the models

    Parameters
    ----------
    p_models: list
        with the models name

    p_global_cases: dict
        With all the info for the global cases

    p_data_folds: dict
        with all the historical data info in folds

    Returns
    -------
    auc_cases:dict
        with all the info of the min and the max case for every model

    """

    # diccionario para almacenar resultados de busqueda
    auc_cases = {j: {i: {'data': {}, 'period':''}
                     for i in ['auc_min', 'auc_max', 'hof_metrics']} for j in p_models}

    # ciclo para busqueda de auc_min y auc_max
    for model in p_models:
        auc_min = 1
        auc_max = 0
        auc_max_params = {}
        auc_min_params = {}
        for period in p_data_folds:
            auc_cases[model]['hof_metrics']['data'][period] = {}
            auc_s = []
            for i in range(0, 10):
                auc_s.append(p_global_cases[model][period]['e_hof'][i]['metrics']['test']['auc'])

                # -- caso 1
                # El individuo de todos los HOF de todos los periodos que produjo la minima AUC
                if p_global_cases[model][period]['e_hof'][i]['metrics']['test']['auc'] < auc_min:
                    auc_min = p_global_cases[model][period]['e_hof'][i]['metrics']['test']['auc']
                    auc_cases[model]['auc_min']['data'] = p_global_cases[model][period]['e_hof'][i]
                    auc_cases[model]['auc_min']['period'] = period
                    auc_min_params = p_global_cases[model][period]['e_hof'][i]['params']

                # -- caso 2
                # El individuo de todos los HOF de todos los periodos que produjo la maxima AUC
                elif p_global_cases[model][period]['e_hof'][i]['metrics']['test']['auc'] > auc_max:
                    auc_max = p_global_cases[model][period]['e_hof'][i]['metrics']['test']['auc']
                    auc_cases[model]['auc_max']['data'] = p_global_cases[model][period]['e_hof'][i]
                    auc_cases[model]['auc_max']['period'] = period
                    auc_max_params = p_global_cases[model][period]['e_hof'][i]['params']

            # Guardar info por periodo
            auc_cases[model]['hof_metrics']['data'][period]['auc_s'] = auc_s
            auc_cases[model]['hof_metrics']['data'][period]['auc_max'] = auc_max
            auc_cases[model]['hof_metrics']['data'][period]['auc_max_params'] = auc_max_params
            auc_cases[model]['hof_metrics']['data'][period]['auc_min'] = auc_min
            auc_cases[model]['hof_metrics']['data'][period]['auc_min_params'] = auc_min_params

    return auc_cases


# -------------------------------------------------------------------------------- Model Global Features -- #
# --------------------------------------------------------------------------------------------------------- #

def model_features(p_model_data, p_memory, p_model, p_global_cases, p_cases):

    import sympy as sym

    data_arf = autoregressive_features(p_data=p_model_data.copy(), p_nmax=p_memory)
    # independent (explanatory) candidate variables separation
    data_arf = data_arf.drop(['timestamp', 'co', 'co_d'], axis=1, inplace=False)
    # function to generate hadamard product features
    data_had = hadamard_features(p_data=data_arf.copy(), p_nmax=p_memory)
    features = pd.concat([data_arf.copy(), data_had.copy()], axis=1)

    period = p_cases[p_model]['auc_min']['period']
    equations = p_global_cases[p_model][period]['features']['features_eq']

    # -- MISSING -- #
    # HOW TO TAKE THE EQUATION GENERATED BY THE GENETIC PROGRAMMING PART, AND, EVALUATE IT WITH
    # THE INFORMATION OF THE COLUMN IN THE DATA, PRODUCE THE FEATURE.

    return 1


# ------------------------------------------------------------------------------ Model Global Evaluation -- #
# --------------------------------------------------------------------------------------------------------- #

def model_evaluation(p_data, p_memory, p_global_cases, p_models, p_cases):
    """
    Evaluation of models with global data and features for particular selected cases of parameters

    Parameters
    ----------
    p_data: dict
        The data to use in the model evaluation

    p_memory: int
        Memory value for the timeseries calculations

    p_global_cases: pd.DataFrame
        with all the features (inputs)

    p_models: list
        with the models name

    p_cases: dict
        with the information of the min and max AUC cases

    Returns
    -------
    global_auc_cases: dict
        with the evaluations

    """

    # Evaluation of auc_min and auc_max cases in all the models
    global_auc_cases = {model: {'auc_min': {}, 'auc_max': {}} for model in p_models}

    # Evaluate all global cases
    for model in p_models:
        for case in ['auc_min', 'auc_max']:

            if model == 'logistic-elasticnet':

                # Calculate case features, according to the information of the features used in
                # the min AUC and max AUC found for the particular model
                case_features = model_features(p_model_data=p_data, p_memory=p_memory,
                                               p_model='logistic-elasticnet',
                                               p_global_cases=p_global_cases,
                                               p_cases=p_cases)

                # get the results of the input features into the model with the optimised parameters
                global_auc_cases[model][case] = logistic_net(p_data=case_features,
                                                             p_params=p_cases[model][case]['data']['params'])
            elif model == 'ls-svm':

                # Calculate case features, according to the information of the features used in
                # the min AUC and max AUC found for the particular model
                case_features = model_features(p_model_data=p_data, p_memory=p_memory,
                                               p_model='ls-svm',
                                               p_global_cases=p_global_cases,
                                               p_cases=p_cases)

                # get the results of the input features into the model with the optimised parameters
                global_auc_cases[model][case] = ls_svm(p_data=case_features,
                                                       p_params=p_cases[model][case]['data']['params'])
            elif model == 'ann-mlp':

                # Calculate case features, according to the information of the features used in
                # the min AUC and max AUC found for the particular model
                case_features = model_features(p_model_data=p_data, p_memory=p_memory,
                                               p_model='ann-mlp',
                                               p_global_cases=p_global_cases,
                                               p_cases=p_cases)

                # get the results of the input features into the model with the optimised parameters
                global_auc_cases[model][case] = ann_mlp(p_data=case_features,
                                                        p_params=p_cases[model][case]['data']['params'])

    return global_auc_cases
