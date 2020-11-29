
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
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split


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

    # For monthly separation of the data
    if p_period == 'month':
        # List of months in the dataset
        months = list(set(time.month for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = list(set(time.year for time in list(p_data['timestamp'])))
        m_data = {}
        # New key for every month_year
        for j in years:
            m_data.update({'m_' + str('0') + str(i) + '_' + str(j) if i <= 9 else str(i) + '_' + str(j):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.month == i) &
                                      (pd.to_datetime(p_data['timestamp']).dt.year == j)]
                           for i in months})
        return m_data

    # For quarterly separation of the data
    elif p_period == 'quarter':
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
    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
                                population_size=5000, hall_of_fame=100, n_components=20,
                                generations=50, tournament_size=20,  stopping_criteria=.65,
                                const_range=None, init_method='half and half', init_depth=(4, 16),
                                metric='pearson', parsimony_coefficient=0.01,
                                p_crossover=0.4, p_subtree_mutation=0.3, p_hoist_mutation=0.1,
                                p_point_mutation=0.2, p_point_replace=.05,
                                verbose=1, random_state=None, n_jobs=-1, feature_names=p_x.columns,
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

    # Symbolic features generation
    fun_sym = symbolic_features(p_x=data_had, p_y=data_y)

    # variables
    data_sym = fun_sym['data']
    data_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]

    # equations for the symbolic features
    # equations = [i.__str__() for i in list(fun_sym['model'])]

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

    return model_data
