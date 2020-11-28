
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


# ----------------------------------------------------------------- FUNCTION: Divide the data in T-Folds -- #
# ----------------------------------------------------------------- --------------------------------------- #

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
            print(y)
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
