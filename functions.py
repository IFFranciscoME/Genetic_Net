
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
import warnings


# ----------------------------------------------------------------- FUNCTION: Divide the data in T-Folds -- #
# ----------------------------------------------------------------- --------------------------------------- #

def t_folds(p_data, p_period):
    """
    Funcion para dividir los datos en m-bloques, donde m es un valor basado en tiempo:
        m={'mensual', 'trimestral'}

    Parameters
    ----------
    p_data : pd.DataFrame
        DataFrame con los datos a dividir

    p_period : str
        'mes': Para dividir datos por periodos mensuales
        'quarter' para dividir datos por periodos trimestrales

    Returns
    -------
    {'periodo_': pd.DataFrame}

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
