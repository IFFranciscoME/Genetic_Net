
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
