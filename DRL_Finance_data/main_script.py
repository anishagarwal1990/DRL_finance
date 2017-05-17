# import StringIO
# import tensorflow as tf
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt

class FinInstrument(object):
    def __init__(self, name, raw_excel):
        self.name = name
        self.dataFrame = pd.read_excel(raw_excel, encoding='utf-8', names = ['date', 'price'])
        self.dataFrame = self.dataFrame.set_index("date").sort_index()
        self.length = len(self.dataFrame)
        self.state = pd.DataFrame()
        self.state["price"] = np.zeros(self.length)
        self.state["shares"] = np.zeros(self.length)
        self.state["value"] = np.zeros(self.length)
        
    def get_price(self, t):
        return self.dataFrame["price"][t]
    
    def get_shares(self, t):
        return self.dataFrame["shares"][t]
    
    def get_lagged_prices(self, time_lag, t):
        return self.dataFrame["price"][t-time_lag:t]

class Portfolio(object):
    def __init__(self, name, fin_instrument_dict): 
        self.name = name
        self.finInstruments = fin_instrument_dict
        
    def get_shares(self, fin_instrument_name, t):
        return self.finInstruments[fin_instrument_name].get_shares(t)

    def get_price(self, fin_instrument_name, t):
        return self.finInstruments[fin_instrument_name].get_price(t)
        
    def calculate_portfolio_value(self, t):
        value_per_instrument = [self.get_shares(instrument, t) * self.get_price(instrument, t) for \
                                instrument in self.finInstruments.keys()]
        return reduce(lambda x, y: x + y, value_per_instrument)

    def alpha_fun(shares, t):
        alpha = shares[0]*current_price_nikkei(t)/portfolio_value(shares,t)
        return alpha

class PortfolioStrategy(object):
    def __init__(self, name, Portfolio):
        self.name = name


