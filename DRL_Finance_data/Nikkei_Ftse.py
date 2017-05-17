# coding: utf-8

# In[1]:

import pandas as pd

import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import matplotlib.pyplot as plt


# In[2]:

# Nikkei and FTSE time series

# nikkei = bq.Query.from_table(bq.Table('bingo-ml-1.market_data.nikkei'), fields=['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
# ftse = bq.Query.from_table(bq.Table('bingo-ml-1.market_data.ftse'), fields=['Date', 'Close']).execute().result().to_dataframe().set_index('Date')

nikkei = pd.read_excel("nikkei.csv", encoding='utf-8', names = ['date', 'price']).set_index("date").sort_index()
ftse = pd.read_excel("ftse.csv", encoding='utf-8', names = ['date', 'price']).set_index("date").sort_index()

# In[3]:nikkei = pd.read_excel("nikkei.csv", encoding='utf-8', names = ['date', 'price'])


closing_data = pd.DataFrame()

closing_data['nikkei_close'] = nikkei['price']
closing_data['ftse_close'] = ftse['price']

closing_data = closing_data.fillna(method = 'ffill')


# In[4]:

closing_data.describe()
closing_data = closing_data.sort_index()
desc = closing_data.describe()
#closing_data

amount_invested = 1
initial_endowment = 1


# In[5]:

# Portfolio definition

len_series = len(nikkei)

def price_vector(time_lag, t):
  return closing_data[t-time_lag:t]

def current_price_nikkei(t):
  return closing_data['nikkei_close'][t]

def current_price_ftse(t):
  return closing_data['ftse_close'][t]

def portfolio_value(portfolio_shares, t):
  prices = [current_price_nikkei(t), current_price_ftse(t)]
  #print(type(prices), type(portfolio_shares))
  return np.inner(portfolio_shares,prices);

def alpha_fun(shares,t):
  alpha = shares[0]*current_price_nikkei(t)/portfolio_value(shares,t)
  return alpha

def stat(strategy):
  result = {}
  result["final value"] = strategy[len_series-1]
  result["mean"] = np.mean(strategy)
  result["sd"] = np.std(strategy)
  result["max"] = max(strategy)
  result["min"] = min(strategy)
  return result

def stat_RL(strategy):
  strategy = strategy[:len_series-2]
  result = {}
  result["final value"] = strategy[len(strategy)-1]
  result["mean"] = np.mean(strategy)
  result["sd"] = np.std(strategy)
  result["max"] = max(strategy)
  result["min"] = min(strategy)
  return result


# In[6]:

# Define common values for basic experiments




# In[7]:

def nikkei_basic(amount_invested):
  nikkei_0 =amount_invested/current_price_nikkei(0)
  ftse_0 = 0
  shares_0 = [nikkei_0, ftse_0]
  value_0 = portfolio_value(shares_0,0)
  alpha_0 = alpha_fun(shares_0,0)
  cash_0 = initial_endowment - amount_invested

  state_nikkei = pd.DataFrame()

  state_nikkei['price_nikkei'] = np.zeros(len_series)
  state_nikkei['price_ftse'] = np.zeros(len_series)
  state_nikkei['portfolio_shares_nikkei'] = np.zeros(len_series)
  state_nikkei['portfolio_shares_ftse'] = np.zeros(len_series)
  state_nikkei['value_nikkei'] = np.zeros(len_series)
  state_nikkei['alpha'] = np.ones(len_series)
  state_nikkei['cash'] = np.zeros(len_series)

  state_nikkei['price_nikkei'][0] = current_price_nikkei(0)
  state_nikkei['price_ftse'][0] = current_price_ftse(0)
  state_nikkei['portfolio_shares_nikkei'] = shares_0[0] * np.ones(len_series)
  state_nikkei['portfolio_shares_ftse'] = shares_0[1] *np.ones(len_series)
  state_nikkei['value_nikkei'][0] = value_0

  for time in range(len_series):
    state_nikkei['price_nikkei'][time] = current_price_nikkei(time)
    state_nikkei['price_ftse'][time] = current_price_ftse(time)
    state_nikkei['value_nikkei'][time] = portfolio_value(shares_0,time)
  return state_nikkei


# In[8]:

def ftse_basic(amount_invested):
  nikkei_0 =0
  ftse_0 = amount_invested/current_price_ftse(0)
  shares_0 = [nikkei_0, ftse_0]
  value_0 = portfolio_value(shares_0,0)
  alpha_0 = alpha_fun(shares_0,0)
  cash_0 = initial_endowment - amount_invested

  state_ftse = pd.DataFrame()

  state_ftse['price_nikkei'] = np.zeros(len_series)
  state_ftse['price_ftse'] = np.zeros(len_series)
  state_ftse['portfolio_shares_nikkei'] = np.zeros(len_series)
  state_ftse['portfolio_shares_ftse'] = np.zeros(len_series)
  state_ftse['value_ftse'] = np.zeros(len_series)
  state_ftse['alpha'] = np.ones(len_series)
  state_ftse['cash'] = np.zeros(len_series)

  state_ftse['price_nikkei'][0] = current_price_nikkei(0)
  state_ftse['price_ftse'][0] = current_price_ftse(0)
  state_ftse['portfolio_shares_nikkei'] = shares_0[0] * np.ones(len_series)
  state_ftse['portfolio_shares_ftse'] = shares_0[1] *np.ones(len_series)
  state_ftse['value_ftse'][0] = value_0

  for time in range(len_series):
      state_ftse['price_nikkei'][time] = current_price_nikkei(time)
      state_ftse['price_ftse'][time] = current_price_ftse(time)
      state_ftse['value_ftse'][time] = portfolio_value(shares_0,time)
  return state_ftse


state_nikkei = nikkei_basic(amount_invested)
state_ftse = ftse_basic(amount_invested)
import cPickle
cPickle.dump(state_nikkei, open('nikkei.p', 'wb'))
cPickle.dump(state_ftse, open('ftse.p', 'wb'))