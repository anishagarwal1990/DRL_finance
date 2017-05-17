
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


# In[9]:



# In[10]:

# Reward function

def reward_function(shares_1, shares_0, time):
  return portfolio_value(shares_1, time) - portfolio_value(shares_0, time-1)

def new_shares(previous_shares, action, time):
  
  prev_shares_instrument_0 =  previous_shares[0]
  prev_shares_instrument_1 = previous_shares[1]
  
  prev_price_instrument_0 = current_price_nikkei(time-1)
  prev_price_instrument_1 = current_price_ftse(time-1)
  
  prev_portfolio_value = portfolio_value(previous_shares, time-1)
  
  new_shares_instrument_0 = prev_shares_instrument_0 + action * prev_portfolio_value / prev_price_instrument_0
  new_shares_instrument_1 = prev_shares_instrument_1 - action * prev_portfolio_value / prev_price_instrument_1

  return [new_shares_instrument_0, new_shares_instrument_1]

def valid_actions(current_shares,time):
  action_range = list(np.linspace(-.4, .4, num=20))
  # action_range = [-.3,-.2,-.1, 0 , .1, .2, .3]
  valid_actions_list = [];
  for action in action_range:
    x_0 = new_shares(current_shares, action,time)[0]
    x_1 = new_shares(current_shares, action,time)[1]
    # print x_0,x_1
    if(x_0>0 and x_1>0):
      valid_actions_list.append(action)
  return valid_actions_list

def RL_next_state(current_shares, action, time_lag, time):
    state = {};
    state["price_matrix"] = price_vector(time_lag,time).as_matrix()
    state["price_matrix"] = np.ndarray.tolist(state["price_matrix"])
    state["shares"] = new_shares(current_shares, action, time)
    state["value_RL"] = portfolio_value(state["shares"], time)
    state["alpha"] = alpha_fun(state["shares"], time) 
    return state

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
  
def RL_greedy_action(current_state, model, time_lag, time):
  current_shares = current_state["shares"]
  valid_act = valid_actions(current_shares, time)
  best_act = 0
  best_val = -10e5
  for act in valid_act:
    new_state = RL_next_state(current_shares, act, time_lag, time)
    current_state_list = flatten(new_state.values())
    act_val = model.predict(current_state_list+ [act])
    if act_val > best_val:
      best_act = act
      best_val = act_val
  return best_act


# In[11]:

def basic_RL(amount_invested, model, model_temp, alpha, tau, lag, train_0, param_string):
  # Initialize the state for alpha = 0, do nothing for first 100 steps so to collect some data;
  #train_0 = 200;
  lag = 20;
  initial_alpha = .5
  df = .99
  X = []
  y = []

  nikkei_0 = amount_invested*initial_alpha/(current_price_nikkei(0))
  ftse_0 = amount_invested*(1-initial_alpha)/current_price_ftse(0)
  shares_0 = [nikkei_0, ftse_0]
  value_0 = portfolio_value(shares_0,0)

  # initialize theta?
  
  
  state_RL = pd.DataFrame()
  state_RL['price_nikkei'] = np.zeros(len_series)
  state_RL['price_ftse'] = np.zeros(len_series)
  state_RL['portfolio_shares_nikkei'] = np.zeros(len_series)
  state_RL['portfolio_shares_ftse'] = np.zeros(len_series)
  state_RL['value_RL'] = np.zeros(len_series)
  state_RL['action_RL'] = np.zeros(len_series)
  state_RL['alpha'] = np.ones(len_series)

  
  for time in range(lag):
    current_shares = shares_0;
    state_RL['price_nikkei'][time] = current_price_nikkei(time)
    state_RL['price_ftse'][time] = current_price_ftse(time)
    state_RL['portfolio_shares_nikkei'][time] = shares_0[0]
    state_RL['portfolio_shares_ftse'][time] = shares_0[1]
    state_RL['value_RL'][time] = portfolio_value(shares_0, time)
    state_RL['action_RL'][time] = 0;
    state_RL['alpha'][time] = alpha_fun(shares_0, time)
    
  state = RL_next_state(shares_0, 0, lag, lag)
  for time in range(lag,train_0):
      current_shares = state["shares"];
      state = RL_next_state(current_shares, 0, lag, time)
      state_values = state.values()
      x = flatten(state_values)
      X.append(x+[0])
      y.append(portfolio_value(current_shares,time+1))
      state_RL['price_nikkei'][time] = current_price_nikkei(time)
      state_RL['price_ftse'][time] = current_price_ftse(time)
      state_RL['portfolio_shares_nikkei'][time] = state["shares"][0]
      state_RL['portfolio_shares_ftse'][time] = state["shares"][1]
      state_RL['value_RL'][time] = state["value_RL"]
      state_RL['action_RL'][time] = 0;
      state_RL['alpha'][time] = alpha_fun(state["shares"], time)
  model.fit(X,y)
  theta = model.coef_
    # Train the model using the training sets

  
  # start reinforcing
  for time in range(lag, len_series-1):
  # for time in range(lag, lag + 5):
    prev_shares = state["shares"];
    greedy_action = RL_greedy_action(state, model, lag, time)
    state = RL_next_state(prev_shares, greedy_action, lag, time)
    next_shares = state["shares"]
    state_values = state.values()
    x = flatten(state_values)
    X.append(x+[greedy_action])    
    next_greedy_action = RL_greedy_action(state, model, lag, time+1)
    Q_value = reward_function(next_shares, current_shares, time) + df* model.predict(x + [next_greedy_action])
    # print Q_value
    y.append(Q_value)   
    model_temp.fit(X, y)
    theta_temp = model_temp.coef_
    theta = (1-tau)*theta + tau*theta_temp
    model.coef_ = theta
    
    state_RL['price_nikkei'][time] = current_price_nikkei(time)
    state_RL['price_ftse'][time] = current_price_ftse(time)
    state_RL['portfolio_shares_nikkei'][time] = next_shares[0]
    state_RL['portfolio_shares_ftse'][time] = next_shares[1]
    state_RL['value_RL'][time] = state["value_RL"]
    state_RL['action_RL'][time] = greedy_action;
    state_RL['alpha'][time] = alpha_fun(next_shares, time)
    if time%50 == 0:
      print time
  state_RL['price_nikkei'][len_series-1] = current_price_nikkei(len_series-1)
  state_RL['price_ftse'][len_series-1] = current_price_ftse(len_series-1)
  state_RL['portfolio_shares_nikkei'][len_series-1] = next_shares[0]
  state_RL['portfolio_shares_ftse'][len_series-1] = next_shares[1]
  state_RL['value_RL'][len_series-1] = state["value_RL"]
  state_RL['action_RL'][len_series-1] = param_string;
  state_RL['alpha'][len_series-1] = alpha_fun(next_shares, len_series-2)

  #state_RL._metadata.append(param_string)
  return state_RL



# In[26]:



from sklearn import linear_model
import multiprocessing as mp


### GLOBAL PARAMETERS
initial_endowment = 1
amount_invested = 1
alpha = 0.5
train_0 = 100

state_nikkei = nikkei_basic(amount_invested)
statistics_nikkei = stat(state_nikkei['value_nikkei'])


state_ftse = ftse_basic(amount_invested)
statistics_ftse = stat(state_ftse['value_ftse'])


# model_1 = linear_model.LinearRegression()
# model_1_temp = linear_model.LinearRegression()
# tau_list = [.8, .9, .95]
# lag_list = [15, 30]
# model_list = [(linear_model.LinearRegression(), linear_model.LinearRegression(), "linReg"), \
#               (linear_model.Lasso(alpha = 0.1), linear_model.Lasso(alpha = 0.1), "Lasso0.1"), \
#               (linear_model.Lasso(alpha = 0.5),linear_model.Lasso(alpha = 0.5), "Lasso0.5"), \
#               (linear_model.Lasso(alpha = 0.9), linear_model.Lasso(alpha = 0.9), "Lasso0.9")]
#
# experiments = []
# for model_tup in model_list:
#   for lag in lag_list:
#     for tau in tau_list:
#       exp_model, exp_model_temp, model_name = model_tup[0], model_tup[1], model_tup[2]
#       param_string = "__model__" + model_name + "__tau__" + str(tau) + "__lag__" + str(lag)
#       temp_param= amount_invested, exp_model, exp_model_temp, alpha, tau, lag, train_0, param_string
#       experiments.append(temp_param)
#
# pool = mp.Pool(processes = mp.cpu_count())
# results = [pool.apply_async(basic_RL, args=exp) for exp in experiments]
# output = [p.get() for p in results]
#
#
# import cPickle
# cPickle.dump(output, open('Experiments_no_shock.p', 'wb'))

# pkl_file =  open('Experiments_no_shock.p', 'rb')
# data = cPickle.load(pkl_file)
#
# for test in data:
#   print test['action_RL'][len(test['action_RL'])-1]

# from sklearn.neural_network import MLPRegressor

# model = linear_model.LinearRegression()
# model_temp = linear_model.LinearRegression()
#
# # model = neural_network.MLPRegressor()
# # model_temp = neural_network.MLPRegressor()
#
# # model = linear_model.Lasso(alpha = 0.1)
# # model_temp = linear_model.Lasso(alpha = 0.1)
# # state_RL_lasso = basic_RL(amount_invested, model, model_temp,  .5, .99, 20, 100)
#
#
# # In[16]:
# #
# model = linear_model.LinearRegression()
# model_temp = linear_model.LinearRegression()
# state_RL_lin = basic_RL(amount_invested, model, model_temp,  .5, .9, 20, 100, 'hhh')
#
# print state_RL_lin.my_attribute


# In[18]:

# fraction_active_lasso = float(np.count_nonzero(state_RL_lasso['action_RL']))/(len_series-20)
# print 'fraction_active_lasso', fraction_active_lasso
# statistics_RL_lasso = stat_RL(state_RL_lasso['value_RL'])
# print 'statistics_RL_lasso', statistics_RL_lasso
#
#
#
# statistics_RL_lin = stat_RL(state_RL_lin['value_RL'])
# print 'statistics_RL_lin', statistics_RL_lin
# _ = pd.concat([state_nikkei['value_nikkei'], state_ftse['value_ftse'], state_RL_lin['value_RL'], state_RL_lasso['value_RL']],axis = 1).plot(figsize=(18,8))
# # plt.savefig('linear_reinforcement.png')
#
#
# # In[19]:
#
# model = linear_model.Lasso(alpha = 0.9)
# model_temp = linear_model.Lasso(alpha = 0.9)
# state_RL_lasso_9 = basic_RL(amount_invested, model, model_temp,  .5, .99, 30, 150)
#
#
# # In[21]:
#
# model = linear_model.Lasso(alpha = 0.01)
# model_temp = linear_model.Lasso(alpha = 0.01)
# state_RL_lasso_0 = basic_RL(amount_invested, model, model_temp,  .5, .99, 30, 150)
#
#
# # In[23]:
#
# model = linear_model.Lasso(alpha = 0.001)
# model_temp = linear_model.Lasso(alpha = 0.001)
# state_RL_lasso_00 = basic_RL(amount_invested, model, model_temp,  .5, .9, 30, 150)
# #
#
# # In[25]:
#
# fraction_active_lasso_9 = float(np.count_nonzero(state_RL_lasso_9['action_RL']))/(len_series-20)
# print 'fraction_active_lasso', fraction_active_lasso_9
# statistics_RL_lasso_9 = stat_RL(state_RL_lasso_9['value_RL'])
# print 'statistics_RL_lasso_9', statistics_RL_lasso_9
#
# fraction_active_lasso_0 = float(np.count_nonzero(state_RL_lasso_0['action_RL']))/(len_series-20)
# print 'fraction_active_lasso_0', fraction_active_lasso_0
# statistics_RL_lasso_0 = stat_RL(state_RL_lasso_0['value_RL'])
# print 'statistics_RL_lasso_0', statistics_RL_lasso_0
#
# fraction_active_lasso_00 = float(np.count_nonzero(state_RL_lasso_00['action_RL']))/(len_series-20)
# print 'fraction_active_lasso_00', fraction_active_lasso_00
# statistics_RL_lasso_00 = stat_RL(state_RL_lasso_00['value_RL'])
# print 'statistics_RL_lasso_00', statistics_RL_lasso_00
#
#
# _ = pd.concat([state_nikkei['value_nikkei'], state_ftse['value_ftse'], state_RL_lasso['value_RL'], state_RL_lasso_9['value_RL'], state_RL_lasso_0['value_RL'], state_RL_lasso_00['value_RL']],axis = 1).plot(figsize=(18,8))
# # plt.savefig('linear_reinforcement_2.png')
#
# Create a figure of size 8x6 inches, 80 dots per inch
# plt.figure(figsize=(20, 8))
#
# # Create a new subplot from a grid of 1x1
# plt.subplot(1, 1, 1)
#
# X = range(len_series)
# value_RL = state_RL_lin["value_RL"]
# value_nikkei = state_nikkei["value_nikkei"]
# value_ftse = state_ftse["value_ftse"]
# #value_RL_lasso = state_RL_lasso_00["value_RL"]
#
# # Plot cosine with a blue continuous line of width 1 (pixels)
# plt.plot(X, value_RL, color="blue", linewidth=1.0, linestyle="-")
# plt.plot(X, value_ftse, color="red", linewidth=1.0, linestyle="-")
# plt.plot(X, value_nikkei, color="black", linewidth=1.0, linestyle="-")
#plt.plot(X, value_RL_lasso, color="orange", linewidth=1.0, linestyle="-")



#
# # Set x limits
# plt.xlim(-4.0, 4.0)
#
# # Set x ticks
# plt.xticks(np.linspace(-4, 4, 9, endpoint=True))
#
# # Set y limits
# plt.ylim(-1.0, 1.0)
#
# # Set y ticks
# plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

# Save figure using 72 dots per inch
# plt.savefig("exercice_2.png", dpi=72)

# Show result on screen
# plt.show()