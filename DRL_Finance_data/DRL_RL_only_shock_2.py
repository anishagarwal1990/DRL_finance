# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn import linear_model
import multiprocessing as mp

def warn(*args, **kwargs):
    pass
warnings.warn = warn


nikkei = pd.read_excel("nikkei.csv", encoding='utf-8', names=['date', 'price']).set_index("date").sort_index()
ftse = pd.read_excel("ftse.csv", encoding='utf-8', names=['date', 'price']).set_index("date").sort_index()


# Portfolio definition
LEN_SERIES = len(nikkei)

np.random.seed(4)

low_time = .5 * LEN_SERIES
high_time = .75 * LEN_SERIES
num_iter = 100


closing_data = pd.DataFrame()

closing_data['nikkei_close'] = nikkei['price']
closing_data['ftse_close'] = ftse['price']

closing_data = closing_data.fillna(method = 'ffill')

prices_nikkei = closing_data['nikkei_close'].sort_index()
prices_ftse = closing_data['ftse_close'].sort_index().as_matrix()


increment_nikkei = list(np.zeros(LEN_SERIES))
increment_nikkei = [prices_nikkei[time] - prices_nikkei[time-1] for time in range(1,LEN_SERIES)]
increment_ftse = list(np.zeros(LEN_SERIES))
increment_ftse = [prices_ftse[time] - prices_ftse[time-1] for time in range(1,LEN_SERIES)]






random_times = np.random.randint(low_time, high_time, size=num_iter)

params_nikkei_mild = [1, 4]
random_shock_nikkei_mild = np.random.beta(params_nikkei_mild[0], params_nikkei_mild[1], num_iter)
params_nikkei_consistent = [2, 4]
random_shock_nikkei_consistent = np.random.beta(params_nikkei_consistent[0], params_nikkei_consistent[1], num_iter)
params_nikkei_severe = [6, 8]
random_shock_nikkei_severe = np.random.beta(params_nikkei_severe[0], params_nikkei_severe[1], num_iter)
params_ftse_mild = [1, 10]
random_shock_ftse_mild = np.random.beta(params_ftse_mild[0], params_ftse_mild[1], num_iter)
params_ftse_consistent = [2, 10]
random_shock_ftse_consistent = np.random.beta(params_ftse_consistent[0], params_ftse_consistent[1], num_iter)
params_ftse_severe = [3, 10]
random_shock_ftse_severe = np.random.beta(params_ftse_severe[0], params_ftse_severe[1], num_iter)

nikkei_mild = np.zeros((num_iter, LEN_SERIES))
for i in range(num_iter):
    prices_shock_nikkei = list(prices_nikkei)
    random_time = random_times[i]
    prices_shock_nikkei[random_time] = (1 - random_shock_nikkei_mild[i]) * prices_nikkei[random_time]
    # print prices_shock_nikkei[random_time], prices_nikkei[random_time]
    for time in range(random_time + 1, LEN_SERIES - 1):
        prices_shock_nikkei[time] = prices_shock_nikkei[time - 1] + increment_nikkei[time]
    prices_shock_nikkei[-1] = prices_shock_nikkei[-2] + increment_nikkei[-1]
    nikkei_mild[i, :] = prices_shock_nikkei

nikkei_consistent = np.zeros((num_iter, LEN_SERIES))
for i in range(num_iter):
    prices_shock_nikkei = list(prices_nikkei)
    random_time = random_times[i]
    prices_shock_nikkei[random_time] = (1 - random_shock_nikkei_consistent[i]) * prices_nikkei[random_time]
    # print prices_shock_nikkei[random_time], prices_nikkei[random_time]
    for time in range(random_time + 1, LEN_SERIES - 1):
        prices_shock_nikkei[time] = prices_shock_nikkei[time - 1] + increment_nikkei[time]
    prices_shock_nikkei[-1] = prices_shock_nikkei[-2] + increment_nikkei[-1]
    nikkei_consistent[i, :] = prices_shock_nikkei

nikkei_severe = np.zeros((num_iter, LEN_SERIES))
for i in range(num_iter):
    prices_shock_nikkei = list(prices_nikkei)
    random_time = random_times[i]
    prices_shock_nikkei[random_time] = (1 - random_shock_nikkei_severe[i]) * prices_nikkei[random_time]
    # print prices_shock_nikkei[random_time], prices_nikkei[random_time]
    for time in range(random_time + 1, LEN_SERIES - 1):
        prices_shock_nikkei[time] = prices_shock_nikkei[time - 1] + increment_nikkei[time]
    prices_shock_nikkei[-1] = prices_shock_nikkei[-2] + increment_nikkei[-1]
    nikkei_severe[i, :] = prices_shock_nikkei

ftse_mild = np.zeros((num_iter, LEN_SERIES))
for i in range(num_iter):
    prices_shock_ftse = list(prices_ftse)
    random_time = random_times[i]
    prices_shock_ftse[random_time] = (1 - random_shock_ftse_mild[i]) * prices_ftse[random_time]
    # print prices_shock_nikkei[random_time], prices_nikkei[random_time]
    for time in range(random_time + 1, LEN_SERIES - 1):
        prices_shock_ftse[time] = prices_shock_ftse[time - 1] + increment_ftse[time]
    prices_shock_ftse[-1] = prices_shock_ftse[-2] + increment_ftse[-1]
    ftse_mild[i, :] = prices_shock_ftse

ftse_consistent = np.zeros((num_iter, LEN_SERIES))
for i in range(num_iter):
    prices_shock_ftse = list(prices_ftse)
    random_time = random_times[i]
    prices_shock_ftse[random_time] = (1 - random_shock_ftse_consistent[i]) * prices_ftse[random_time]
    # print prices_shock_nikkei[random_time], prices_nikkei[random_time]
    for time in range(random_time + 1, LEN_SERIES - 1):
        prices_shock_ftse[time] = prices_shock_ftse[time - 1] + increment_ftse[time]
    prices_shock_ftse[-1] = prices_shock_ftse[-2] + increment_ftse[-1]
    ftse_consistent[i, :] = prices_shock_ftse

ftse_severe = np.zeros((num_iter, LEN_SERIES))
for i in range(num_iter):
    prices_shock_ftse = list(prices_ftse)
    random_time = random_times[i]
    prices_shock_ftse[random_time] = (1 - random_shock_ftse_severe[i]) * prices_ftse[random_time]
    # print prices_shock_nikkei[random_time], prices_nikkei[random_time]
    for time in range(random_time + 1, LEN_SERIES - 1):
        prices_shock_ftse[time] = prices_shock_ftse[time - 1] + increment_ftse[time]
    prices_shock_ftse[-1] = prices_shock_ftse[-2] + increment_ftse[-1]
    ftse_severe[i, :] = prices_shock_ftse


def price_vector(time_lag, time_series, t):
  return [time_series[0][t-time_lag:t].tolist(), time_series[1][t-time_lag:t].tolist()]

def current_price_nikkei(time_series, t):
  return time_series[0][t]

def current_price_ftse(time_series, t):
  return time_series[1][t]

def portfolio_value(portfolio_shares, time_series, t):
  prices = [current_price_nikkei(time_series, t), current_price_ftse(time_series, t)]
  return np.inner(portfolio_shares,prices);

def alpha_fun(shares, time_series, t):
  alpha = shares[0]*current_price_nikkei(time_series, t)/portfolio_value(shares, time_series, t)
  return alpha

def stat(strategy):
  result = {}
  result["final value"] = strategy[LEN_SERIES-1]
  result["mean"] = np.mean(strategy)
  result["sd"] = np.std(strategy)
  result["max"] = max(strategy)
  result["min"] = min(strategy)
  return result


# In[7]:
def nikkei_basic(AMOUNT_INVESTED, time_series):
  nikkei_0 =AMOUNT_INVESTED/current_price_nikkei(time_series, 0)
  ftse_0 = 0
  shares_0 = [nikkei_0, ftse_0]
  value_0 = portfolio_value(shares_0, time_series, 0)
  alpha_0 = alpha_fun(shares_0, time_series, 0)
  cash_0 = INITIAL_ENDOWMENT - AMOUNT_INVESTED

  state_nikkei = pd.DataFrame()

  state_nikkei['price_nikkei'] = np.zeros(LEN_SERIES)
  state_nikkei['price_ftse'] = np.zeros(LEN_SERIES)
  state_nikkei['portfolio_shares_nikkei'] = np.zeros(LEN_SERIES)
  state_nikkei['portfolio_shares_ftse'] = np.zeros(LEN_SERIES)
  state_nikkei['value_nikkei'] = np.zeros(LEN_SERIES)
  state_nikkei['alpha'] = np.ones(LEN_SERIES)
  state_nikkei['cash'] = np.zeros(LEN_SERIES)

  state_nikkei['price_nikkei'][0] = current_price_nikkei(time_series, 0)
  state_nikkei['price_ftse'][0] = current_price_ftse(time_series, 0)
  state_nikkei['portfolio_shares_nikkei'] = shares_0[0] * np.ones(LEN_SERIES)
  state_nikkei['portfolio_shares_ftse'] = shares_0[1] *np.ones(LEN_SERIES)
  state_nikkei['value_nikkei'][0] = value_0

  for time in range(LEN_SERIES):
    state_nikkei['price_nikkei'][time] = current_price_nikkei(time_series, time)
    state_nikkei['price_ftse'][time] = current_price_ftse(time_series, time)
    state_nikkei['value_nikkei'][time] = portfolio_value(shares_0, time_series, time)
  return state_nikkei

# In[8]:

def ftse_basic(AMOUNT_INVESTED, time_series):
  nikkei_0 =0
  ftse_0 = AMOUNT_INVESTED/current_price_ftse(time_series, 0)
  shares_0 = [nikkei_0, ftse_0]
  value_0 = portfolio_value(shares_0, time_series, 0)
  alpha_0 = alpha_fun(shares_0, time_series, 0)
  cash_0 = INITIAL_ENDOWMENT - AMOUNT_INVESTED

  state_ftse = pd.DataFrame()

  state_ftse['price_nikkei'] = np.zeros(LEN_SERIES)
  state_ftse['price_ftse'] = np.zeros(LEN_SERIES)
  state_ftse['portfolio_shares_nikkei'] = np.zeros(LEN_SERIES)
  state_ftse['portfolio_shares_ftse'] = np.zeros(LEN_SERIES)
  state_ftse['value_ftse'] = np.zeros(LEN_SERIES)
  state_ftse['alpha'] = np.ones(LEN_SERIES)
  state_ftse['cash'] = np.zeros(LEN_SERIES)

  state_ftse['price_nikkei'][0] = current_price_nikkei(time_series, 0)
  state_ftse['price_ftse'][0] = current_price_ftse(time_series, 0)
  state_ftse['portfolio_shares_nikkei'] = shares_0[0] * np.ones(LEN_SERIES)
  state_ftse['portfolio_shares_ftse'] = shares_0[1] *np.ones(LEN_SERIES)
  state_ftse['value_ftse'][0] = value_0

  for time in range(LEN_SERIES):
      state_ftse['price_nikkei'][time] = current_price_nikkei(time_series, time)
      state_ftse['price_ftse'][time] = current_price_ftse(time_series, time)
      state_ftse['value_ftse'][time] = portfolio_value(shares_0, time_series, time)
  return state_ftse


def reward_function(shares_1, shares_0, time_series, time):
    return portfolio_value(shares_1, time_series, time) - portfolio_value(shares_0, time_series, time - 1)


def new_shares(previous_shares, action, time_series, time):
    prev_shares_instrument_0 = previous_shares[0]
    prev_shares_instrument_1 = previous_shares[1]

    prev_price_instrument_0 = current_price_nikkei(time_series, time - 1)
    prev_price_instrument_1 = current_price_ftse(time_series, time - 1)

    prev_portfolio_value = portfolio_value(previous_shares, time_series, time - 1)

    new_shares_instrument_0 = prev_shares_instrument_0 + action * prev_portfolio_value / prev_price_instrument_0
    new_shares_instrument_1 = prev_shares_instrument_1 - action * prev_portfolio_value / prev_price_instrument_1

    return [new_shares_instrument_0, new_shares_instrument_1]


def valid_actions(current_shares, time_series, time):
    action_range = list(np.linspace(-.4, .4, num=20))
    valid_actions_list = [];
    for action in action_range:
        x_0 = new_shares(current_shares, action, time_series, time)[0]
        x_1 = new_shares(current_shares, action, time_series, time)[1]
        # print x_0,x_1
        if (x_0 > 0 and x_1 > 0):
            valid_actions_list.append(action)
    return valid_actions_list


def RL_next_state(current_shares, action, time_lag, time_series, time):
    state = {};
    state["price_matrix"] = price_vector(time_lag, time_series, time)
    state["price_matrix"] = flatten(state["price_matrix"])
    state["shares"] = new_shares(current_shares, action, time_series, time)
    state["value_RL"] = portfolio_value(state["shares"], time_series, time)
    state["alpha"] = alpha_fun(state["shares"], time_series, time)
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


def RL_greedy_action(current_state, model, time_lag, time_series, time):
    current_shares = current_state["shares"]
    valid_act = valid_actions(current_shares, time_series, time)
    best_act = 0
    best_val = -10e5
    for act in valid_act:
        new_state = RL_next_state(current_shares, act, time_lag, time_series, time)
        current_state_list = flatten(new_state.values())
        act_val = model.predict(current_state_list + [act])
        if act_val > best_val:
            best_act = act
            best_val = act_val
    return best_act

def basic_RL(AMOUNT_INVESTED, model, model_temp, alpha, tau, lag, train_0, time_series, param_string):
    # Initialize the state for alpha = 0, do nothing for first 100 steps so to collect some data;
    # train_0 = 200;
    lag = 20;
    initial_alpha = .5
    df = .99
    X = []
    y = []
    #print "accessed"

    nikkei_0 = AMOUNT_INVESTED * initial_alpha / (current_price_nikkei(time_series, 0))
    ftse_0 = AMOUNT_INVESTED * (1 - initial_alpha) / current_price_ftse(time_series, 0)
    shares_0 = [nikkei_0, ftse_0]
    value_0 = portfolio_value(shares_0, time_series, 0)

    # initialize theta?


    state_RL = pd.DataFrame()
    state_RL['price_nikkei'] = np.zeros(LEN_SERIES)
    state_RL['price_ftse'] = np.zeros(LEN_SERIES)
    state_RL['portfolio_shares_nikkei'] = np.zeros(LEN_SERIES)
    state_RL['portfolio_shares_ftse'] = np.zeros(LEN_SERIES)
    state_RL['value_RL'] = np.zeros(LEN_SERIES)
    state_RL['action_RL'] = np.zeros(LEN_SERIES)
    state_RL['alpha'] = np.ones(LEN_SERIES)

    for time in range(lag):
        #print time
        current_shares = shares_0;
        state_RL['price_nikkei'][time] = current_price_nikkei(time_series, time)
        state_RL['price_ftse'][time] = current_price_ftse(time_series, time)
        state_RL['portfolio_shares_nikkei'][time] = shares_0[0]
        state_RL['portfolio_shares_ftse'][time] = shares_0[1]
        state_RL['value_RL'][time] = portfolio_value(shares_0, time_series, time)
        state_RL['action_RL'][time] = 0;
        state_RL['alpha'][time] = alpha_fun(shares_0, time_series, time)

    state = RL_next_state(shares_0, 0, lag, time_series, lag)
    for time in range(lag, train_0):
        # print time
        current_shares = state["shares"];
        state = RL_next_state(current_shares, 0, lag, time_series, time)
        state_values = state.values()
        x = flatten(state_values)
        X.append(x + [0])
        y.append(portfolio_value(current_shares, time_series, time + 1))
        state_RL['price_nikkei'][time] = current_price_nikkei(time_series, time)
        state_RL['price_ftse'][time] = current_price_ftse(time_series, time)
        state_RL['portfolio_shares_nikkei'][time] = state["shares"][0]
        state_RL['portfolio_shares_ftse'][time] = state["shares"][1]
        state_RL['value_RL'][time] = state["value_RL"]
        state_RL['action_RL'][time] = 0;
        state_RL['alpha'][time] = alpha_fun(state["shares"], time_series, time)
    #print x
    model.fit(X, y)
    theta = model.coef_
    # Train the model using the training sets


    # start reinforcing
    for time in range(lag, LEN_SERIES - 1):
    #for time in range(lag, lag + 5):
        prev_shares = state["shares"];
        greedy_action = RL_greedy_action(state, model, lag, time_series, time)
        state = RL_next_state(prev_shares, greedy_action, lag, time_series, time)
        next_shares = state["shares"]
        state_values = state.values()
        x = flatten(state_values)
        X.append(x + [greedy_action])
        next_greedy_action = RL_greedy_action(state, model, lag, time_series, time + 1)
        Q_value = reward_function(next_shares, current_shares, time_series, time) + df * model.predict(x + [next_greedy_action])
        y.append(Q_value)
        model_temp.fit(X, y)
        theta_temp = model_temp.coef_
        theta = (1 - tau) * theta + tau * theta_temp
        model.coef_ = theta

        state_RL['price_nikkei'][time] = current_price_nikkei(time_series, time)
        state_RL['price_ftse'][time] = current_price_ftse(time_series, time)
        state_RL['portfolio_shares_nikkei'][time] = next_shares[0]
        state_RL['portfolio_shares_ftse'][time] = next_shares[1]
        state_RL['value_RL'][time] = state["value_RL"]
        state_RL['action_RL'][time] = greedy_action;
        state_RL['alpha'][time] = alpha_fun(next_shares, time_series, time)
        if time % 50 == 0:
            print time
    state_RL['price_nikkei'][LEN_SERIES - 1] = current_price_nikkei(time_series, LEN_SERIES - 1)
    state_RL['price_ftse'][LEN_SERIES - 1] = current_price_ftse(time_series, LEN_SERIES - 1)
    state_RL['portfolio_shares_nikkei'][LEN_SERIES - 1] = next_shares[0]
    state_RL['portfolio_shares_ftse'][LEN_SERIES - 1] = next_shares[1]
    state_RL['value_RL'][LEN_SERIES - 1] = state["value_RL"]
    state_RL['action_RL'][LEN_SERIES - 1] = param_string;
    state_RL['alpha'][LEN_SERIES - 1] = alpha_fun(next_shares, time_series, LEN_SERIES - 2)

    # state_RL._metadata.append(param_string)
    return state_RL


### GLOBAL PARAMETERS
INITIAL_ENDOWMENT = 1
AMOUNT_INVESTED = 1
alpha = 0.5
train_0 = 100

# state_nikkei = nikkei_basic(AMOUNT_INVESTED, time_series)
# statistics_nikkei = stat(state_nikkei['value_nikkei'])
#
# state_ftse = ftse_basic(AMOUNT_INVESTED, time_series)
# statistics_ftse = stat(state_ftse['value_ftse'])


tau_list = [.8, .9, .95]
# lag_list = [15, 30]
# model_list = [(linear_model.LinearRegression(), linear_model.LinearRegression(), "linReg"), \
#               (linear_model.Lasso(alpha=0.1), linear_model.Lasso(alpha=0.1), "Lasso0.1"), \
#               (linear_model.Lasso(alpha=0.5), linear_model.Lasso(alpha=0.5), "Lasso0.5"), \
#               (linear_model.Lasso(alpha=0.9), linear_model.Lasso(alpha=0.9), "Lasso0.9")]

model_list = [(linear_model.LinearRegression(), linear_model.LinearRegression(), "linReg"), \
              (linear_model.Lasso(alpha=0.1), linear_model.Lasso(alpha=0.1), "Lasso0.1")]
lag_list = [15]


experiments = []
for i in range(5):
    time_series = [nikkei_severe[i], ftse_severe[i]]
    for model_tup in model_list:
        for lag in lag_list:
            for tau in tau_list:
                exp_model, exp_model_temp, model_name = model_tup[0], model_tup[1], model_tup[2]
                param_string = "__model__" + model_name + "__tau__" + str(tau) + "__lag__" + str(lag) + "__ts__" + str(i)
                temp_param = AMOUNT_INVESTED, exp_model, exp_model_temp, alpha, tau, lag, train_0, time_series, param_string
                experiments.append(temp_param)

# res = basic_RL(*experiments[0])
#print res['value_RL'], res['action_RL']
pool = mp.Pool(processes=mp.cpu_count())
results = [pool.apply_async(basic_RL, args=exp) for exp in experiments]
output = [p.get() for p in results]
# state_nikkei = nikkei_basic(AMOUNT_INVESTED, time_series)
# state_ftse = ftse_basic(AMOUNT_INVESTED, time_series)


import cPickle

cPickle.dump(output, open('Experiments_shock_2_severe.p', 'wb'))
# cPickle.dump(nikkei_mild, open('nikkei_mild.p', 'wb'))
# cPickle.dump(ftse_mild, open('ftse_mild.p', 'wb'))
# cPickle.dump(nikkei_consistent, open('nikkei_consistent.p', 'wb'))
# cPickle.dump(ftse_consistent, open('ftse_consistent.p', 'wb'))
# cPickle.dump(nikkei_severe, open('nikkei_severe.p', 'wb'))
# cPickle.dump(ftse_severe, open('ftse_severe.p', 'wb'))
