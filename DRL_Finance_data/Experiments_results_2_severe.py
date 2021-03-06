import os
import cPickle
import matplotlib.pyplot as plt
plt.rc('legend',**{'fontsize':10})
my_path = os.path.abspath(__file__)

pkl_file =  open('Experiments_shock_2_severe.p', 'rb')
data = cPickle.load(pkl_file)




# pkl_file =  open('nikkei.p', 'rb')
# nikkei = cPickle.load(pkl_file)
pkl_file =  open('ftse_severe.p', 'rb')
ftse = cPickle.load(pkl_file)

pkl_file =  open('nikkei_severe.p', 'rb')
nikkei = cPickle.load(pkl_file)








for test in data:
    u = test['action_RL'][len(test['action_RL'])-1]
    index_time_series = int(u[-1])
    strategy = test['value_RL']
    ftse_val = ftse[index_time_series]/ftse[index_time_series][0]
    nikkei_val = nikkei[index_time_series]/nikkei[index_time_series][0]
    X = range(len(strategy))


    plt.plot(X, strategy, color="red", linewidth=1.0, linestyle="-")

    plt.plot(X, ftse_val, color="green", linewidth=1.0, linestyle="-", label = "ftse")

    plt.plot(X, nikkei_val, color="blue", linewidth=1.0, linestyle="-", label = "nikkei")
    plt.ylim(0.1,2.25)
    plt.legend(loc='upper left')
    plt.title(u)
    plt.savefig('/home/lorenzo/PycharmProjects/DRL_Finance/' + u + '.png')
    plt.close()
