import os
import cPickle
import matplotlib.pyplot as plt
my_path = os.path.abspath(__file__)

pkl_file =  open('Experiments_no_shock.p', 'rb')
data = cPickle.load(pkl_file)
pkl_file =  open('nikkei.p', 'rb')
nikkei = cPickle.load(pkl_file)
pkl_file =  open('ftse.p', 'rb')
ftse = cPickle.load(pkl_file)

for test in data:
    u = test['action_RL'][len(test['action_RL'])-1]
    strategy = test['value_RL']
    ftse_val = ftse['value_ftse']
    nikkei_val = nikkei['value_nikkei']
    X = range(len(strategy))


    plt.plot(X, strategy, color="red", linewidth=1.0, linestyle="-")

    plt.plot(X, ftse_val, color="green", linewidth=1.0, linestyle="-", label = "ftse")

    plt.plot(X, nikkei_val, color="blue", linewidth=1.0, linestyle="-", label = "nikkei")
    plt.ylim(0.5,2.25)
    plt.legend(loc='upper left')
    plt.title(u)
    plt.savefig('/home/lorenzo/PycharmProjects/DRL_Finance/' + u + '.png')
    plt.close()


    # Save figure using 72 dots per inch
    # plt.savefig("exercice_2.png", dpi=72)

    # Show result on screen


