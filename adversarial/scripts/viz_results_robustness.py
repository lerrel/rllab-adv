import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import scipy, scipy.signal
import argparse
import os
from IPython import embed
from test_friction_robustness import get_robustness

## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('r', type=str, help='folder to results')
parser.add_argument('env_name', type=str, default=1, help='Environment name')
parser.add_argument('g', type=int, default=1, help='0 for mean')

args = parser.parse_args()
savename = args.r
env_name = args.env_name

L = os.listdir(savename)
const_test_rew_summary = []
pol_nam = []
pol_set = []
pol_M =[]
pol_V =[]
for i,l in reversed(list(enumerate(L))):
    if 'temp' in l or '.p' not in l:
        del(L[i])
    else:
        res_D = pickle.load(open(os.path.join(savename,l),'rb'))
        const_test_rew_summary.append(res_D['zero_test'][0])
        print(os.path.join(savename,l))
        print(res_D['zero_test'][0][-1])
        pol_nam.append(os.path.join(savename,l))
        pol_set.append(res_D['pro_policy'])
        M,V,_,mis = get_robustness(res_D['pro_policy'],env_name, fric_fractions=[1.0],mass_fractions=np.linspace(0.5,1.5,21), num_evals=25)
        pol_M.append(M); pol_V.append(V)
all_patches = []
pol_M = np.array(pol_M)
pol_V = np.array(pol_V)
for i,l in enumerate(pol_M):
    rand_c = (np.random.rand(),np.random.rand(),np.random.rand())
    plt.plot(mis[0,:],l[0],color=rand_c, linewidth=2.0)
    plt.fill_between(mis[0,:], l[0]-pol_V[i][0], l[0]+pol_V[i][0],color=rand_c,alpha=0.5)

if args.g==0:
    plt.plot(mis[0,:],pol_M.mean(0)[0],color=(0,0,0), linewidth=5.0)
    plt.fill_between(mis[0,:], pol_M.mean(0)[0]-pol_V.mean(0)[0], pol_M.mean(0)[0]+pol_V.mean(0)[0],color=(0,0,0),alpha=0.5)

#plt.fill_between(x, mean_con-std_con, mean_con+std_con,color=(0.5,0.1,0.1), alpha=0.5)
axes = plt.gca()
axes.set_ylim([0,4000])
plt.title(savename)
axes.yaxis.tick_right()
plt.grid(True)
plt.show()
embed()
#from IPython import embed;embed()
