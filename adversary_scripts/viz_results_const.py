import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import scipy
import argparse

## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('r', type=str, help='filepath to results')
parser.add_argument('f', type=int, default=1, help='0 for no filtering. 1 for filtering')

args = parser.parse_args()
savename = args.r
if_filtering = bool(args.f)

res_D = pickle.load(open(savename,'rb'))
const_test_rew_summary = res_D['zero_test']

all_patches = []

con_rew = np.array(const_test_rew_summary)
mean_con = con_rew.mean(0)
std_con = con_rew.std(0)
if if_filtering==True:
    mean_window_size = 15
    mean_order = 3
    std_window_size = 45
    std_order = 2
    mean_con = scipy.signal.savgol_filter(mean_con, mean_window_size, mean_order)
    std_con = scipy.signal.savgol_filter(std_con, std_window_size, std_order)
x = [i for i in range(len(mean_con))]
plt.plot(x,mean_con,color=(0.5,0.1,0.1), linewidth=2.0)
plt.fill_between(x, mean_con-std_con, mean_con+std_con,color=(0.5,0.1,0.1), alpha=0.5)
all_patches.append(mpatches.Patch(color=(0.5,0.1,0.1), label='zero_test_rew_summary'))

for l in con_rew:
    plt.plot(x,l,color=(0.1,0.5,0.1), linewidth=2.0)

plt.legend(handles=all_patches)
axes = plt.gca()
axes.set_ylim([-500,6000])
plt.title(savename)
plt.show()
