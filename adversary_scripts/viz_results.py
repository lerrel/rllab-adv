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
rand_test_rew_summary = res_D['rand_test']
adv_test_rew_summary = res_D['adv_test']

## PRINTING ##
con_rew = np.array(const_test_rew_summary)
ran_rew = np.array(rand_test_rew_summary)
adv_rew = np.array(adv_test_rew_summary)
mean_con = con_rew.mean(0)
std_con = con_rew.std(0)
mean_ran = ran_rew.mean(0)
std_ran = ran_rew.std(0)
mean_adv = adv_rew.mean(0)
std_adv = adv_rew.std(0)

if if_filtering==True:
    mean_window_size = 15
    mean_order = 3
    std_window_size = 45
    std_order = 2
    mean_con = scipy.signal.savgol_filter(mean_con, mean_window_size, mean_order)
    mean_ran = scipy.signal.savgol_filter(mean_ran, mean_window_size, mean_order)
    mean_adv = scipy.signal.savgol_filter(mean_adv, mean_window_size, mean_order)
    std_con = scipy.signal.savgol_filter(std_con, std_window_size, std_order)
    std_ran = scipy.signal.savgol_filter(std_ran, std_window_size, std_order)
    std_adv = scipy.signal.savgol_filter(std_adv, std_window_size, std_order)

x = [i for i in range(len(mean_con))]
plt.plot(x,mean_con,color=(1.,0.,0.), linewidth=2.0)
plt.fill_between(x, mean_con-std_con, mean_con+std_con,color=(0.5,0.1,0.1), alpha=0.5)
plt.plot(x,mean_ran,color=(0.,1.,0.), linewidth=2.0)
plt.fill_between(x, mean_ran-std_ran, mean_ran+std_ran,color=(0.1,0.5,0.1), alpha=0.5)
plt.plot(x,mean_adv,color=(0.,0.,1.), linewidth=2.0)
plt.fill_between(x, mean_adv-std_adv, mean_adv+std_adv,color=(0.1,0.1,0.5), alpha=0.5)

red_patch = mpatches.Patch(color=(1.,0.,0.), label='Testing with 0 adversary')
green_patch = mpatches.Patch(color=(0.,1.,0.), label='Testing with random adversary')
blue_patch = mpatches.Patch(color=(0.,0.,1.), label='Testing with learnt adversary')
plt.legend(handles=[red_patch,green_patch,blue_patch])
axes = plt.gca()
axes.set_ylim([-500,4000])
plt.title(savename)
plt.show()
