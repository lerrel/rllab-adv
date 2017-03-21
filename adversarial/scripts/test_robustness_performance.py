import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from test import test_const_adv,test_const_noise_adv, test_rand_adv, test_rand_step_adv, test_step_adv, test_learnt_adv
from IPython import embed
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

def test_folder(test_type, folder_name, env_name, adv_fraction=1.0, n_traj=5, percentile=True):
    L = os.listdir(folder_name)
    file_name_summary = []
    test_rew_summary = []
    test_rew_std_summary = []
    for i,l in reversed(list(enumerate(L))):
        if 'temp' in l or '.p' not in l:
            del(L[i])
        else:
            res_D = pickle.load(open(os.path.join(folder_name,l),'rb'))
            file_name_summary.append(os.path.join(folder_name,l))
            P = res_D['pro_policy']
            env = normalize(GymEnv(env_name, adv_fraction))
            N = np.zeros(n_traj)
            for i in range(n_traj):
                N[i] = test_type(env, P, 1000, 1)
            M =N.mean(); V=N.std()
            test_rew_summary.append(M)
            test_rew_std_summary.append(V)
    if percentile:
        vals = np.sort(test_rew_summary)
        x = np.linspace(0,100,vals.shape[0])
        plt.plot(x,vals,linewidth=2.0)
        plt.grid(True)
        axes = plt.gca();axes.set_ylim([0,6000]);
        #plt.show()

    return test_rew_summary, test_rew_std_summary, file_name_summary

def test_noise_folder(folder_name, env_name, adv_fraction=1.0, n_traj=5, percentile=True, std_noise=0.001):
    L = os.listdir(folder_name)
    file_name_summary = []
    test_rew_summary = []
    test_rew_std_summary = []
    for i,l in reversed(list(enumerate(L))):
        if 'temp' in l or '.p' not in l:
            del(L[i])
        else:
            res_D = pickle.load(open(os.path.join(folder_name,l),'rb'))
            file_name_summary.append(os.path.join(folder_name,l))
            P = res_D['pro_policy']
            env = normalize(GymEnv(env_name, adv_fraction))
            N = np.zeros(n_traj)
            for i in range(n_traj):
                N[i] = test_const_noise_adv(env, P, 1000, 1, std_noise=std_noise)
            M =N.mean(); V=N.std()
            test_rew_summary.append(M)
            test_rew_std_summary.append(V)
    if percentile:
        vals = np.sort(test_rew_summary)
        x = np.linspace(0,100,vals.shape[0])
        plt.plot(x,vals,linewidth=2.0)
        plt.grid(True)
        axes = plt.gca();axes.set_ylim([0,6000]);
        #plt.show()

    return test_rew_summary, test_rew_std_summary, file_name_summary

def test_noise_file(file_name, env_name, adv_fraction=1.0, n_traj=5, std_noise=np.linspace(0,0.1,11)):
    test_rew_summary = []
    test_rew_std_summary = []
    test_noise_summary = []
    res_D = pickle.load(open(file_name,'rb'))
    P = res_D['pro_policy']
    env = normalize(GymEnv(env_name, adv_fraction))
    for stdn in std_noise:
        N = np.zeros(n_traj)
        for i in range(n_traj):
            N[i] = test_const_noise_adv(env, P, 1000, 1, std_noise=stdn)
        M =N.mean(); V=N.std()
        test_noise_summary.append(stdn)
        test_rew_summary.append(M)
        test_rew_std_summary.append(V)

    return test_rew_summary, test_rew_std_summary, test_noise_summary



def test_adversary_folder(folder_name, percentile=True):
    L = os.listdir(folder_name)
    file_name_summary = []
    test_rew_summary = []
    test_rew_std_summary = []
    for i,l in reversed(list(enumerate(L))):
        if 'temp' in l or '.p' not in l:
            del(L[i])
        else:
            res_D = pickle.load(open(os.path.join(folder_name,l),'rb'))
            file_name_summary.append(os.path.join(folder_name,l))
            M = res_D['adv_test'][0][-1]
            test_rew_summary.append(M)
    if percentile:
        vals = np.sort(test_rew_summary)
        x = np.linspace(0,100,vals.shape[0])
        plt.plot(x,vals,linewidth=2.0)
        plt.grid(True)
        #axes = plt.gca();axes.set_ylim([-150,400]);
        axes = plt.gca();axes.set_ylim([0,6000]);
        #plt.show()

    return test_rew_summary, test_rew_std_summary, file_name_summary

def plot_best_folder(folder_name):
    L = os.listdir(folder_name)
    file_name_summary = []
    test_rew_summary = []
    test_rew_std_summary = []
    for i,l in reversed(list(enumerate(L))):
        if 'temp' in l or '.p' not in l:
            del(L[i])
        else:
            res_D = pickle.load(open(os.path.join(folder_name,l),'rb'))
            file_name_summary.append(os.path.join(folder_name,l))
            M = res_D['zero_test'][0][-1]
            test_rew_summary.append(M)
    id_f = np.array(test_rew_summary).argmax()
    fname = file_name_summary[id_f]
    res_D = pickle.load(open(fname,'rb'))
    M = res_D['zero_test'][0]
    P = res_D['pro_policy']
    x = [i for i in range(len(M))]
    plt.plot(x,M,linewidth=2.0)
    plt.grid(True)
    #axes = plt.gca();axes.set_ylim([0,400]);
    axes = plt.gca();axes.set_ylim([0,6000]);

    return M[-1],np.array(M[-100:]).std()*np.sqrt(5)

