import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from test import test_const_adv, test_rand_adv, test_rand_step_adv, test_step_adv, test_learnt_adv
from IPython import embed
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

def test_mass_rob(test_type, file_name, env_name, mass_fractions=np.linspace(0.5,1.5,11),mass_bodies=[b'pole'], adv_fraction=1.0, n_traj=5):
    mass_vals = []
    test_rew_summary = []
    test_rew_std_summary = []
    print(file_name)
    res_D = pickle.load(open(file_name,'rb'))
    P = res_D['pro_policy']
    for mf in mass_fractions:
        env = normalize(GymEnv(env_name, 1.0))
        me = np.array(env.wrapped_env.env.model.body_mass)
        mass_ind = env.wrapped_env.env.model.body_names.index(mass_bodies[0])
        me[mass_ind,0] = me[mass_ind,0]*mf
        env.wrapped_env.env.model.body_mass = me

        N = np.zeros(n_traj)
        for i in range(n_traj):
            N[i] = test_type(env, P, 1000, 1)
        M =N.mean(); V=N.std()
        mass_vals.append(me[mass_ind])
        test_rew_summary.append(M)
        test_rew_std_summary.append(V)

    return test_rew_summary, test_rew_std_summary, mass_vals

def test_mass_rob_folder(test_type, folder_name, env_name, mass_fractions=np.linspace(0.5,1.5,11),mass_bodies=[b'pole'], adv_fraction=1.0, n_traj=5):
    L = os.listdir(folder_name)
    file_name_summary = []
    all_t_rew = []
    for i,l in reversed(list(enumerate(L))):
        if 'temp' in l or '.p' not in l:
            del(L[i])
        else:
            file_name = os.path.join(folder_name,l)
            file_name_summary.append(file_name)
            t_rew,_,m_v = test_mass_rob(test_type, file_name, env_name, mass_fractions=mass_fractions,mass_bodies=mass_bodies, adv_fraction=adv_fraction, n_traj=n_traj)
            all_t_rew.append(t_rew)
    return all_t_rew, file_name_summary, m_v
