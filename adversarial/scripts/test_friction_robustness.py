import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import scipy, scipy.signal
import argparse
import os
import numpy as np
from test import test_const_adv
from IPython import embed
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

def get_robustness(policy, env_name, fric_fractions=[1.0], fric_bodies = [b'foot'], mass_fractions=[1.0], mass_bodies = [b'torso'], num_evals=5):
    P = policy
    #M=[];V=[];
    M = np.zeros((len(fric_fractions), len(mass_fractions)))
    V=np.zeros((len(fric_fractions), len(mass_fractions)))
    fis=np.zeros((len(fric_fractions), len(mass_fractions)))
    mis=np.zeros((len(fric_fractions), len(mass_fractions)))
    for fi,f in enumerate(fric_fractions):
        for mi,m in enumerate(mass_fractions):
            print('{}/{}'.format((fi*len(mass_fractions))+mi,len(mass_fractions)*len(fric_fractions)))
            env = normalize(GymEnv(env_name, 1.0));
            e = np.array(env.wrapped_env.env.model.geom_friction)
            fric_ind = env.wrapped_env.env.model.body_names.index(fric_bodies[0])
            e[fric_ind,0] = e[fric_ind,0]*f
            env.wrapped_env.env.model.geom_friction = e
            me = np.array(env.wrapped_env.env.model.body_mass)
            mass_ind = env.wrapped_env.env.model.body_names.index(mass_bodies[0])
            me[mass_ind,0] = me[mass_ind,0]*m
            env.wrapped_env.env.model.body_mass = me
            t = []
            for _ in range(num_evals):
                t.append(test_const_adv(env, P, 1000, 1))
            t=np.array(t)
            M[fi,mi] = t.mean()
            V[fi,mi] = t.std()
            fis[fi,mi] = e[fric_ind,0]
            mis[fi,mi] = me[mass_ind,0]
    return M,V,fis,mis
