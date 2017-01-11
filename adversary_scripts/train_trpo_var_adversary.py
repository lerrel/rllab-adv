from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.random_uniform_control_policy import RandomUniformControlPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
from rllab.policies.step_control_policy import StepControlPolicy
from IPython import embed
import matplotlib.pyplot as plt
import rllab.misc.logger as logger
import numpy as np
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv
import pickle
import argparse
import os
import gym

## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True, help='Name of adversarial environment')
parser.add_argument('--adv_name', type=str, required=True, help='adv if training with adversary, no_adv if training without adversary')
parser.add_argument('--path_length', type=int, default=1000, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[100,100,100], help='layer definition')
parser.add_argument('--if_render', type=int, default=1, help='Should we animate at all?')
parser.add_argument('--after_render', type=int, default=100, help='After how many to animate')
parser.add_argument('--n_exps', type=int, default=3, help='')
parser.add_argument('--n_itr', type=int, default=100, help='')
parser.add_argument('--n_pro_itr', type=int, default=1, help='')
parser.add_argument('--n_adv_itr', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=4000, help='')
parser.add_argument('--filename_append', type=str, default='', help='stuff to append to save filename')
parser.add_argument('--save_every', type=int, default=100, help='')
parser.add_argument('--n_process', type=int, default=16, help='Number of threads for sampling environment')
parser.add_argument('--adv_fraction', type=float, default=1.0, help='fraction of maximum adversarial force to be applied')

args = parser.parse_args()

## Number of experiments to run ##
env_name = args.env
adv_name = args.adv_name
path_length = args.path_length
layer_size = tuple(args.layer_size)
ifRender = bool(args.if_render)
afterRender = args.after_render
n_exps = args.n_exps
n_itr = args.n_itr
n_pro_itr = args.n_pro_itr
n_adv_itr = args.n_adv_itr
batch_size = args.batch_size
save_every = args.save_every
n_process = args.n_process
adv_fraction = args.adv_fraction

const_test_rew_summary = []
rand_test_rew_summary = []
step_test_rew_summary = []
rand_step_test_rew_summary = []
adv_test_rew_summary = []
save_prefix = 'env-{}_{}_Exp{}_Itr{}_BS{}_Adv{}'.format(env_name, adv_name, n_exps, n_itr, batch_size, adv_fraction)
save_dir = os.environ['HOME']+'/results/variable_adversary'
fig_dir = 'figs'
save_name = save_dir+'/'+save_prefix+'.p'
fig_name = fig_dir+'/'+save_prefix+'.png'

for ne in range(n_exps):
    ## Environment definition ##
    env = normalize(GymEnv(env_name, adv_fraction))
    ## Protagonist policy definition ##
    pro_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=True
    )
    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Zero Adversary for the protagonist training ##
    zero_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val = 0.0
    )

    ## Adversary policy definition ##
    adv_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=False
    )
    adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Optimizer for the Protagonist ##
    from rllab.sampler import parallel_sampler
    parallel_sampler.initialize(n_process)
    if adv_name=='adv':
        pro_algo = TRPO(
            env=env,
            pro_policy=pro_policy,
            adv_policy=adv_policy,
            pro_baseline=pro_baseline,
            adv_baseline=adv_baseline,
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_pro_itr,
            discount=0.99,
            step_size=0.01,
            is_protagonist=True
        )
    elif adv_name=='no_adv':
        pro_algo = TRPO(
            env=env,
            pro_policy=pro_policy,
            adv_policy=zero_adv_policy,
            pro_baseline=pro_baseline,
            adv_baseline=adv_baseline,
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_pro_itr,
            discount=0.99,
            step_size=0.01,
            is_protagonist=True
        )

    ## Optimizer for the Adversary ##
    adv_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_adv_itr,
        discount=0.99,
        step_size=0.01,
        is_protagonist=False,
        scope='adversary_optim'
    )

    ## Joint optimization ##
    if ifRender==True: test_const_adv(env, pro_policy, path_length=path_length, n_traj = 1, render=True)
    pro_rews = []
    adv_rews = []
    all_rews = []
    const_testing_rews = []
    const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
    rand_testing_rews = []
    rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
    step_testing_rews = []
    step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
    rand_step_testing_rews = []
    rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
    adv_testing_rews = []
    adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))
    #embed()
    for ni in range(n_itr):
        logger.log('\n\n\n####expNO{}_{} global itr# {}####\n\n\n'.format(ne,adv_name,ni))
        pro_algo.train()
        pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
        adv_algo.train()
        adv_rews += adv_algo.rews; all_rews += adv_algo.rews;
        logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))
        const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
        rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
        rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
        adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))
        if ni%afterRender==0 and ifRender==True:
            test_const_adv(env, pro_policy, path_length=path_length, n_traj=1, render=True);
        if ni!=0 and ni%save_every==0:
            ## SAVING INFO ##
            pickle.dump({'args': args,
                         'pro_policy': pro_policy,
                         'adv_policy': adv_policy,
                         'zero_test': const_test_rew_summary,
                         'rand_test': rand_test_rew_summary,
                         'step_test': step_test_rew_summary,
                         'rand_step_test': rand_step_test_rew_summary,
                         'iter_save': ni,
                         'exp_save': ne,
                         'adv_test': adv_test_rew_summary}, open(save_name+'.temp','wb'))

    ## Shutting down the optimizer ##
    pro_algo.shutdown_worker()
    adv_algo.shutdown_worker()
    const_test_rew_summary.append(const_testing_rews)
    rand_test_rew_summary.append(rand_testing_rews)
    step_test_rew_summary.append(step_testing_rews)
    rand_step_test_rew_summary.append(rand_step_testing_rews)
    adv_test_rew_summary.append(adv_testing_rews)

## SAVING INFO ##
pickle.dump({'args': args,
             'pro_policy': pro_policy,
             'adv_policy': adv_policy,
             'zero_test': const_test_rew_summary,
             'rand_test': rand_test_rew_summary,
             'step_test': step_test_rew_summary,
             'rand_step_test': rand_step_test_rew_summary,
             'adv_test': adv_test_rew_summary}, open(save_name,'wb'))

logger.log('\n\n\n#### DONE ####\n\n\n')
