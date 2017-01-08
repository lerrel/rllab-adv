from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.random_uniform_control_policy import RandomUniformControlPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
from IPython import embed
import matplotlib.pyplot as plt
import rllab.misc.logger as logger
import numpy as np
from test import test_const_adv, test_rand_adv, test_learnt_adv
import pickle
import argparse

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

const_test_rew_summary = []
rand_test_rew_summary = []
adv_test_rew_summary = []
save_prefix = 'env-{}_{}_Exp{}_Itr{}_BS{}'.format(env_name, adv_name, n_exps, n_itr, batch_size)
save_dir = 'results'
fig_dir = 'figs'
save_name = save_dir+'/'+save_prefix+'.p'
fig_name = fig_dir+'/'+save_prefix+'.png'

for ne in range(n_exps):
    ## Environment definition ##
    env = normalize(GymEnv(env_name))

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
    parallel_sampler.initialize(1)
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
    adv_testing_rews = []
    adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))
    #embed()
    for ni in range(n_itr):
        logger.log('\n\n\n####expNO{} global itr# {}####\n\n\n'.format(ne,ni))
        pro_algo.train()
        pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
        adv_algo.train()
        adv_rews += adv_algo.rews; all_rews += adv_algo.rews;
        logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))
        const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
        rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))
        if ni%afterRender==0 and ifRender==True:
            test_const_adv(env, pro_policy, path_length=path_length, n_traj=1, render=True);

    ## Shutting down the optimizer ##
    pro_algo.shutdown_worker()
    adv_algo.shutdown_worker()
    const_test_rew_summary.append(const_testing_rews)
    rand_test_rew_summary.append(rand_testing_rews)
    adv_test_rew_summary.append(adv_testing_rews)

## PRINTING ##
import matplotlib.pyplot as plt
con_rew = np.array(const_test_rew_summary)
ran_rew = np.array(rand_test_rew_summary)
adv_rew = np.array(adv_test_rew_summary)
mean_con = con_rew.mean(0)
std_con = con_rew.std(0)
mean_ran = ran_rew.mean(0)
std_ran = ran_rew.std(0)
mean_adv = adv_rew.mean(0)
std_adv = adv_rew.std(0)

x = [i for i in range(len(mean_con))]
plt.plot(x,mean_con,color=(1.,0.,0.), linewidth=2.0)
plt.fill_between(x, mean_con-std_con, mean_con+std_con,color=(0.5,0.1,0.1), alpha=0.5)
plt.plot(x,mean_ran,color=(0.,1.,0.), linewidth=2.0)
plt.fill_between(x, mean_ran-std_ran, mean_ran+std_ran,color=(0.1,0.5,0.1), alpha=0.5)
plt.plot(x,mean_adv,color=(0.,0.,1.), linewidth=2.0)
plt.fill_between(x, mean_adv-std_adv, mean_adv+std_adv,color=(0.1,0.1,0.5), alpha=0.5)
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color=(1.,0.,0.), label='Testing with 0 adversary')
green_patch = mpatches.Patch(color=(0.,1.,0.), label='Testing with random adversary')
blue_patch = mpatches.Patch(color=(0.,0.,1.), label='Testing with learnt adversary')
plt.legend(handles=[red_patch,green_patch,blue_patch])
axes = plt.gca()
axes.set_ylim([0,4000])
plt.title(save_name)
pickle.dump({'zero_test': const_test_rew_summary,
             'rand_test': rand_test_rew_summary,
             'adv_test': adv_test_rew_summary}, open(save_name,'wb'))
plt.savefig(fig_name)
#plt.show()
embed()
