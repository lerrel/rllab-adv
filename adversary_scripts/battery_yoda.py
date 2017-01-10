import sys,os


env_names = ['HopperAdv-v1', 'HalfCheetahAdv-v1', 'Walker2dAdv-v1']
adv_fractions = [0.1, 0.25, 0.5, 1.0, 2.0, 10.0]


for e_n in env_names:
    for a_f in adv_fractions:
        run_script = 'python train_trpo_var_adversary.py --env {} --adv_name adv --n_exps 3 --n_itr 500 --layer_size 64 64 --batch_size 25000 --if_render 0 --adv_fraction {}'.format(e_n, a_f)


