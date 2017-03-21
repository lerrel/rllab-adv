## Testing ##
from rllab.sampler.utils import rollout
from rllab.policies.constant_control_policy import ConstantControlPolicy
from rllab.policies.random_uniform_control_policy import RandomUniformControlPolicy
from rllab.policies.step_control_policy import StepControlPolicy

def test_const_adv(env, protag_policy, path_length=100, n_traj=5, render=False):
    const_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val = 0.0
    )
    paths = []
    sum_rewards = 0.0
    for _ in range(n_traj):
        path = rollout(env, protag_policy, path_length, adv_agent=const_adv_policy, animated=render, test=True)
        sum_rewards += path['rewards'].sum()
        paths.append(path)
    avg_rewards = sum_rewards/n_traj
    return avg_rewards

def test_rand_adv(env, protag_policy, path_length=100, n_traj=5, render=False):
    adv_policy = RandomUniformControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
    )
    paths = []
    sum_rewards = 0.0
    for _ in range(n_traj):
        path = rollout(env, protag_policy, path_length, adv_agent=adv_policy, animated=render, test=True)
        sum_rewards += path['rewards'].sum()
        paths.append(path)
    avg_rewards = sum_rewards/n_traj
    return avg_rewards

def test_rand_step_adv(env, protag_policy, path_length=100, n_traj=5, render=False):
    paths = []
    sum_rewards = 0.0
    characteristic_length = path_length/5
    step_size = path_length/10
    for _ in range(n_traj):
        adv_policy = StepControlPolicy(
            env_spec=env.spec,
            characteristic_length=characteristic_length,
            step_size=step_size,
            is_random_mag=True,
            is_protagonist=False,
        )
        path = rollout(env, protag_policy, path_length, adv_agent=adv_policy, animated=render, test=True)
        sum_rewards += path['rewards'].sum()
        paths.append(path)
    avg_rewards = sum_rewards/n_traj
    return avg_rewards

def test_step_adv(env, protag_policy, path_length=100, n_traj=5, render=False):
    paths = []
    sum_rewards = 0.0
    characteristic_length = path_length/5
    step_size = path_length/10
    for _ in range(n_traj):
        adv_policy = StepControlPolicy(
            env_spec=env.spec,
            characteristic_length=characteristic_length,
            step_size=step_size,
            is_random_mag=False,
            is_protagonist=False,
        )
        path = rollout(env, protag_policy, path_length, adv_agent=adv_policy, animated=render, test=True)
        sum_rewards += path['rewards'].sum()
        paths.append(path)
    avg_rewards = sum_rewards/n_traj
    return avg_rewards


def test_learnt_adv(env, protag_policy, adv_policy, path_length=100, n_traj=5, render=False):
    paths = []
    sum_rewards = 0.0
    for _ in range(n_traj):
        path = rollout(env, protag_policy, path_length, adv_agent=adv_policy, animated=render, test=True)
        sum_rewards += path['rewards'].sum()
        paths.append(path)
    avg_rewards = sum_rewards/n_traj
    return avg_rewards
