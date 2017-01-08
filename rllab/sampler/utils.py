import numpy as np
from rllab.misc import tensor_utils
import time

def rollout(env, pro_agent, max_path_length=np.inf, animated=False, speedup=1, adv_agent=None):
    observations = []
    pro_actions = []
    if adv_agent: adv_actions = []
    rewards = []
    pro_agent_infos = []
    if adv_agent: adv_agent_infos = []
    env_infos = []
    o = env.reset()
    pro_agent.reset()
    if adv_agent: adv_agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        pro_a, pro_agent_info = pro_agent.get_action(o)
        if adv_agent:
            adv_a, adv_agent_info = adv_agent.get_action(o)
            class temp_action(object): pro=None; adv=None;
            #from IPython import embed;embed()
            cum_a = temp_action()
            cum_a.pro = pro_a; cum_a.adv = adv_a;
            #print(type(adv_agent))
            #print('adversary_action = {}'.format(adv_a))
            next_o, r, d, env_info = env.step(cum_a)
            pro_actions.append(env.pro_action_space.flatten(pro_a))
            adv_actions.append(env.adv_action_space.flatten(adv_a))
            adv_agent_infos.append(adv_agent_info)
        else:
            next_o, r, d, env_info = env.step(pro_a)
            pro_actions.append(env.action_space.flatten(pro_a))

        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        pro_agent_infos.append(pro_agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    #if animated:
    #    return
    if adv_agent:
        return dict(
            observations=tensor_utils.stack_tensor_list(observations),
            pro_actions=tensor_utils.stack_tensor_list(pro_actions),
            adv_actions=tensor_utils.stack_tensor_list(adv_actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            pro_agent_infos=tensor_utils.stack_tensor_dict_list(pro_agent_infos),
            adv_agent_infos=tensor_utils.stack_tensor_dict_list(adv_agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )
    else:
        return dict(
            observations=tensor_utils.stack_tensor_list(observations),
            actions=tensor_utils.stack_tensor_list(pro_actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(pro_agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )
