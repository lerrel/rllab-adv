from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.distributions.delta import Delta
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides
import random

class StepControlPolicy(Policy, Serializable):
    def __init__(
            self,
            env_spec,
            characteristic_length,
            step_size,
            is_random_mag=True,
            is_protagonist=True,
    ):
        Serializable.quick_init(self, locals())
        if is_protagonist==True: cur_action_space = env_spec.pro_action_space;
        else: cur_action_space = env_spec.adv_action_space
        action_dim = cur_action_space.flat_dim
        self._action_space = cur_action_space
        self._prob_step = 1.0/characteristic_length
        self._in_step_looper = 0
        self._step_size = step_size
        self._step_val = None
        self._is_random_mag = is_random_mag
        self._cached_params={}

    @overrides
    def get_action(self, observation):
        if self._in_step_looper==0:
            if random.random() > self._prob_step:
                return self._action_space.sample()*0.0, dict()
            else:
                if self._is_random_mag == True:
                    self._step_val = self._action_space.sample()
                else:
                    self._step_val = self._action_space.bounds[1]
                self._in_step_looper = (self._in_step_looper + 1)%self._step_size
                return self._step_val, dict()
        else:
            self._in_step_looper = (self._in_step_looper + 1)%self._step_size
            return self._step_val, dict()

    def get_params_internal(self, **tags):
        return []

    def get_param_values(self, **tags):
        return []

    def get_actions(self, observations):
        raise NotImplementedError

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        pass

    @property
    def distribution(self):
        # Just a placeholder
        return Delta()
