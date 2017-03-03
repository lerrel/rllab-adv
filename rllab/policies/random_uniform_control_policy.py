from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.distributions.delta import Delta
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides


class RandomUniformControlPolicy(Policy, Serializable):
    def __init__(
            self,
            env_spec,
            is_protagonist=True,
    ):
        Serializable.quick_init(self, locals())
        if is_protagonist==True: cur_action_space = env_spec.pro_action_space;
        else: cur_action_space = env_spec.adv_action_space
        action_dim = cur_action_space.flat_dim
        self._action_space = cur_action_space
        #assert isinstance(self.action_space, Box)
        #super(UniformControlPolicy, self).__init__(env_spec=env_spec)
        self._cached_params={}

    @overrides
    def get_action(self, observation):
        return self._action_space.sample(), dict()

    def get_params_internal(self, **tags):
        return []

    def get_param_values(self, **tags):
        return []

    def get_actions(self, observations):
        all_samples = []
        for obs in observations:
            sample_action = self._action_space.sample()
            all_samples.append(sample_action)
        all_samples = np.array(all_samples)
        return all_samples, dict()

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        pass

    @property
    def distribution(self):
        # Just a placeholder
        return Delta()
