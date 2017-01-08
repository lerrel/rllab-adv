from rllab.core.serializable import Serializable
from rllab.spaces.base import Space


class EnvSpec(Serializable):

    def __init__(
            self,
            observation_space,
            pro_action_space,
            adv_action_space):
        """
        :type observation_space: Space
        :type action_space: Space
        """
        Serializable.quick_init(self, locals())
        self._observation_space = observation_space
        self._pro_action_space = pro_action_space
        self._adv_action_space = adv_action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def pro_action_space(self):
        return self._pro_action_space

    @property
    def adv_action_space(self):
        return self._adv_action_space
