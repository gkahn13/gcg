import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.dqn_policy import DQNPolicy

class NstepDQNPolicy(DQNPolicy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        Serializable.quick_init(self, locals())

        DQNPolicy.__init__(self,
                           hidden_layers=hidden_layers,
                           activation=activation,
                           **kwargs)

        assert(self._N > 1)
        assert(self._H == 1)
