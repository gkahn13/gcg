from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.discrete_dqn_policy import DiscreteDQNPolicy

class NstepDiscreteDQNPolicy(DiscreteDQNPolicy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 conv_hidden_layers=None,
                 conv_kernels=None,
                 conv_strides=None,
                 conv_activation=None,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        Serializable.quick_init(self, locals())

        DiscreteDQNPolicy.__init__(self,
                                   hidden_layers,
                                   activation,
                                   conv_hidden_layers=None,
                                   conv_kernels=None,
                                   conv_strides=None,
                                   conv_activation=None,
                                   **kwargs)

        # assert(self._N > 1)
        assert(self._H == 1)
        assert(self._cost_type == 'combined')
