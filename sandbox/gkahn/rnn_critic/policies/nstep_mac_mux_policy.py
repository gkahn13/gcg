from rllab.core.serializable import Serializable

from sandbox.gkahn.rnn_critic.policies.mac_mux_policy import MACMuxPolicy
from sandbox.gkahn.rnn_critic.policies.nstep_mac_policy import NstepMACPolicy

class NstepMACMuxPolicy(MACMuxPolicy, NstepMACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        MACMuxPolicy.__init__(self, **kwargs)

        assert(self._H == 1)
