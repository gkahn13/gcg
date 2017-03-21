import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.multiaction_combinedcost_mlp_policy import MultiactionCombinedcostMLPPolicy

class MultiactionSeparatedcostMLPPolicy(MultiactionCombinedcostMLPPolicy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        Serializable.quick_init(self, locals())

        MultiactionCombinedcostMLPPolicy.__init__(self, hidden_layers, activation, **kwargs)

        assert(self._N > 1)
        assert(self._H > 1)
        assert(self._N == self._H)
        assert(self._cost_type == 'separated')

    ##################
    ### Properties ###
    ##################

    @property
    def N_output(self):
        return self._N + 1 # b/c output value too

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_cost(self, tf_rewards_ph, tf_actions_ph, tf_rewards, tf_target_rewards, tf_target_mask_ph):
        # for training, len(tf_obs_ph) == len(tf_actions_ph)
        # but len(tf_actions_target_ph) == N * len(tf_action_ph),
        # so need to be selective about what to take the max over
        tf_target_values = self._graph_calculate_values(tf_target_rewards)
        batch_size = tf.shape(tf_rewards_ph)[0]
        tf_target_values_flat = tf.reshape(tf_target_values, (batch_size, -1))
        tf_target_values_max = tf.reduce_max(tf_target_values_flat, reduction_indices=1, keep_dims=True)

        if self._use_target:
            tf_rewards_ph_concat = tf.concat(1, [tf_rewards_ph, tf_target_values_max])
        else:
            tf_rewards_ph_concat = tf.concat(1, [tf_rewards_ph, 0])
            tf_rewards[:, -1] = 0.
        mse = tf.reduce_mean(self._graph_calculate_values(tf.square(tf_rewards - tf_rewards_ph_concat)))

        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            weight_decay = 0
        cost = mse + weight_decay
        return cost, mse
