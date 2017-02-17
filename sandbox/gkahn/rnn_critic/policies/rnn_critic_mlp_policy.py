from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides

import numpy as np
import tensorflow as tf


class RNNCriticMLPPolicy(Policy, LayersPowered, Serializable):
    def __init__(self,
                 name,
                 env_spec,
                 H,
                 hidden_sizes=(40, 40),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=lambda x: x):
        """
        :param env_spec: A spec for the mdp.
        :param H: horizon
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity used for output layer
        """
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            network = MLP(
                input_shape=(env_spec.observation_space.flat_dim + H * env_spec.action_space.flat_dim,),
                output_dim=H,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name='network',
            )

            self._l_rewards = network.output_layer
            self._l_obs_actions = network.input_layer
            self._f_rewards = tensor_utils.compile_function(
                [network.input_layer.input_var],
                L.get_output(network.output_layer)
            )

            super(RNNCriticMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [network.output_layer])

        self._H = H

    @property
    def vectorized(self):
        return True

    @overrides
    def get_action(self, observation):
        ### random sampler
        N = 1000 # TODO: pass in as param
        action_lower, action_upper = self._env_spec.action_space.bounds
        actions = np.random.uniform(action_lower.tolist(), action_upper.tolist(),
                                    size=(N, self._H, self._env_spec.action_space.flat_dim))
        actions = actions.reshape(N, self._H * self._env_spec.action_space.flat_dim)

        flat_obs = self.observation_space.flatten(observation)
        network_input = np.hstack(([flat_obs] * len(actions), actions))
        rewards = self._f_rewards(network_input)

        cost = -rewards.sum(axis=1) # TODO: sum of discounted rewards
        chosen_action = actions[cost.argmin()][:self._env_spec.action_space.flat_dim]

        return chosen_action, dict()
