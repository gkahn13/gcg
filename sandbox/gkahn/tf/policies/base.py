from sandbox.gkahn.tf.core.parameterized import Parameterized

class Policy(Parameterized):
    def __init__(self, env_spec, sess=None):
        Parameterized.__init__(self, sess=sess)
        self._env_spec = env_spec

    # Should be implemented by all policies

    def get_action(self, observation):
        raise NotImplementedError

    def get_actions(self, observations):
        raise NotImplementedError

    def reset(self, dones=None):
        pass

    @property
    def vectorized(self):
        """
        Indicates whether the policy is vectorized. If True, it should implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return False

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def env_spec(self):
        return self._env_spec

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the policy's state when taking an action.
        :return:
        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """
        Return keys and shapes for the information related to the policy's state when taking an action.
        :return:
        """
        return list()

    def terminate(self):
        """
        Clean up operation
        """
        pass
