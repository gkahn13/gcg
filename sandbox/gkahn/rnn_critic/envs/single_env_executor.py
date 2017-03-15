
class SingleEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        assert(len(envs) == 1)
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.max_path_length = max_path_length

    def step(self, action_n):
        assert(len(action_n) == 1)
        obs, reward, done, env_info = self.envs[0].step(action_n[0])
        if done:
            obs = self.envs[0].reset()
        obs = [obs]
        rewards = [reward]
        dones = [done]
        env_infos = [env_info]

        return obs, rewards, dones, env_infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
