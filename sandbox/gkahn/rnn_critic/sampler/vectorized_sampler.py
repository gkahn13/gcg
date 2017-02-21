import pickle
import itertools
import time
import numpy as np

from rllab.sampler.base import Sampler
from rllab.sampler.stateful_pool import ProgBarCounter
from rllab.misc import tensor_utils

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

class RNNCriticVectorizedSampler(Sampler):

    def __init__(self, env, policy, n_envs, max_path_length, rollouts_per_sample):
        self._env = env
        self._policy = policy
        self._n_envs = n_envs
        self._max_path_length = max_path_length
        self._rollouts_per_sample = rollouts_per_sample

    def start_worker(self):
        if getattr(self._env, 'vectorized', False):
            self._vec_env = self._env.vec_env_executor(n_envs=self._n_envs, max_path_length=self._max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self._env)) for _ in range(self._n_envs)]
            self._vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self._max_path_length
            )

    def obtain_samples(self):
        paths = []
        n_samples = 0
        obses = self._vec_env.reset()
        dones = np.asarray([True] * self._vec_env.num_envs)
        running_paths = [None] * self._vec_env.num_envs

        pbar = ProgBarCounter(self._max_path_length * self._rollouts_per_sample)
        policy_time = 0
        env_time = 0
        process_time = 0

        while n_samples < self._rollouts_per_sample:
            t = time.time()
            self._policy.reset(dones)
            actions, agent_infos = self._policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self._vec_env.step(actions)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self._vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self._vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths.append(dict(
                        observations=self._env.spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self._env.spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += 1
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger_stats = {
            'PolicyExecTime': policy_time,
            'EnvExecTime': env_time,
            'ProcessExecTime': process_time
        }

        return paths[:self._rollouts_per_sample], logger_stats

    def process_samples(self, itr, paths):
        pass

    def shutdown_worker(self):
        self._vec_env.terminate()
