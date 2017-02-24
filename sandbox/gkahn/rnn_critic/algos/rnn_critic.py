from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler

class RNNCritic(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 max_path_length,
                 exploration_strategy,
                 total_steps,
                 learn_after_n_steps,
                 train_every_n_steps,
                 save_every_n_steps,
                 update_target_after_n_steps,
                 update_target_every_n_steps,
                 update_preprocess_every_n_steps,
                 log_every_n_steps,
                 batch_size,
                 n_envs=1,
                 replay_pool_size=int(1e6),
                 render=False):
        """
        :param env: Environment
        :param policy: RNNCriticPolicy
        :param max_path_length: maximum length of a single rollout
        :param exploration_strategy: how actions are modified
        :param total_steps: how many steps to take in total
        :param learn_after_n_steps: before this many steps, take random actions
        :param train_every_n_steps: train policy every n steps
        :param save_every_n_steps: save policy every n steps
        :param update_target_after_n_steps:
        :param update_target_every_n_steps: update target every n steps
        :param update_preprocess_every_n_steps
        :param log_every_n_steps: log every n steps
        :param batch_size: batch size per gradient step
        :param render: show env
        """
        assert(learn_after_n_steps % n_envs == 0)
        assert(train_every_n_steps % n_envs == 0)
        assert(save_every_n_steps % n_envs == 0)
        assert(update_target_every_n_steps % n_envs == 0)
        assert(update_preprocess_every_n_steps % n_envs == 0)

        self._env = env
        self._policy = policy
        self._max_path_length = max_path_length
        self._total_steps = total_steps
        self._learn_after_n_steps = learn_after_n_steps
        self._train_every_n_steps = train_every_n_steps
        self._save_every_n_steps = save_every_n_steps
        self._update_target_after_n_steps = update_target_after_n_steps
        self._update_target_every_n_steps = update_target_every_n_steps
        self._update_preprocess_every_n_steps = update_preprocess_every_n_steps
        self._log_every_n_steps = log_every_n_steps
        self._batch_size = batch_size

        policy.set_exploration_strategy(exploration_strategy)

        self._sampler = RNNCriticSampler(
            policy=policy,
            env=env,
            n_envs=n_envs,
            replay_pool_size=replay_pool_size,
            max_path_length=max_path_length,
        )

    @overrides
    def train(self):
        save_itr = 0
        target_updated = False
        for step in range(0, self._total_steps, self._sampler.n_envs):
            self._sampler.step()

            if step > self._learn_after_n_steps:
                ### training step
                if step % self._train_every_n_steps == 0:
                    self._policy.train_step(*self._sampler.sample(self._batch_size),
                                            use_target=target_updated)

                ### update target network
                if step > self._update_target_after_n_steps and step % self._update_target_every_n_steps == 0:
                    # logger.log('Updating target network...')
                    self._policy.update_target()
                    target_updated = True

                ### update preprocess
                if step % self._update_preprocess_every_n_steps == 0:
                    self._policy.update_preprocess(self._sampler.statistics)

                if step % self._log_every_n_steps == 0:
                    logger.log('step %.3e' % step)
                    logger.record_tabular('Step', step)
                    self._sampler.log()
                    self._policy.log()
                    logger.dump_tabular(with_prefix=False)

                ### save model
                if step % self._save_every_n_steps == 0:
                    # logger.log('Saving...')
                    with self._policy.session.as_default(), self._policy.session.graph.as_default():
                        itr_params = dict(
                            itr=save_itr,
                            policy=self._policy,
                            env=self._env,
                            rollouts=self._sampler.get_recent_paths()
                        )
                        logger.save_itr_params(save_itr, itr_params)
                    save_itr += 1
