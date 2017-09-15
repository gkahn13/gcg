import os
import yaml
import argparse
import joblib

from rllab.misc.ext import set_seed
import rllab.misc.logger as logger

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler

from sandbox.gkahn.rnn_critic.envs.env_utils import create_env

class EvalExp(object):
    def __init__(self, folder, num_rollouts):
        """
        :param kwargs: holds random extra properties
        """
        self._folder = folder
        self._num_rollouts = num_rollouts

        ### load data
        # logger.log('AnalyzeRNNCritic: Loading data')
        self.name = os.path.basename(self._folder)
        # logger.log('AnalyzeRNNCritic: params_file: {0}'.format(self._params_file))
        with open(self._params_file, 'r') as f:
            self.params = yaml.load(f)

        self.env = create_env(self.params['alg']['env'])

    #############
    ### Files ###
    #############

    def _itr_file(self, itr):
        return os.path.join(self._folder, 'itr_{0:d}.pkl'.format(itr))

    @property
    def _params_file(self):
        yamls = [fname for fname in os.listdir(self._folder) if os.path.splitext(fname)[-1] == '.yaml' and os.path.basename(self._folder) in fname]
        assert(len(yamls) == 1)
        return os.path.join(self._folder, yamls[0])

    def save_eval_rollouts(self, itr, rollouts):
        fname = os.path.join(self._folder, 'itr_{0:d}_exp_eval.pkl'.format(itr))
        joblib.dump({'rollouts': rollouts}, fname, compress=3)

    ####################
    ### Data loading ###
    ####################

    def _load_itr_policy(self, itr):
        d = joblib.load(self._itr_file(itr))
        policy = d['policy']
        return policy

    def eval_policy(self, itr, gpu_device=None, gpu_frac=None):
        if itr == -1:
            itr = 0
            while os.path.exists(self._itr_file(itr)):
                itr += 1
            itr -= 1

        if self.params['seed'] is not None:
            set_seed(self.params['seed'])

        if gpu_device is None:
            gpu_device = self.params['policy']['gpu_device']
        if gpu_frac is None:
            gpu_frac = self.params['policy']['gpu_frac']
        sess, graph = MACPolicy.create_session_and_graph(gpu_device=gpu_device, gpu_frac=gpu_frac)
        with graph.as_default(), sess.as_default():
            policy = self._load_itr_policy(itr)

            logger.log('Evaluating policy for itr {0}'.format(itr))
            n_envs = 1
            if 'max_path_length' in self.params['alg']:
                max_path_length = self.params['alg']['max_path_length']
            else:
                max_path_length = self.env.horizon

            sampler = RNNCriticSampler(
                policy=policy,
                env=self.env,
                n_envs=n_envs,
                replay_pool_size=int(1e4),
                max_path_length=max_path_length,
                save_rollouts=True,
                sampling_method=self.params['alg']['replay_pool_sampling']
            )
            rollouts = []
            step = 0
            logger.log('Starting rollout {0}'.format(len(rollouts)))
            while len(rollouts) < self._num_rollouts:
                sampler.step(step)
                step += n_envs
                new_rollouts = sampler.get_recent_paths()
                if len(new_rollouts) > 0:
                    while True:
                        yn = raw_input('Keep rollout?')
                        if yn[0] == 'y':
                            logger.log('Keeping rollout')
                            logger.log('Starting rollout {0}'.format(len(rollouts)))
                            rollouts += new_rollouts
                            self.save_eval_rollouts(itr, rollouts)
                            break
                        elif yn[0] == 'n':
                            logger.log('Not keeping rollout')
                            logger.log('Redoing rollout {0}'.format(len(rollouts)))
                            break
                        else:
                            logger.log('Invalid response')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('numrollouts', type=int)
    args = parser.parse_args()

    eval_exp = EvalExp(args.folder, args.numrollouts)
    eval_exp.eval_policy(-1, gpu_device=0, gpu_frac=0.4)
