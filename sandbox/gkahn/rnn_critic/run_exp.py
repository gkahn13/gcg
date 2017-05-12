import os
import argparse
import yaml

from rllab import config
from rllab.misc.instrument import run_experiment_lite

from sandbox.gkahn.rnn_critic.examples.run_rnn_critic import run_rnn_critic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exps', nargs='+')
    parser.add_argument('-docker_image', type=str, default=None)
    args = parser.parse_args()

    for exp in args.exps:
        yaml_path = os.path.abspath('examples/yamls/{0}.yaml'.format(exp))
        assert(os.path.exists(yaml_path))
        with open(yaml_path, 'r') as f:
            params = yaml.load(f)
        with open(yaml_path, 'r') as f:
            params_txt = ''.join(f.readlines())
        params['txt'] = params_txt

        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
        config.USE_TF = True

        kwargs = dict()
        if args.docker_image is not None:
            kwargs['mode'] = 'local_docker'
            kwargs['docker_image'] = args.docker_image
            kwargs['docker_args'] = ' --name {0} '.format(params['exp_name'])
            kwargs['post_commands'] = [' sleep 1 ']

        try:
            run_experiment_lite(
                run_rnn_critic,
                snapshot_mode="all",
                exp_name=params['exp_name'],
                exp_prefix=params['exp_prefix'],
                use_gpu=True,
                variant=params,
                **kwargs
            )
        except:
            print('Experiment {0} failed!'.format(exp))
