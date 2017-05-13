import os
import argparse
import yaml

from rllab import config
from sandbox.gkahn.rnn_critic.algos.run_rnn_critic import run_rnn_critic
from rllab.misc.instrument import stub, run_experiment_lite

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
parser.add_argument('-docker_image', type=str, default=None)
args = parser.parse_args()

# stub(globals())

for exp in args.exps:
    yaml_path = os.path.abspath('yamls/{0}.yaml'.format(exp))
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

    # try:
    run_experiment_lite(
        run_rnn_critic,
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix'],
        variant=params,
        use_gpu=True,
        use_cloudpickle=True,
        mode='local',
        # mode='ec2_mujoco',
        # sync_s3_pkl=True,
        # aws_config={
        #     'image_id': 'ami-f399bf93',
        #     'security_groups': ['rllab-sg'],
        #     'key_name': 'id_rsa',
        #     'instance_type': 'g2.2xlarge'},
        # dry=True,
        **kwargs
    )
    # except Exception as e:
    #     print('Experiment {0} failed: {1}'.format(exp, str(e)))
