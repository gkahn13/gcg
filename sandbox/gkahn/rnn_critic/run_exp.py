import os, time
import argparse
import yaml

from rllab import config
from sandbox.gkahn.rnn_critic.algos.run_rnn_critic import run_rnn_critic
from rllab.misc.instrument import stub, run_experiment_lite

from botocore.exceptions import ClientError

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
parser.add_argument('-mode', type=str, default='local')
parser.add_argument('--confirm_remote', action='store_false')
parser.add_argument('--dry', action='store_true')
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

    while True:
        try:
            # run_rnn_critic(params)
            run_experiment_lite(
                run_rnn_critic,
                snapshot_mode="all",
                exp_name=params['exp_name'],
                exp_prefix=params['exp_prefix'],
                variant=params,
                use_gpu=True,
                use_cloudpickle=True,
                mode=args.mode,
                sync_s3_pkl=True,
                aws_config={
                    'image_id': 'ami-8a2b0aea',
                    'security_groups': ['rllab-sg'],
                    'key_name': 'id_rsa',
                    'instance_type': 'g2.2xlarge'},
                confirm_remote=args.confirm_remote,
                dry=args.dry
            )
            break
        except ClientError as e:
            print('ClientError: {0}\nSleep for a bit and try again'.format(e))
            time.sleep(30)
        # except Exception as e:
        #     if str(e) != 'madeit':
        #         input('Experiment {0} failed: {1}'.format(exp, str(e)))
        #     break
