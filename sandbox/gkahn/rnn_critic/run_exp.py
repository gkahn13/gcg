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
parser.add_argument('-region', type=str, choices=('us-west-1', 'us-west-2', 'us-east-1', 'us-east-2'), default='us-west-1')
args = parser.parse_args()

# stub(globals())

aws_config = {
    'security_groups': ['rllab-sg'],
    'key_name': 'id_rsa',
    'instance_type': 'p2.xlarge',
    'spot_price': '0.5',
}
if args.region == 'us-west-1':
    aws_config.update({
        'image_id': 'ami-8a2b0aea',
        'region_name': 'us-west-1',
        'security_group_ids': ['sg-88f4d7ef']
    })
elif args.region == 'us-west-2':
    aws_config.update({
        'image_id': 'ami-f4101a8d',
        'region_name': 'us-west-2',
        'security_group_ids': ['sg-1dd6bc66']
    })
elif args.region == 'us-east-1':
    aws_config.update({
        'security_groups': [],
        'key_name': 'rllab-us-east-1',
        'image_id': 'ami-d7a99dac',
        'region_name': 'us-east-1',
        'subnet_id': 'subnet-941746a8', # TODO
        'security_group_ids': ['sg-9e9e00e0']
    })
elif args.region == 'us-east-2':
    aws_config.update({
        'security_groups': [],
        # 'key_name': 'rllab-us-east-2',
        'image_id': 'ami-8b86a6ee',
        'region_name': 'us-east-2',
        'subnet_id': 'subnet-24ad045f',  # TODO
        'security_group_ids': ['sg-ee707e87']
    })
else:
    raise NotImplementedError

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
                aws_config=aws_config,
                confirm_remote=args.confirm_remote,
                dry=args.dry
            )
            time.sleep(1)
            break
        except ClientError as e:
            print('ClientError: {0}\nSleep for a bit and try again'.format(e))
            time.sleep(30)
        # except Exception as e:
        #     if str(e) != 'madeit':
        #         input('Experiment {0} failed: {1}'.format(exp, str(e)))
        #     break
