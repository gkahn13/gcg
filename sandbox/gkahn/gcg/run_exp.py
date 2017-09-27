import os, time
import argparse
import yaml

from rllab import config
from sandbox.gkahn.gcg.algos.gcg import run_gcg
from rllab.misc.instrument import stub, run_experiment_lite

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
parser.add_argument('-mode', type=str, default='local')
parser.add_argument('--confirm_remote', action='store_false')
parser.add_argument('--dry', action='store_true')
parser.add_argument('-region', type=str, choices=('us-west-1', 'us-west-2', 'us-east-1', 'us-east-2'), default='us-west-1')
args = parser.parse_args()

for exp in args.exps:
    yaml_path = os.path.abspath('yamls/{0}.yaml'.format(exp))
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    with open(yaml_path, 'r') as f:
        params_txt = ''.join(f.readlines())
    params['txt'] = params_txt

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])
    config.USE_TF = True

    run_experiment_lite(
        run_gcg,
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix'],
        variant=params,
        use_gpu=True,
        use_cloudpickle=True,
    )
