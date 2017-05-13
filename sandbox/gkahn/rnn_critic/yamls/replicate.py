import itertools
import shutil

SOURCE_EXP_NAME = 'pend'
SOURCE_EXP_NUMBER = 200

def exp_name(name, number):
    return '{0}{1:03d}'.format(name, number)

def merge_dicts(dict_list):
    d_merge = {}
    for d in dict_list:
        for k, v in d.items():
            d_merge[k] = v
    return d_merge

keys = ['V_class', 'V_N', 'V_H', 'V_test_H', 'V_target_H', 'V_softmax', 'V_exp_lambda', 'V_retrace_lambda', 'V_use_target', 'V_share_weights']

# CDQN
# Feedforward MAC
# Model-based only MAC
# MAC
type_replacements = [{'V_class': 'CDQNPolicy', 'V_N': N, 'V_H': 1, 'V_test_H': 1, 'V_target_H': 1, 'V_use_target': 'True'} for N in [1, 5, 10, 20]] + \
                    [{'V_class': 'FeedforwardMACPolicy', 'V_N': H, 'V_H': H, 'V_test_H': H, 'V_target_H': H, 'V_use_target': 'True'} for H in [1, 5, 10, 20]] + \
                    [{'V_class': 'MACPolicy', 'V_N': H, 'V_H': H, 'V_test_H': H, 'V_target_H': H, 'V_use_target': 'False', 'V_share_weights': True} for H in [1, 5, 10, 20]] + \
                    [{'V_class': 'MACPolicy', 'V_N': H, 'V_H': H, 'V_test_H': H, 'V_target_H': H, 'V_use_target': 'True', 'V_share_weights': False} for H in [1, 5, 10, 20]] + \
                    [{'V_class': 'MACPolicy', 'V_N': H, 'V_H': H, 'V_test_H': H, 'V_target_H': H, 'V_use_target': 'True', 'V_share_weights': True} for H in [1, 5, 10, 20]]

softmax_replacements = [
    {'V_softmax': 'mean', 'V_retrace_lambda': ''}, # mean
    {'V_softmax': 'final', 'V_retrace_lambda': ''}, # final
    {'V_softmax': 'exponential', 'V_exp_lambda': 0.25, 'V_retrace_lambda': ''}, # exponential
    {'V_softmax': 'exponential', 'V_exp_lambda': 0.5, 'V_retrace_lambda': ''}, # exponential
    {'V_softmax': 'exponential', 'V_exp_lambda': 0.75, 'V_retrace_lambda': ''}, # exponential
    {'V_softmax': 'exponential', 'V_exp_lambda': 0.9, 'V_retrace_lambda': ''}, # exponential
    {'V_softmax': 'final', 'V_retrace_lambda': 1}, # IS
]


seed_replacements = [
    {'S0': 1},
    {'S0': 2},
    {'S0': 3},
]

all_replacements = (type_replacements, softmax_replacements, seed_replacements)

replacements = [
    merge_dicts([{exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER): exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER+i+1)}] + list(dict_list))
    for i, dict_list in enumerate(itertools.product(*all_replacements))
]

for r in replacements:
    with open(exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER)+'.yaml', 'r') as f:
        text = f.read()

    for k, v in r.items():
        assert(k in text)
        text = text.replace(k, str(v))

    print_str = ' '.join(['{0}: {1},'.format(k, r[k] if k in r else '') for k in keys])
    print('{0}\t{1}'.format(r[exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER)], print_str))
#    with open(r[exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER)]+'.yaml', 'w') as f:
#        f.write(text)

    
