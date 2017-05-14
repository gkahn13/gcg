import itertools
import shutil

SOURCE_EXP_NAME = 'hc'
SOURCE_EXP_NUMBER = 70

def exp_name(name, number):
    return '{0}{1:03d}'.format(name, number)

def merge_dicts(dict_list):
    d_merge = {}
    for d in dict_list:
        for k, v in d.items():
            d_merge[k] = v
    return d_merge

keys = ['V_class', 'V_N', 'V_H', 'G0', 'G1']

type_replacements = [
    {'V_class': 'CDQNPolicy', 'V_N': 5,  'V_H': 1}, # 71, 72, 73
    {'V_class': 'MACPolicy',  'V_N': 5,  'V_H': 5}, # 74, 75, 76
    {'V_class': 'CDQNPolicy', 'V_N': 10, 'V_H': 1}, # 77, 78, 79
    {'V_class': 'MACPolicy',  'V_N': 10, 'V_H': 10},# 80, 81, 82
]

seed_replacements = [
    {'S0': 1, 'G0': 0, 'G1': 0.4},
    {'S0': 2, 'G0': 1, 'G1': 0.4},
    {'S0': 3, 'G0': 2, 'G1': 0.4},
]

all_replacements = (type_replacements, seed_replacements)

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
    with open(r[exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER)]+'.yaml', 'w') as f:
        f.write(text)



