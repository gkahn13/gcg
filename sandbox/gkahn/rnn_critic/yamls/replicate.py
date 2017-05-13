import itertools
import shutil

SOURCE_EXP_NAME = 'walker'
SOURCE_EXP_NUMBER = 000

def exp_name(name, number):
    return '{0}{1:03d}'.format(name, number)

#replacements = [
#    {
#        exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER): exp_name(SOURCE_EXP_NAME, 55 + i),
#        'seed: 1': 'seed: {0}'.format(i + 1),
#        'N: 1': 'N: 12',
#        'H: 1': 'H: 12',
#        'gpu_device: 0': 'gpu_device: 1'
#    } for i in range(3)
#]

def merge_dicts(dict_list):
    d_merge = {}
    for d in dict_list:
        for k, v in d.items():
            d_merge[k] = v
    return d_merge

type_replacements = [
    {'V_N': 5, 'V_H': 1,   'V_class': 'CDQNPolicy', 'G0': 0, 'G1': 0.2, 'S0': 1}, # ahsoka 001
    {'V_N': 5, 'V_H': 1,   'V_class': 'CDQNPolicy', 'G0': 0, 'G1': 0.2, 'S0': 2}, # ahsoka 002
    {'V_N': 10, 'V_H': 1,  'V_class': 'CDQNPolicy', 'G0': 1, 'G1': 0.4, 'S0': 1}, # newton4 003
    {'V_N': 10, 'V_H': 1,  'V_class': 'CDQNPolicy', 'G0': 1, 'G1': 0.4, 'S0': 2}, # newton4 004
    {'V_N': 5, 'V_H': 5,   'V_class': 'MACPolicy',  'G0': 0, 'G1': 0.4, 'S0': 1}, # newton1 005
    {'V_N': 5, 'V_H': 5,   'V_class': 'MACPolicy',  'G0': 1, 'G1': 0.2, 'S0': 2}, # ahsoka 006
    {'V_N': 10, 'V_H': 10, 'V_class': 'MACPolicy',  'G0': 2, 'G1': 0.4, 'S0': 1}, # newton4 007
    {'V_N': 10, 'V_H': 10, 'V_class': 'MACPolicy',  'G0': 3, 'G1': 0.4, 'S0': 2}, # newton4 008
]


#seed_replacements = [
#    {'S0': 1},
#    {'S0': 2},
#    {'S0': 3},
#]

all_replacements = (type_replacements, )

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
    
    with open(r[exp_name(SOURCE_EXP_NAME, SOURCE_EXP_NUMBER)]+'.yaml', 'w') as f:
        f.write(text)

    
