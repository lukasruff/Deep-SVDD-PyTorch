import json
import numpy as np


base_path = '/Users/lukasruff/Repos/Deep-SVDD-PyTorch/log/mnist/test/mnist/soft_deepSVDD'
n_exps = 3
n_seeds = 3

exps = range(n_exps)
seeds = range(1, n_seeds)

for exp in exps:

    exp_folder = str(exp) + 'vsall'
    aucs = np.zeros(n_seeds, dtype=np.float32)

    for seed in seeds:

        seed_folder = 'seed_' + str(seed)
        file_name = 'results.json'
        file_path = base_path + '/' + exp_folder + '/' + seed_folder + '/' + file_name

        with open(file_path, 'r') as fp:
            results = json.load(fp)

        aucs[seed - 1] = results['test_auc']

    mean = np.mean(aucs[aucs > 0])
    std = np.std(aucs[aucs > 0])

    # Write results
    log_file = '{}/result.txt'.format(base_path)
    log = open(log_file, 'a')
    log.write('Experiment: {}\n'.format(exp_folder))
    log.write('Test Set AUC [mean]: {} %\n'.format(round(float(mean * 100), 4)))
    log.write('Test Set AUC [std]: {} %\n'.format(round(float(std * 100), 4)))
    log.write('\n')

log.write('\n')
log.close()
