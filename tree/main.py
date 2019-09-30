from coverage import Coverage
import pandas as pd
import json

# node_list = [[[13, 2, 2], 8, [4, 4], [2, 22]],
#              [2, 13], [[2, 2, 2, 2, 2], 10], [[2, 2, 4], [2, 6]],
#              [4, 4, [8, 2]]]
alpha_true = [0.4, -0.3, 0.3]
base_rate_true = 0.1
n_cov = 3

node_list = [[[13, 2, 1], 8, [4, 4], [1, 22]],
             [2, 13], [[2, 2, 2, 2, 1], 10], [[2, 2, 4], [2, 6]],
             [4, 4, [8, 2]]]
data = pd.read_csv('../data/test/sim_data_tiny_1569777314.csv')

file_path = '../data/test/20190929_1569777314_coverage/'
hold_out = ['1_5', '1_4_2']
hold_out_leaves = [('1_1_1_1', .4), ('1_2_2', .4)]
cvr = Coverage(data, node_list, n_cov, file_path, hold_out, hold_out_leaves)

iota_dist_all, alpha_dist_all, u_dist_all, sample_idx = cvr.compare_methods(n_draws=10, n_runs=1,
                                                                            plot=True, n_samples=5)

print(sample_idx)

with open(file_path+'dist.json', 'w') as f:
    json.dump([iota_dist_all, alpha_dist_all, u_dist_all], f)
