import numpy as np
import json
import pandas as pd
import os


def simulate_tree_data(base_rate, alpha_true, meas_std, node_list, gamma_list,
                       data_per_leaf_test=0, add_cov=True, noise_density='gaussian',
                       center_u=True, add_noise=True, seed=617, file_path=None, cov_noise=True):
    n_cov = len(alpha_true)
    np.random.seed(seed)
    data = []
    X_all = []

    def recurse(node_list, path, level, cov_mean):
        if type(node_list) is int:
            n_leaf = node_list
            X = np.zeros((n_leaf + data_per_leaf_test,n_cov))
            if add_cov:
                np.random.seed(int(hash(path)%5000))  # need to fix hashseed also to fix covariate values
                if cov_noise:
                    X += np.random.randn(n_leaf + data_per_leaf_test, n_cov)/(level+1) + cov_mean
                else:
                    X += cov_mean
                X_all.append(X)
            for i in range(n_leaf + data_per_leaf_test):
                row = {}
                for j in range(n_cov):
                    row['cov'+str(j+1)] = X[i, j]
                row['level_'+str(level)] = path
                row['leaf'] = path
                row['node'] = path
                row['true_val'] = base_rate*np.exp(np.dot(X[i, :], alpha_true))
                row['n_data_in_leaf'] = n_leaf
                if i < n_leaf:
                    row['hold_out'] = False
                else:
                    row['hold_out'] = True
                data.append(row)
        else:
            K = len(node_list)
            u = np.random.randn(K)*np.sqrt(gamma_list[level])
            if center_u:
                u = u - np.mean(u)
            cov_means = np.linspace(cov_mean - 0.5/(level+1), cov_mean + 0.5/(level+1), K)
            #print(path, cov_means)
            for k in range(K):
                if node_list:
                    recurse(node_list[k], path+'_'+str(k+1), level+1, cov_means[k])
                    for row in data:
                        if 'level_'+str(level+1) in row and row['level_'+str(level+1)] == path + '_' + str(k+1):
                            row['level_'+str(level)] = path
                            row['true_val'] *= np.exp(u[k])
                            row['level_'+str(level)+'_u'] = u[k]

    recurse(node_list, '1', 0, 1.)

    if add_noise:
        for row in data:
            row['meas_std'] = row['true_val']*meas_std
            if row['meas_std'] <= 0.:
                print(row)
            if noise_density == 'gaussian':
                row['meas_val'] = row['true_val'] + np.random.randn()*row['meas_std']
            elif noise_density == 'log_gaussian':
                row['meas_val'] = row['true_val'] + np.random.lognormal()*row['meas_std']
    else:
        for row in data:
            row['meas_std'] = row['true_val']*meas_std
            row['meas_val'] = row['true_val']

    if file_path:
        print('save files at', file_path)
        pd.DataFrame(data).to_csv(file_path+'_data.csv')

    return data, np.vstack(X_all)
