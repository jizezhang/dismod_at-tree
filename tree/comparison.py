import numpy as np
from mixetree import MixETree
import time
import matplotlib.pyplot as plt


def compare(data_no_noise, node_list, alpha_true, iota_true, holdout_node, file_path, cv=0):
    data = data_no_noise
    n_data = data.shape[0]
    n_cov = len(alpha_true)
    if cv > 0:
        data['meas_std'] = data['true_val'] * cv
        data['meas_val'] = data['true_val'] + np.random.randn(n_data) * data['meas_std']

    rel_err_all = []
    alphas_fit_all = []
    elapsed_times = []

    # 1. fit given true alpha and iota
    mtr = MixETree(data, node_list, n_cov, file_path+'fit_given_true_alpha/', holdout_node)

    true_val_obs = mtr.df[mtr.df['hold_out'] == False]['true_val']
    true_val_mis = mtr.df[mtr.df['hold_out'] == True]['true_val']

    t0 = time.time()
    mtr.fitLowLevel({'iota': ([iota_true], [0], 'uniform'), 'alpha': (alpha_true, [0] * n_cov, 'uniform')},
                    zero_sum=True, use_lambda=False)
    #print('fit with true alpha, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit_all.append(alpha_true)

    est_val_obs = mtr.df[mtr.df['hold_out'] == False]['est_val_ns']
    est_val_mis = mtr.df[mtr.df['hold_out'] == True]['est_val_ns']

    rel_err_all.append((est_val_obs - true_val_obs) / np.abs(true_val_obs))
    rel_err_all.append((est_val_mis - true_val_mis) / np.abs(true_val_mis))

    # 2. Direct bottom level fit
    mtr = MixETree(data, node_list, n_cov, file_path+'direct_bottom_fit/', holdout_node)

    true_val_obs = mtr.df[mtr.df['hold_out'] == False]['true_val']
    true_val_mis = mtr.df[mtr.df['hold_out'] == True]['true_val']
    mtr.reset()
    t0 = time.time()
    mtr.fitLowLevel(zero_sum=True)
    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))
    mtr.fitLowLevel({'alpha': (alphas_fit, [0] * n_cov, 'uniform')}, zero_sum=True)
    #print('direct bottom level fit, elapsed ', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit_all.append(np.round(alphas_fit, 3))

    est_val_obs = mtr.df[mtr.df['hold_out'] == False]['est_val_ns']
    est_val_mis = mtr.df[mtr.df['hold_out'] == True]['est_val_ns']

    rel_err_all.append((est_val_obs - true_val_obs) / np.abs(true_val_obs))
    rel_err_all.append((est_val_mis - true_val_mis) / np.abs(true_val_mis))

    # 3. Top-bottom fit
    mtr = MixETree(data, node_list, n_cov, file_path+'top_bottom/', holdout_node)

    true_val_obs = mtr.df[mtr.df['hold_out'] == False]['true_val']
    true_val_mis = mtr.df[mtr.df['hold_out'] == True]['true_val']

    t0 = time.time()
    mtr.fitNoSim(zero_sum=True, use_lambda=True)
    #print('top-bottom fit, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        alphas_fit.append(mtr.alpha_est['1']['cov' + str(i + 1)])

    alphas_fit_all.append(np.round(alphas_fit, 3))

    est_val_obs = mtr.df[mtr.df['hold_out'] == False]['est_val_ns']
    est_val_mis = mtr.df[mtr.df['hold_out'] == True]['est_val_ns']

    rel_err_all.append((est_val_obs - true_val_obs) / np.abs(true_val_obs))
    rel_err_all.append((est_val_mis - true_val_mis) / np.abs(true_val_mis))

    # 4. Cascade fit, no lambda
    mtr = MixETree(data, node_list, n_cov, file_path+'cascade/', holdout_node)

    true_val_obs = mtr.df[mtr.df['hold_out'] == False]['true_val']
    true_val_mis = mtr.df[mtr.df['hold_out'] == True]['true_val']

    t0 = time.time()
    mtr.fitSim(n_sim=10, use_lambda=False, zero_sum=True)
    #print('cascade fit, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))

    alphas_fit_all.append(np.round(alphas_fit, 3))

    est_val_obs = mtr.df[mtr.df['hold_out'] == False]['est_val_s']
    est_val_mis = mtr.df[mtr.df['hold_out'] == True]['est_val_s']

    rel_err_all.append((est_val_obs - true_val_obs) / np.abs(true_val_obs))
    rel_err_all.append((est_val_mis - true_val_mis) / np.abs(true_val_mis))

    # 5. Cascade fit, with lambda
    mtr = MixETree(data, node_list, n_cov, file_path+'cascade_lambda/', holdout_node)

    true_val_obs = mtr.df[mtr.df['hold_out'] == False]['true_val']
    true_val_mis = mtr.df[mtr.df['hold_out'] == True]['true_val']

    t0 = time.time()
    mtr.fitSim(n_sim=10, use_lambda=True, zero_sum=True)
    #print('cascade fit with lambda, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))

    alphas_fit_all.append(np.round(alphas_fit, 3))

    est_val_obs = mtr.df[mtr.df['hold_out'] == False]['est_val_s']
    est_val_mis = mtr.df[mtr.df['hold_out'] == True]['est_val_s']

    rel_err_all.append((est_val_obs - true_val_obs) / np.abs(true_val_obs))
    rel_err_all.append((est_val_mis - true_val_mis) / np.abs(true_val_mis))

    return rel_err_all, alphas_fit_all, elapsed_times


def plot_result(rel_err_all, alphas_fit_all, gamma_list, cv=0, legend_loc=['upper left', 'upper left']):
    # ---- plotting -------
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    labels = ['fit given true alpha', 'direct bottom fit', 'top bottom', 'cascade', 'cascade with lambda']
    colors = ['pink', 'lightblue', 'lightgreen', 'lightgray', 'yellow']

    # plot results on observed data
    bp1 = axs[0].boxplot(rel_err_all[0::2], patch_artist=True, labels=labels)
    axs[0].set_xticklabels(labels, rotation='vertical')
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axs[0].legend(bp1["boxes"], alphas_fit_all, loc=legend_loc[0])
    axs[0].set_title('on observed data')

    # plot results on missing data
    bp2 = axs[1].boxplot(rel_err_all[1::2], patch_artist=True, labels=labels)
    axs[1].set_xticklabels(labels, rotation='vertical')
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axs[1].legend(bp2["boxes"], alphas_fit_all, loc=legend_loc[1])
    axs[1].set_title('on missing data')

    if cv > 0:
        fig.suptitle('relative errors, cv =' + str(cv) + ', RE variance = ' + str(gamma_list))
    else:
        fig.suptitle('relative errors, no noise, RE variance = ' + str(gamma_list))

    plt.show()
