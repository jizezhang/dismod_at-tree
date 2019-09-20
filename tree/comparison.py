import numpy as np
from mixetree import MixETree
import time
import matplotlib.pyplot as plt


def compare(data, node_list, alpha_true, iota_true, file_path, save_to, cv=0, holdout_nodes=None, zero_sum=True):
    n_data = data.shape[0]
    n_cov = len(alpha_true)
    if cv > 0:
        data['meas_std'] = data['true_val'] * cv
        data['meas_val'] = data['true_val'] + np.random.randn(n_data) * data['meas_std']

    elapsed_times = []
    alphas_fit_all = []

    # 1. fit given true alpha and iota
    mtr = MixETree(data, node_list, n_cov, file_path+'fit_given_true_alpha/', holdout_nodes)

    t0 = time.time()
    mtr.fit_low_level('oracle', {'iota': ([iota_true], [0], 'uniform'), 'alpha': (alpha_true, [0] * n_cov, 'uniform')},
                    zero_sum=zero_sum, use_lambda=True)
    #print('fit with true alpha, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)
    alphas_fit_all.append(alpha_true)

    # 2. Direct bottom level fit
    mtr = MixETree(data, node_list, n_cov, file_path+'bottom_fit/', holdout_nodes)
    mtr.reset()
    t0 = time.time()
    mtr.fit_low_level('bottom', zero_sum=zero_sum, use_lambda=True)
    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))
    #mtr.df['bottom_avgint'] = 0.0
    #mtr.df['bottom_res'] = 0.0
    mtr.fit_low_level('bottom', {'alpha': (alphas_fit, [0] * n_cov, 'uniform')}, zero_sum=zero_sum, use_lambda=True)
    #print('direct bottom level fit, elapsed ', time.time() - t0)
    elapsed_times.append(time.time() - t0)
    alphas_fit_all.append(np.round(alphas_fit, 3))

    # 3. Top-bottom fit
    mtr = MixETree(data, node_list, n_cov, file_path+'top_bottom/', holdout_nodes)

    t0 = time.time()
    mtr.fit_no_sim('top-bottom', zero_sum=zero_sum, use_lambda=True)
    #print('top-bottom fit, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        alphas_fit.append(mtr.alpha_est['1']['cov' + str(i + 1)])
    alphas_fit_all.append(np.round(alphas_fit, 3))

    # 4. Top bottom fit, always go to parent
    mtr = MixETree(data, node_list, n_cov, file_path + 'top_bottom2/', holdout_nodes)

    t0 = time.time()
    mtr.fit_no_sim('top-bottom2', zero_sum=zero_sum, use_lambda=True, no_leaf=True)
    # print('top-bottom fit, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        alphas_fit.append(mtr.alpha_est['1']['cov' + str(i + 1)])
    alphas_fit_all.append(np.round(alphas_fit, 3))

    # 5. Cascade fit, no lambda
    mtr = MixETree(data, node_list, n_cov, file_path+'cascade/', holdout_nodes)

    t0 = time.time()
    mtr.fit_sim('cascade', n_sim=10, use_lambda=False, zero_sum=zero_sum)
    #print('cascade fit, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))

    alphas_fit_all.append(np.round(alphas_fit, 3))

    # 5. Cascade fit, with lambda
    mtr = MixETree(data, node_list, n_cov, file_path+'cascade_lambda/', holdout_nodes)

    t0 = time.time()
    mtr.fit_sim('cascade-lambda', n_sim=10, use_lambda=True, zero_sum=zero_sum)
    #print('cascade fit with lambda, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))

    alphas_fit_all.append(np.round(alphas_fit, 3))

    # 6. Cascade fit, with skips
    mtr = MixETree(data, node_list, n_cov, file_path + 'cascade_skip/', holdout_nodes)

    t0 = time.time()
    mtr.fit_sim('cascade-skip', n_sim=10, use_lambda=False, zero_sum=zero_sum, skip=True)
    # print('cascade fit with lambda, elapsed', time.time() - t0)
    elapsed_times.append(time.time() - t0)

    alphas_fit = []
    for i in range(n_cov):
        values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
        alphas_fit.append(np.median(values))

    alphas_fit_all.append(np.round(alphas_fit, 3))

    data.to_csv(file_path+save_to)
    return alphas_fit_all, data[['hold_out', 'hold_out_branch', 'true_val', 'oracle_avgint', 'bottom_avgint', 'top-bottom_avgint',
                                 'top-bottom2_avgint', 'cascade_avgint', 'cascade-lambda_avgint', 'cascade-skip_avgint']]


def plot_result(results, alphas_fit_all, gamma_list, holdout_leaves=False, holdout_nodes=None, sep_branch=False, plot_ranges=None, cv=0, legend_loc='upper left'):
    # ---- plotting -------
    n_plots = 1
    if holdout_nodes is not None:
        if not sep_branch:
            n_plots += 1
        else:
            n_plots += len(holdout_nodes)
    if holdout_leaves:
        n_plots += 1
    fig, axs = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    labels = ['direct bottom fit', 'top bottom', 'top bottom 2', 'cascade', 'cascade with lambda']
    colors = ['pink', 'lightblue', 'lightgreen', 'lightgray', 'yellow']

    col_names = ['bottom_avgint', 'top-bottom_avgint', 'top-bottom2_avgint',
                 'cascade_avgint', 'cascade-lambda_avgint']
    true_observed = results[results['hold_out'] == False]['true_val'].values.reshape((-1, 1))
    #true_missing = results[results['hold_out'] == True]['true_val'].values.reshape((-1, 1))

    rel_err_observed = (results[results['hold_out'] == False][col_names].values - true_observed)/np.abs(true_observed)
    #rel_err_missing = (results[results['hold_out'] == True][col_names].values - true_missing)/np.abs(true_missing)

    # plot results on observed data
    bp1 = axs[0].boxplot(rel_err_observed, patch_artist=True, labels=labels)
    axs[0].set_xticklabels(labels, rotation='vertical')
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axs[0].legend(bp1["boxes"], alphas_fit_all, loc=legend_loc)
    axs[0].set_title('on observed data')
    if plot_ranges[0] is not None:
        axs[0].set_ylim(plot_ranges[0])

    if holdout_nodes is not None:
        if not sep_branch:
            true_missing = results[results['hold_out'] == True]['true_val'].values.reshape((-1, 1))
            rel_err_missing = (results[results['hold_out'] == True][col_names].values - true_missing)/np.abs(true_missing)
            # plot results on missing data
            bp2 = axs[1].boxplot(rel_err_missing, patch_artist=True, labels=labels)
            axs[1].set_xticklabels(labels, rotation='vertical')
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
            #axs[1].legend(bp2["boxes"], alphas_fit_all, loc=legend_loc[1])
            axs[1].set_title('on missing data')
            if plot_ranges[1] is not None:
                axs[1].set_ylim(plot_ranges[1])
        else:
            for i, node in enumerate(holdout_nodes):
                true_missing = results[(results['hold_out'] == True) & (results['hold_out_branch'] == node)]\
                    ['true_val'].values.reshape((-1, 1))
                rel_err_missing = (results[(results['hold_out'] == True) & (results['hold_out_branch'] == node)]
                                   [col_names].values - true_missing)/np.abs(true_missing)
                bp = axs[i+1].boxplot(rel_err_missing, patch_artist=True, labels=labels)
                axs[i+1].set_xticklabels(labels, rotation='vertical')
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                axs[i+1].set_title('on missing data from branch '+ node)
                if plot_ranges[1] is not None:
                    axs[i+1].set_ylim(plot_ranges[1])
    if holdout_leaves:
        true_missing = results[(results['hold_out'] == True) & (~results['hold_out_branch'].isin(holdout_nodes))]\
            ['true_val'].values.reshape((-1, 1))
        rel_err_missing = (results[(results['hold_out'] == True) & (~results['hold_out_branch'].isin(holdout_nodes))]
                           [col_names].values - true_missing) / np.abs(true_missing)
        bp = axs[-1].boxplot(rel_err_missing, patch_artist=True, labels=labels)
        axs[-1].set_xticklabels(labels, rotation='vertical')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axs[-1].set_title('on missing data from leaves')
        if plot_ranges[1] is not None:
            axs[-1].set_ylim(plot_ranges[1])

    fig.suptitle('relative errors, cv =' + str(cv) + ', RE variance = ' + str(gamma_list))

    plt.show()
