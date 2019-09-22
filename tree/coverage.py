import numpy as np
from scipy.stats import norm
from mixetree import MixETree
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


class Coverage:

    def __init__(self, data, node_list, n_cov, file_path, holdout_nodes=None, holdout_leaves=None):
        self.data = data
        self.meas_std = data['meas_std'].values
        self.n_data = data.shape[0]
        self.n_cov = n_cov
        self.node_list = node_list
        self.file_path = file_path
        self.holdout_nodes = holdout_nodes
        self.holdout_leaves = holdout_leaves
        self.node_child_parent = None
        self.node_parent_children = None

    def fit_oracle(self, file_path, iota_true, alpha_true, y=None):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
        if self.node_parent_children is None:
            self.node_parent_children = mtr.node_parent_children
        mtr.reset()
        mtr.fit_low_level('oracle',
                          {'iota': ([iota_true], [0], 'uniform'), 'alpha': (alpha_true, [0] * n_cov, 'uniform')},
                          zero_sum=True, use_lambda=False)
        return mtr.base_rate_est, mtr.alpha_est, mtr.u_est

    def fit_bottom(self, file_path, y=None):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
        if self.node_parent_children is None:
            self.node_parent_children = mtr.node_parent_children
        mtr.reset()
        mtr.fit_low_level('bottom', zero_sum=True, use_lambda=False, add_intercept=False, fit_fixed=True)
        alphas_fit = []
        for i in range(self.n_cov):
            values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
            alphas_fit.append(np.median(values))
        mtr.fit_low_level('bottom', {'alpha': (alphas_fit, [0] * self.n_cov, 'uniform')},
                          zero_sum=True, use_lambda=False, add_intercept=False, fit_fixed=True)
        # print('direct bottom level fit, elapsed ', time.time() - t0)
        return mtr.base_rate_est, mtr.alpha_est, mtr.u_est

    def fit_top_bottom(self, file_path, y=None, no_leaf=False):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
        if self.node_parent_children is None:
            self.node_parent_children = mtr.node_parent_children
        mtr.fit_no_sim('top-bottom', zero_sum=True, use_lambda=False, no_leaf=no_leaf)
        return mtr.base_rate_est, mtr.alpha_est, mtr.u_est

    def fit_cascade(self, file_path, y=None, n_sim=10, use_lambda=False, skip=False):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
        if self.node_parent_children is None:
            self.node_parent_children = mtr.node_parent_children
        mtr.fit_sim('cascade', n_sim=n_sim, use_lambda=use_lambda, zero_sum=True, skip=skip)
        return mtr.base_rate_est, mtr.alpha_est, mtr.u_est

    def get_params_distribution(self, method, n_run=30, n_sim=10):
        file_path = self.file_path + method + '/'

        base_rate_all = defaultdict(list)
        alpha_all = defaultdict(list)
        u_all = defaultdict(list)

        def run_model(method, y=None):
            if method == 'bottom':
                return self.fit_bottom(file_path, y=y)
            elif method == 'top-bottom':
                return self.fit_top_bottom(file_path, y=y)
            elif method == 'cascade':
                return self.fit_cascade(file_path, n_sim=n_sim, y=y)

        def integrate_result(base_rate_est, alpha_est, u_est):
            for node, value in base_rate_est.items():
                base_rate_all[node].append(value)
            for node, value in alpha_est.items():
                alpha_all[node].append(value)
            for node, value in u_est.items():
                u_all[node].append(value)

        run_model(method)

        yfit = self.data[method + '_avgint'].values.reshape((-1, 1))
        ysim = np.random.randn(self.n_data, n_run) * self.meas_std.reshape((-1, 1)) + yfit

        for k in range(n_run):
            print('---- run', k, '--------')
            base_rate, alpha, u = run_model(method, ysim[:,k])
            integrate_result(base_rate, alpha, u)

        base_rate_dist = {}
        for node, values in base_rate_all.items():
            base_rate_dist[node] = (np.mean(values), np.std(values))

        alpha_dist = {}
        for node, covs in alpha_all.items():
            cov_comb = defaultdict(list)
            for cov in covs:
                for name, value in cov.items():
                    cov_comb[name].append(value)
            cov_dist = {}
            for name, values in cov_comb.items():
                cov_dist[name] = (np.mean(values), np.std(values))
            alpha_dist[node] = cov_dist

        u_dist = {}
        for node, values in u_all.items():
            u_dist[node] = (np.mean(values), np.std(values))

        return base_rate_dist, alpha_dist, u_dist

    def draw_params(self, base_rate_dist, alpha_dist, u_dist, n_draws=1000):
        base_rate_draws = {}
        for node, dist in base_rate_dist.items():
            mu, sigma = dist[0], dist[1]
            base_rate_draws[node] = np.random.randn(n_draws)*sigma + mu

        alpha_draws = {}
        for node, covs in alpha_dist.items():
            draws = {}
            for cov, dist in covs.items():
                mu, sigma = dist[0], dist[1]
                draws[cov] = np.random.randn(n_draws)*sigma + mu
            alpha_draws[node] = draws

        u_draws = {}
        for node, dist in u_dist.items():
            mu, sigma = dist[0], dist[1]
            u_draws[node] = np.random.randn(n_draws) * sigma + mu

        return base_rate_draws, alpha_draws, u_draws

    def draw_ys(self, base_rate_dist, alpha_dist, u_dist, n_draws=100):
        base_rate_draws, alpha_draws, u_draws = self.draw_params(base_rate_dist, alpha_dist, u_dist, n_draws)
        ydraws_obs = []
        ydraws_miss_branch = defaultdict(list)
        ydraws_miss_leaves = []


        # ---- sample u for missing branches based on empirical variance from their siblings
        u_samples = {}
        for node in self.holdout_nodes:
            parent = '_'.join(node.split('_')[:-1])
            samples = []
            for kid in self.node_parent_children[parent]:
                samples.extend(u_draws[kid])
            u_std = np.std(samples)
            u_samples[node] = np.random.randn(n_draws)*u_std

        for i, row in self.data.iterrows():
            if row['hold_out'] and row['hold_out_branch'] != row['node']:
                node = row['hold_out_branch']
                parent = '_'.join(node.split('_')[:-1])
                y_draws = base_rate_draws[parent]*np.ones(n_draws)  # otherwise need deepcopy
                y_draws *= np.exp(u_samples[node])
                for name, values in alpha_draws[parent].items():
                    if name != 'a':
                        y_draws *= np.exp(values*row[name])
                    else:
                        y_draws *= np.exp(values)
                if node in ydraws_miss_branch:
                    ydraws_miss_branch[node].append((i, row['node'], row['true_val'], y_draws))
                else:
                    ydraws_miss_branch[node] = [(i, row['node'], row['true_val'], y_draws)]
            else:
                node = row['node']
                if node in base_rate_draws:
                    #print(node)
                    y_draws = base_rate_draws[node]*np.ones(n_draws)
                    for name, values in alpha_draws[node].items():
                        if name != 'a':
                            y_draws *= np.exp(values * row[name])
                        else:
                            y_draws *= np.exp(values)
                    if row['hold_out']:
                        ydraws_miss_leaves.append((i, node, row['true_val'], y_draws))
                    else:
                        ydraws_obs.append((i, node, row['true_val'], y_draws))
                else:
                    parent = self.node_child_parent[node]
                    y_draws = base_rate_draws[parent] * np.exp(u_draws[node])
                    for name, values in alpha_draws[parent].items():
                        if name != 'a':
                            y_draws *= np.exp(values * row[name])
                        else:
                            y_draws *= np.exp(values)
                    if row['hold_out']:
                        ydraws_miss_leaves.append((i, node, row['true_val'], y_draws))
                    else:
                        ydraws_obs.append((i, node, row['true_val'], y_draws))

        return ydraws_obs, ydraws_miss_leaves, ydraws_miss_branch

    def plot_draws(self, method, ydraws_obs, ydraws_miss_leaves=None, ydraws_miss_branch=None,
                   samples_per_type=2, yrange=None, legend=True):
        n_plots = 1
        if ydraws_miss_leaves is not None:
            n_plots += 1
        if ydraws_miss_branch is not None:
            n_plots += len(ydraws_miss_branch)
        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots*5, 5))

        def draw_samples(draws_all):
            n = len(draws_all)
            idx = np.random.choice(n, size=samples_per_type, replace=False)
            samples = []
            labels = []
            data_ids = []
            for i in idx:
                samples.append(draws_all[i][3] - draws_all[i][2])
                labels.append(draws_all[i][1])
                data_ids.append(draws_all[i][0])
            return samples, labels, data_ids

        # ---- plot coverage for observed data -----------
        samples_draws_obs, labels_obs, data_obs_ids = draw_samples(ydraws_obs)

        bp = axs[0].boxplot(samples_draws_obs, patch_artist=True, labels=labels_obs)
        axs[0].set_xticklabels(labels_obs)
        axs[0].set_title('on observed data')
        if yrange is not None:
            axs[0].set_ylim(range)
        if legend:
            axs[0].legend(bp["boxes"], data_obs_ids)

        # ---- plot coverage for data missing from leaves -------
        k_plot = 1
        if ydraws_miss_leaves is not None:
            samples_draws_miss_leaves, labels_miss_leaves, data_miss_leaves_ids = draw_samples(ydraws_miss_leaves)

            bp = axs[k_plot].boxplot(samples_draws_miss_leaves, patch_artist=True, labels=labels_miss_leaves)
            axs[k_plot].set_xticklabels(labels_miss_leaves)
            axs[k_plot].set_title('on data missing from leaves')
            if yrange is not None:
                axs[k_plot].set_ylim(range)
            if legend:
                axs[k_plot].legend(bp['boxes'], data_miss_leaves_ids)

            k_plot += 1

        if ydraws_miss_branch is not None:
            for node, node_draws in ydraws_miss_branch.items():
                samples_draws_miss_branch, labels_miss_branch, data_miss_branch_ids = draw_samples(node_draws)
                bp = axs[k_plot].boxplot(samples_draws_miss_branch, patch_artist=True, labels=labels_miss_branch)
                axs[k_plot].set_xticklabels(labels_miss_branch)
                axs[k_plot].set_title('on data missing from branch '+node)
                k_plot += 1
                if legend:
                    axs[k_plot].legend(bp['boxes'], data_miss_branch_ids)
        fig.suptitle('draw_value - true_value, method ' + method)
        plt.show()

    def get_y_coverage(self, method, n_draws=100, n_run=10, n_sim=10,
                       plot=True, samples_per_plot=5, save_draws=True, yrange=None, legend=True):
        base_rate_dist, alpha_dist, u_dist = self.get_params_distribution(method, n_run, n_sim)
        print('--- drawing params samples --------')
        ydraws_obs, ydraws_miss_leaves, ydraws_miss_branch = self.draw_ys(base_rate_dist,
                                                                          alpha_dist, u_dist, n_draws)

        if save_draws:
            ydraws_all = ydraws_obs + ydraws_miss_leaves
            for node, draws in ydraws_miss_branch.items():
                ydraws_all.extend(draws)
            df = pd.DataFrame(ydraws_all, columns=('data_id', 'node', 'true_val', 'draws'))
            df.to_csv(self.file_path + 'ydraws_' + method + '.csv')

        if plot:
            self.plot_draws(method, ydraws_obs, ydraws_miss_leaves, ydraws_miss_branch,
                            samples_per_plot, yrange=yrange, legend=legend)

        return ydraws_obs, ydraws_miss_leaves, ydraws_miss_branch
