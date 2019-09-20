import numpy as np
from scipy.stats import norm
from mixetree import MixETree
import time
from collections import defaultdict


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

    def fit_oracle(self, file_path, y=None):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
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
        mtr.reset()
        mtr.fit_low_level('bottom', zero_sum=True, use_lambda=False, add_intercept=False)
        alphas_fit = []
        for i in range(self.n_cov):
            values = [x['cov' + str(i + 1)] for k, x in mtr.alpha_est.items()]
            alphas_fit.append(np.median(values))
        mtr.fit_low_level('bottom', {'alpha': (alphas_fit, [0] * self.n_cov, 'uniform')},
                          zero_sum=True, use_lambda=False, add_intercept=False)
        # print('direct bottom level fit, elapsed ', time.time() - t0)
        return mtr.base_rate_est, mtr.alpha_est, mtr.u_est

    def fit_top_bottom(self, file_path, y=None, no_leaf=False):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
        mtr.fit_no_sim('top-bottom', zero_sum=True, use_lambda=False, no_leaf=no_leaf)
        return mtr.base_rate_est, mtr.alpha_est, mtr.u_est

    def fit_cascade(self, file_path, y=None, n_sim=10, use_lambda=False, skip=False):
        if y is not None:
            self.data['meas_val'] = y
        mtr = MixETree(self.data, self.node_list, self.n_cov, file_path, self.holdout_nodes, self.holdout_leaves)
        if self.node_child_parent is None:
            self.node_child_parent = mtr.node_child_parent
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

    def draw_params(self, base_rate_dist, alpha_dist, u_dist, n_draw=1000):
        base_rate_draws = {}
        for node, dist in base_rate_dist.items():
            mu, sigma = dist[0], dist[1]
            base_rate_draws[node] = np.random.randn(n_draw)*sigma + mu

        alpha_draws = {}
        for node, covs in alpha_dist.items():
            draws = {}
            for cov, dist in covs.items():
                mu, sigma = dist[0], dist[1]
                draws[cov] = np.random.randn(n_draw)*sigma + mu
            alpha_draws[node] = draws

        u_draws = {}
        for node, dist in u_dist.items():
            mu, sigma = dist[0], dist[1]
            u_draws[node] = np.random.randn(n_draw) * sigma + mu

        return base_rate_draws, alpha_draws, u_draws

    def get_y_coverage(self, method, n_run=30, n_draw=1000, n_sim=10):
        base_rate_dist, alpha_dist, u_dist = self.get_params_distribution(method, n_run, n_sim)
        print('--- drawing params samples --------')
        base_rate_draws, alpha_draws, u_draws = self.draw_params(base_rate_dist, alpha_dist, u_dist, n_draw)
        for i, row in self.data.iterrows():
            mean = row[method+'_avgint']
            std = row['meas_std']
            lb = norm.ppf(.025, mean, std)
            ub = norm.ppf(.975, mean, std)
            if row['hold_out']:
                node = row['hold_out_branch']
                parent = self.node_child_parent[node]
                y_draws = base_rate_draws[parent]*np.exp(u_draws[node])
                for name, values in alpha_draws[parent].items():
                    y_draws += values*row[name]
                self.data.loc[i, 'cvr'] = len(np.where((y_draws >= lb) & (y_draws <= ub)))/float(n_draw)
            else:
                node = row['node']
                if node in base_rate_draws:
                    y_draws = base_rate_draws[node]
                    for name, values in alpha_draws[node].items():
                        y_draws += values*row[name]
                    self.data.loc[i, 'cvr'] = len(np.where((y_draws >= lb) & (y_draws <= ub)))/float(n_draw)
                else:
                    parent = self.node_child_parent[node]
                    y_draws = base_rate_draws[parent] * np.exp(u_draws[node])
                    for name, values in alpha_draws[parent]:
                        y_draws += values * row[name]
                    self.data.loc[i, 'cvr'] = len(np.where((y_draws >= lb) & (y_draws <= ub))) / float(n_draw)





