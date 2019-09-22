import numpy as np
import subprocess
import shutil
import sys
import os
import pandas as pd
import collections
from simDB import SimDB
import dismod_at
import time

program = '/home/prefix/dismod_at.release/bin/dismod_at'


def system_command(command, verbose=True):
    if verbose:
        print(' '.join(command[1:]))
    flag = subprocess.call(command)
    if flag != 0:
        sys.exit('command failed: flag = ' + str(flag))
    return


class MixETree:

    def __init__(self, df, node_list, n_cov, file_path, nodes_holdout=None, leaves_holdout=None):
        self.df = df
        self.n_cov = n_cov
        self.db = SimDB(df, node_list, n_cov, nodes_holdout, leaves_holdout)
        self.nodes_holdout = nodes_holdout
        self.node_one_level_above_leaves = self.db.node_one_level_above_leaves
        self.node_leaves = self.db.node_leaves
        self.node_has_leaves = sorted(list(self.db.node_has_leaves))
        self.node_parent_children = self.db.node_parent_children
        self.node_child_parent = self.db.node_child_parent
        self.node_depth = self.db.node_depth
        self.node_height = self.db.node_height
        self.file_path = file_path
        if os.path.exists(self.file_path):
            shutil.rmtree(self.file_path)
        os.makedirs(self.file_path)
        assert self.file_path[-1] == "/"
        self.reset()

    def reset(self):
        self.base_rate_est = {}
        self.alpha_est = collections.defaultdict(dict)
        self.u_est = {}
        #self.a_est = {}
        self.gamma_est = {}
        self.lambda_est = {}
        self.db.reset()

    def fit(self, file_name, depth=1, fit_fixed=False, fit_both=True,
            verbose=True, write_to_csv=True):
        self.db.initialize(file_name)
        system_command([program, file_name, 'init'], verbose)
        if depth == 0 or fit_fixed:
            system_command([program, file_name, 'fit', 'fixed'], verbose)
            system_command([program, file_name, 'set', 'start_var', 'fit_var'], verbose)
            system_command([program, file_name, 'set', 'scale_var', 'fit_var'], verbose)
        if depth > 0 and fit_both:
            system_command([program, file_name, 'fit', 'both'], verbose)
        if write_to_csv:
            dismod_at.db2csv_command(file_name)

    def get_fitted_values(self, node):
        fit_var = pd.read_csv(self.file_path + 'variable.csv')
        for i, row in fit_var.iterrows():
            if row['fixed'] and row['var_type'] == 'rate':
                assert node == row['node']
                self.base_rate_est[row['node']] = row['fit_value']
            if not row['fixed'] and row['var_type'] == 'rate':
                if row['node'] in self.db.node_parent_children[node]:
                    self.u_est[row['node']] = row['fit_value']
                else:
                    print('holdout ', row['node'])
            if row['var_type'] == 'mulstd_value':
                self.lambda_est[node] = row['fit_value']
            if row['var_type'] == 'mulcov_meas_noise':
                self.gamma_est[node] = row['fit_value']
            if row['var_type'] == 'mulcov_rate_value':
                self.alpha_est[node][row['covariate']] = row['fit_value']

    # def get_fitted_values(self, node, save_u=True, save_yfit=True, has_indicators=False):
    #     fit_var = pd.read_csv(self.file_path + 'variable.csv')
    #     for i, row in fit_var.iterrows():
    #         if row['fixed'] == True and row['var_type'] == 'rate':
    #             self.base_rate_est[node] = row['fit_value']
    #         if row['fixed'] == False and row['var_type'] == 'rate':
    #             if save_u:
    #                 self.u_est[row['node']] = row['fit_value']
    #         if row['var_type'] == 'mulcov_rate_value' and row['covariate'] != 'a':
    #             self.alpha_est[node][row['covariate']] = row['fit_value']
    #         if row['var_type'] == 'mulcov_meas_noise':
    #             self.gamma_est[node] = row['fit_value']
    #         if row['var_type'] == 'mulstd_value':
    #             self.lambda_est[node] = row['fit_value']
    #         if row['var_type'] == 'mulcov_rate_value' and row['covariate'] == 'a':
    #             self.a_est[node] = row['fit_value']
    #     if not has_indicators:
    #         assert len(self.alpha_est[node]) == self.n_cov
    #     else:
    #         assert len(self.alpha_est[node]) == self.n_cov + len(self.node_parent_children[node])
    #
    #     avgint = []
    #     residual = []
    #     if save_yfit:
    #         data = pd.read_csv(self.file_path + 'data.csv')
    #         avgint = data['avgint'].values
    #         residual = data['residual'].values
    #     return avgint, residual

    def get_leaf_fit(self, col_prefix, is_leaf=False):
        data = pd.read_csv(self.file_path + 'data.csv')
        for i, row in data.iterrows():
            try:
                assert row['node'] == self.df.loc[int(row['data_id']), 'leaf']
            except ValueError:
                print(row['node'], row['data_id'])
            if row['node'] == row['child']:
                #print('writing...', row['node'])
                self.df.loc[row['data_id'], col_prefix+'_avgint'] = row['avgint']
                self.df.loc[row['data_id'], col_prefix+'_res'] = row['residual']
            elif self.nodes_holdout and row['child'] in self.nodes_holdout:
            #elif row['out'] == 1:
                #print('writing...', row['node'], row['child'], row['avgint'], row['residual'])
                self.df.loc[row['data_id'], col_prefix + '_avgint'] = row['avgint']
                self.df.loc[row['data_id'], col_prefix + '_res'] = row['residual']
            elif is_leaf:
                self.df.loc[row['data_id'], col_prefix + '_avgint'] = row['avgint']
                self.df.loc[row['data_id'], col_prefix + '_res'] = row['residual']

    # def get_leaf_fit(self, avgint, residual, sim=False, write_to_df=True):
    #     """ cannot be used with indicators currently """
    #     col_name = 'est_val'
    #     if sim:
    #         col_name = col_name + '_s'
    #     else:
    #         col_name = col_name + '_ns'
    #
    #     for i, row in self.df.iterrows():
    #         effects = 0.
    #         node_name = row['node']
    #         parent_name = '_'.join(node_name.split('_')[:-1])
    #         if parent_name in self.alpha_est:
    #             # if parent_name in self.node_1level_above_leaf:
    #             for j in range(self.n_cov):
    #                 effects += self.alpha_est[parent_name]['cov' + str(j + 1)] * row['cov' + str(j + 1)]
    #             effects += self.a_est.get(parent_name, 0) + self.u_est[node_name]
    #             self.df.loc[i, col_name] = self.base_rate_est[parent_name] * np.exp(effects)
    #         elif node_name in self.alpha_est:
    #             # elif node_name in self.leaves:
    #             for j in range(self.n_cov):
    #                 effects += self.alpha_est[node_name]['cov' + str(j + 1)] * row['cov' + str(j + 1)]
    #             effects += self.a_est.get(node_name, 0)
    #             self.df.loc[i, col_name] = self.base_rate_est[node_name] * np.exp(effects)
    #         else:
    #             print(node_name, parent_name)
    #             node = '_'.join(self.node_holdout.split('_')[:-1])
    #             print(node)
    #             assert node in self.alpha_est
    #             for j in range(self.n_cov):
    #                 effects += self.alpha_est[node]['cov' + str(j + 1)] * row['cov' + str(j + 1)]
    #             effects += self.a_est.get(node, 0)
    #             self.df.loc[i, col_name] = self.base_rate_est[node] * np.exp(effects)

        # if write_to_df:
        #     for i, row in self.df.iterrows():
        #         if not sim:
        #             self.df.loc[i, 'avgint_ns'] = avgint[i]
        #             self.df.loc[i, 'residual_ns'] = residual[i]
        #         else:
        #             self.df.loc[i, 'avgint_s'] = avgint[i]
        #             self.df.loc[i, 'residual_s'] = residual[i]

    def fit_root(self, priors=None, fit_fixed=False, fit_both=True, use_indicators=False):
        t_start = time.time()
        file_name = self.file_path + 'node_' + self.db.root + '.db'
        if priors is not None:
            for name, prior in priors.items():
                self.db.pass_priors(name, prior[0], prior[1], prior[2])
        if use_indicators:
            self.db.use_indicators(self.db.root)
            fit_both = False
        t0 = time.time()
        self.fit(file_name, fit_fixed=fit_fixed, fit_both=fit_both)
        tfit = time.time() - t0
        # print('root fit elapsed', time.time() - t0)
        self.get_fitted_values(self.db.root)
        # print('root total time', time.time() - t_start, 'fit time', tfit)

        # os.rename(self.file_path+'variable.csv',self.file_path+'ns_variable_1.csv')
        # os.rename(self.file_path+'data.csv',self.file_path+'ns_data_1.csv')

    def fit_low_level(self, col_prefix, priors=None, zero_sum=False, use_lambda=False, add_intercept=True,
                      fit_fixed=False, no_leaf=False, fitted_nodes=None):
        t_start = time.time()
        tfit = 0.
        if add_intercept:
            self.db.add_intercept()
        self.db.disable_gamma()
        if zero_sum:
            self.db.add_zero_sum()
        if use_lambda:
            self.db.use_lambda()
        else:
            self.db.disable_lambda()
        if priors is not None:
            for name, prior in priors.items():
                self.db.pass_priors(name, prior[0], prior[1], prior[2])

        fitted = set([])
        if fitted_nodes is not None:
            fitted = set(fitted_nodes)

        for node in self.node_has_leaves:
            if node not in fitted and self.node_height[node] > 1:
                #print('node has leaves')
                fitted.add(node)
                self.db.update_parent_node(node)
                self.db.use_gamma()
                file_name = self.file_path + 'ns_node_' + node + '.db'
                self.fit(file_name, fit_fixed=fit_fixed)
                self.get_fitted_values(node)
                self.get_leaf_fit(col_prefix)

                os.rename(self.file_path + 'variable.csv', self.file_path + 'ns_variable_' + node + '.csv')
                os.rename(self.file_path + 'data.csv', self.file_path + 'ns_data_' + node + '.csv')

        self.db.disable_gamma()
        for node in self.node_one_level_above_leaves:
            t0 = time.time()
            if node not in fitted and (no_leaf or self.node_height[node] == 1):
                #print('node parent level')
                fitted.add(node)
                self.db.update_parent_node(node)
                file_name = self.file_path + 'ns_node_' + node + '.db'
                self.fit(file_name, fit_fixed=fit_fixed)
                self.get_fitted_values(node)
                self.get_leaf_fit(col_prefix)

                os.rename(self.file_path + 'variable.csv', self.file_path + 'ns_variable_' + node + '.csv')
                os.rename(self.file_path + 'data.csv', self.file_path + 'ns_data_' + node + '.csv')
            else:
                for kid in self.node_parent_children[node]:
                    #print(kid)
                    if kid in self.node_leaves:
                        self.db.update_parent_node(kid)
                        file_name = self.file_path + 'ns_node_' + kid + '.db'
                        self.fit(file_name, depth=0)
                        self.get_fitted_values(kid)
                        self.get_leaf_fit(col_prefix, True)

                        os.rename(self.file_path + 'variable.csv', self.file_path + 'ns_variable_' + kid + '.csv')
                        os.rename(self.file_path + 'data.csv', self.file_path + 'ns_data_' + kid + '.csv')

        # print(node, 'total time', time.time() - t_start, 'fit time', tfit)

    def fit_no_sim(self, col_prefix, use_indicators=False, use_lambda=False, zero_sum=False, no_leaf=False):
        self.reset()
        if zero_sum:
            self.db.add_zero_sum()
        if use_lambda:
            self.db.use_lambda()

        # ---stage 1: fit root-----
        self.fit_root(use_indicators=use_indicators)
        self.get_leaf_fit(col_prefix)

        os.rename(self.file_path + 'variable.csv', self.file_path + 'ns_variable_1.csv')
        os.rename(self.file_path + 'data.csv', self.file_path + 'ns_data_1.csv')

        # -----stage 2: fit bottom nodes-------
        priors = {'alpha': (
            [self.alpha_est['1']['cov' + str(j + 1)] for j in range(self.n_cov)], [0] * self.n_cov, 'uniform'),
            'iota': ([self.base_rate_est['1']], [0], 'uniform')}
        self.db.reset()
        self.fit_low_level(col_prefix, priors, no_leaf=no_leaf, zero_sum=zero_sum,
                           use_lambda=use_lambda, fitted_nodes=['1'])

    def simulate(self, node, file_name, n_sim=10):
        N_str = str(n_sim)
        system_command([program, file_name, 'set', 'truth_var', 'fit_var'])
        system_command([program, file_name, 'set', 'start_var', 'fit_var'])
        system_command([program, file_name, 'set', 'scale_var', 'fit_var'])
        system_command([program, file_name, 'simulate', N_str])
        dismod_at.db2csv_command(file_name)
        system_command([program, file_name, 'sample', 'simulate', N_str])
        system_command([program, file_name, 'predict', 'sample'])
        dismod_at.db2csv_command(file_name)

        predict = pd.read_csv(self.file_path + 'predict.csv')
        predict_rate_dict = collections.defaultdict(list)
        predict_alpha_dict = collections.defaultdict(list)
        for i, row in predict.iterrows():
            if row['integrand'] == 'Sincidence':
                predict_rate_dict[row['node']].append(row['avgint'])
            else:
                predict_alpha_dict[row['integrand']].append(row['avgint'])

        assert len(predict_rate_dict) == len(self.node_parent_children[node])
        assert len(predict_alpha_dict) == self.n_cov

        rate_mean_std = {k: (np.mean(v), np.std(v)) for k, v in predict_rate_dict.items()}
        alpha_mean_std = {k: (np.mean(v), np.std(v)) for k, v in predict_alpha_dict.items()}

        os.rename(self.file_path + 'predict.csv', self.file_path + 'predict_' + node + '.csv')

        return rate_mean_std, alpha_mean_std

    def fit_sim(self, col_prefix, n_sim=10, use_lambda=True, zero_sum=False,
                fit_fixed=False, print_level=5, random_seed=0,
                skip=False):
        t_start = time.time()
        tfit = [0.0]
        tsim = [0.0]
        self.reset()
        if use_lambda:
            self.db.use_lambda()
        if zero_sum:
            self.db.add_zero_sum()

        def recurse(node, depth):
            self.db.update_parent_node(node)
            file_name = self.file_path + 's_node_' + node + '.db'

            if node in self.node_one_level_above_leaves and self.node_height[node] == 1:
                self.db.disable_gamma()
                t0 = time.time()
                self.fit(file_name, fit_fixed=fit_fixed)
                tfit[0] += time.time() - t0
                # print(node, 'fit elapsed', time.time() - t0)
                self.get_fitted_values(node)
                self.get_leaf_fit(col_prefix)
                # print(node, 'elapsed', time.time() - t0)
                os.rename(self.file_path + 'variable.csv', self.file_path + 's_variable_' + node + '.csv')
                os.rename(self.file_path + 'data.csv', self.file_path + 's_data_' + node + '.csv')
                return

            t0 = time.time()
            self.fit(file_name, fit_fixed=fit_fixed)
            tfit[0] += time.time() - t0
            # print(node, 'fit elapsed', time.time() - t0)
            self.get_fitted_values(node)
            if node in self.node_has_leaves or node in self.node_one_level_above_leaves:
                self.get_leaf_fit(col_prefix)

            os.rename(self.file_path + 'data.csv', self.file_path + 's_data_' + node + '.csv')
            os.rename(self.file_path + 'variable.csv', self.file_path + 's_variable_' + node + '.csv')

            t0 = time.time()
            rate_mean_std, alpha_mean_std = self.simulate(node, file_name, n_sim)
            # print(node, 'sim elapsed', time.time() - t0)
            tsim[0] += time.time() - t0

            alpha_mean = [alpha_mean_std['mulcov_' + str(j + 2)][0] for j in range(self.n_cov)]
            alpha_std = [(1 + self.gamma_est[node]) * alpha_mean_std['mulcov_' + str(j + 2)][1]
                         for j in range(self.n_cov)]

            self.db.pass_priors('alpha', alpha_mean, alpha_std, 'gaussian')

            for kid in self.node_parent_children[node]:
                iota_mean = rate_mean_std[kid][0]
                iota_std = rate_mean_std[kid][1] * (1 + self.gamma_est[node])
                self.db.pass_priors('iota', [iota_mean], [iota_std], 'gaussian')
                if skip and depth == 0:
                    self.db.add_intercept()
                    stack = [kid]
                    while stack:
                        v = stack.pop()
                        if v in self.node_one_level_above_leaves or v in self.node_has_leaves:
                            #print(v)
                            recurse(v, depth + 1)
                        else:
                            stack.extend(self.node_parent_children[v])
                else:
                    if kid not in self.node_leaves:
                        recurse(kid, depth + 1)
                        self.db.use_gamma()

        recurse('1', 0)
        #self.get_leaf_fit(col_prefix)
        #print('fit total elapsed', tfit[0])
        #print('sim total elapsed', tsim[0])
