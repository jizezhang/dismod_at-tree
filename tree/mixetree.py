import numpy as np
import copy
import subprocess
import shutil
import sys
import os
import collections
import json
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

    def __init__(self, df, node_list, n_cov, file_path, node_holdout=None):
        self.df = df
        self.n_cov = n_cov
        self.db = SimDB(df, node_list, n_cov, node_holdout)
        self.node_holdout = node_holdout
        self.node_1level_above_leaf = self.db.node_1level_above_leaf
        self.leaves = self.db.leaves
        self.node_parent_children = self.db.node_parent_children
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
        self.a_est = {}
        self.gamma_est = {}
        self.lambda_est = {}
        self.db.reset()

    def fit(self, file_name, depth=1, fit_fixed=False, fit_both=True,
            verbose=True, write_to_csv=True):
        self.db.initialize(file_name)
        system_command([program, file_name, 'init'], verbose)
        if depth == 0 or fit_fixed == True:
            system_command([program, file_name, 'fit', 'fixed'], verbose)
            system_command([program, file_name, 'set', 'start_var', 'fit_var'], verbose)
        if depth > 0 and fit_both == True:
            system_command([program, file_name, 'fit', 'both'], verbose)
        if write_to_csv:
            dismod_at.db2csv_command(file_name)

    def getFittedValues(self, node, save_u=True, save_yfit=True, has_indicators=False):
        fit_var = pd.read_csv(self.file_path + 'variable.csv')
        for i, row in fit_var.iterrows():
            if row['fixed'] == True and row['var_type'] == 'rate':
                self.base_rate_est[node] = row['fit_value']
            if row['fixed'] == False and row['var_type'] == 'rate':
                if save_u:
                    self.u_est[row['node']] = row['fit_value']
            if row['var_type'] == 'mulcov_rate_value' and row['covariate'] != 'a':
                self.alpha_est[node][row['covariate']] = row['fit_value']
            if row['var_type'] == 'mulcov_meas_noise':
                self.gamma_est[node] = row['fit_value']
            if row['var_type'] == 'mulstd_value':
                self.lambda_est[node] = row['fit_value']
            if row['var_type'] == 'mulcov_rate_value' and row['covariate'] == 'a':
                self.a_est[node] = row['fit_value']
        if has_indicators == False:
            assert len(self.alpha_est[node]) == self.n_cov
        else:
            assert len(self.alpha_est[node]) == self.n_cov + len(self.node_parent_children[node])

        avgint = []
        residual = []
        if save_yfit == True:
            data = pd.read_csv(self.file_path + 'data.csv')
            avgint = data['avgint'].values
            residual = data['residual'].values
        return avgint, residual

    def getLeafFit(self, avgint, residual, sim=False, write_to_df=True):
        """ cannot be used with indicators currently """
        col_name = 'est_val'
        if sim:
            col_name = col_name + '_s'
        else:
            col_name = col_name + '_ns'

        for i, row in self.df.iterrows():
            effects = 0.
            node_name = row['node']
            parent_name = '_'.join(node_name.split('_')[:-1])
            if parent_name in self.alpha_est:
                # if parent_name in self.node_1level_above_leaf:
                for j in range(self.n_cov):
                    effects += self.alpha_est[parent_name]['cov' + str(j + 1)] * row['cov' + str(j + 1)]
                effects += self.a_est.get(parent_name, 0) + self.u_est[node_name]
                self.df.loc[i, col_name] = self.base_rate_est[parent_name] * np.exp(effects)
            elif node_name in self.alpha_est:
                # elif node_name in self.leaves:
                for j in range(self.n_cov):
                    effects += self.alpha_est[node_name]['cov' + str(j + 1)] * row['cov' + str(j + 1)]
                effects += self.a_est.get(node_name, 0)
                self.df.loc[i, col_name] = self.base_rate_est[node_name] * np.exp(effects)
            else:
                print(node_name, parent_name)
                node = '_'.join(self.node_holdout.split('_')[:-1])
                print(node)
                assert node in self.alpha_est
                for j in range(self.n_cov):
                    effects += self.alpha_est[node]['cov' + str(j + 1)] * row['cov' + str(j + 1)]
                effects += self.a_est.get(node, 0)
                self.df.loc[i, col_name] = self.base_rate_est[node] * np.exp(effects)

                # raise RuntimeError('Error')

        if write_to_df == True:
            for i, row in self.df.iterrows():
                if sim == False:
                    self.df.loc[i, 'avgint_ns'] = avgint[i]
                    self.df.loc[i, 'residual_ns'] = residual[i]
                else:
                    self.df.loc[i, 'avgint_s'] = avgint[i]
                    self.df.loc[i, 'residual_s'] = residual[i]

    def fitRoot(self, priors=None, fit_fixed=True, fit_both=True, use_indicators=False):
        t_start = time.time()
        file_name = self.file_path + 'node_' + self.db.root + '.db'
        if priors != None:
            for name, prior in priors.items():
                self.db.passPriors(name, prior[0], prior[1], prior[2])
        if use_indicators:
            self.db.useIndicators(self.db.root)
            fit_both = False
        t0 = time.time()
        self.fit(file_name, fit_fixed=fit_fixed, fit_both=fit_both)
        tfit = time.time() - t0
        print('root fit elapsed', time.time() - t0)
        self.getFittedValues(self.db.root, save_u=True, has_indicators=use_indicators)
        print('root total time', time.time() - t_start, 'fit time', tfit)

        # os.rename(self.file_path+'variable.csv',self.file_path+'ns_variable_1.csv')
        # os.rename(self.file_path+'data.csv',self.file_path+'ns_data_1.csv')

    def fitLowLevel(self, priors=None, zero_sum=False, use_lambda=False, fit_fixed=False, no_leaf=False):
        t_start = time.time()
        tfit = 0.
        self.db.addIntercept()
        self.db.disableGamma()
        if zero_sum == True:
            self.db.addZeroSum()
        if use_lambda == True:
            self.db.useLambda()
        if priors != None:
            for name, prior in priors.items():
                self.db.passPriors(name, prior[0], prior[1], prior[2])

        avgint_from_files = []
        residual_from_files = []

        for node in self.node_1level_above_leaf:
            t0 = time.time()
            if node in self.leaves:
                if no_leaf == False:
                    self.db.updateParentNode(node)
                    file_name = self.file_path + 'ns_node_' + node + '.db'
                    self.fit(file_name, depth=0)
                else:
                    parent = '_'.join(node.split('_')[:-1])
                    self.db.updateParentNode(parent)
                    file_name = self.file_path + 'ns_node_' + parent + '.db'
                    self.fit(file_name, fit_fixed=fit_fixed)
            else:
                self.db.updateParentNode(node)
                file_name = self.file_path + 'ns_node_' + node + '.db'
                self.fit(file_name, fit_fixed=fit_fixed)
            tfit += time.time() - t0
            print(node, 'fit elapsed', time.time() - t0)
            if node in self.leaves and no_leaf == True:
                parent = '_'.join(node.split('_')[:-1])
                avgint, residual = self.getFittedValues(parent)
            else:
                avgint, residual = self.getFittedValues(node)
                avgint_from_files.extend(avgint)
                residual_from_files.extend(residual)

            os.rename(self.file_path + 'variable.csv', self.file_path + 'ns_variable_' + node + '.csv')
            os.rename(self.file_path + 'data.csv', self.file_path + 'ns_data_' + node + '.csv')
        if no_leaf == False:
            self.getLeafFit(avgint_from_files, residual_from_files)
        else:
            self.getLeafFit(avgint_from_files, residual_from_files, write_to_df=False)

        print(node, 'total time', time.time() - t_start, 'fit time', tfit)

        # return avgint_from_files, residual_from_files

    def fitNoSim(self, use_indicators=False, use_lambda=False, zero_sum=False, no_leaf=False):
        self.reset()
        if zero_sum == True:
            self.db.addZeroSum()
        if use_lambda == True:
            self.db.useLambda()

        # ---stage 1: fit root-----
        self.fitRoot(use_indicators=use_indicators)

        os.rename(self.file_path + 'variable.csv', self.file_path + 'ns_variable_1.csv')
        os.rename(self.file_path + 'data.csv', self.file_path + 'ns_data_1.csv')

        # -----stage 2: fit bottom nodes-------
        priors = {'alpha': (
        [self.alpha_est['1']['cov' + str(j + 1)] for j in range(self.n_cov)], [0] * self.n_cov, 'uniform'),
                  'iota': ([self.base_rate_est['1']], [0], 'uniform')}
        self.db.reset()
        self.fitLowLevel(priors, no_leaf=no_leaf, zero_sum=zero_sum, use_lambda=use_lambda)

    def simulate(self, node, file_name, n_sim=10):
        N_str = str(n_sim)
        system_command([program, file_name, 'set', 'truth_var', 'fit_var'])
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
        # return predict_rate_dict, predict_alpha_dict

    def fitSim(self, n_sim=10, use_lambda=True, zero_sum=False,
               fit_fixed=False, print_level=5, random_seed=0,
               skip=False):
        t_start = time.time()
        tfit = [0.0]
        tsim = [0.0]
        self.reset()
        if use_lambda == True:
            self.db.useLambda()
        if zero_sum == True:
            self.db.addZeroSum()
        avgint_from_files = []
        residual_from_files = []

        def recurse(node, depth):
            self.db.updateParentNode(node)
            file_name = self.file_path + 's_node_' + node + '.db'

            # if node in self.node_1level_above_leaf:
            #     if node in self.leaves:
            #         self.fit(file_name, depth=0)
            #     else:
            #         self.fit(file_name, fit_fixed=fit_fixed)
            #     avgint, residual = self.getFittedValues(node)
            #     avgint_from_files.extend(avgint)
            #     residual_from_files.extend(residual)
            #     os.rename(self.file_path+'variable.csv',self.file_path+'s_variable_'+node+'.csv')
            #     os.rename(self.file_path+'data.csv',self.file_path+'s_data_'+node+'.csv')
            #     return
            #
            if node in self.node_1level_above_leaf:
                self.db.disableGamma()
                if node not in self.leaves:
                    t0 = time.time()
                    self.fit(file_name, fit_fixed=fit_fixed)
                    tfit[0] += time.time() - t0
                    print(node, 'fit elapsed', time.time() - t0)
                    avgint, residual = self.getFittedValues(node)
                    avgint_from_files.extend(avgint)
                    residual_from_files.extend(residual)
                    # print(node, 'elapsed', time.time() - t0)
                    os.rename(self.file_path + 'variable.csv', self.file_path + 's_variable_' + node + '.csv')
                    os.rename(self.file_path + 'data.csv', self.file_path + 's_data_' + node + '.csv')
                elif skip == True:
                    t0 = time.time()
                    self.fit(file_name, depth=0)
                    tfit[0] += time.time() - t0
                    print(node, 'fit elapsed', time.time() - t0)
                    avgint, residual = self.getFittedValues(node)
                    avgint_from_files.extend(avgint)
                    residual_from_files.extend(residual)
                    # print(node, 'elapsed', time.time() - t0)
                    os.rename(self.file_path + 'variable.csv', self.file_path + 's_variable_' + node + '.csv')
                    os.rename(self.file_path + 'data.csv', self.file_path + 's_data_' + node + '.csv')
                return

            t0 = time.time()
            self.fit(file_name, fit_fixed=fit_fixed)
            tfit[0] += time.time() - t0
            print(node, 'fit elapsed', time.time() - t0)
            self.getFittedValues(node, save_u=True, save_yfit=False)

            os.rename(self.file_path + 'data.csv', self.file_path + 's_data_' + node + '.csv')
            os.rename(self.file_path + 'variable.csv', self.file_path + 's_variable_' + node + '.csv')

            t0 = time.time()
            rate_mean_std, alpha_mean_std = self.simulate(node, file_name, n_sim)
            print(node, 'sim elapsed', time.time() - t0)
            tsim[0] += time.time() - t0

            alpha_mean = [alpha_mean_std['mulcov_' + str(j + 2)][0] for j in range(self.n_cov)]
            alpha_std = [(1 + self.gamma_est[node]) * alpha_mean_std['mulcov_' + str(j + 2)][1]
                         for j in range(self.n_cov)]

            self.db.passPriors('alpha', alpha_mean, alpha_std, 'gaussian')

            for kid in self.node_parent_children[node]:
                iota_mean = rate_mean_std[kid][0]
                iota_std = rate_mean_std[kid][1] * (1 + self.gamma_est[node])
                self.db.passPriors('iota', [iota_mean], [iota_std], 'gaussian')
                if skip == True and depth == 0:
                    self.db.addIntercept()
                    stack = [kid]
                    while stack:
                        n = stack.pop()
                        if n in self.node_1level_above_leaf:
                            recurse(n, depth + 1)
                        else:
                            stack.extend(self.node_parent_children[n])
                else:
                    recurse(kid, depth + 1)
                    self.db.useGamma()

        recurse('1', 0)
        self.getLeafFit(avgint_from_files, residual_from_files, write_to_df=False, sim=True)
        print('fit total elapsed', tfit[0])
        print('sim total elapsed', tsim[0])
