import numpy as np
import subprocess
import collections
import copy
import sys
import dismod_at
program = '/home/prefix/dismod_at.release/bin/dismod_at'


class SimDB:

    def __init__(self, df, node_list, n_cov, nodes_holdout=None, leaves_holdout=None):
        self.node_list = node_list
        self.n_cov = n_cov
        self.zerosum = False
        self.root = '1'

        self.create_node_table()
        if nodes_holdout is not None:
            for node in nodes_holdout:
                self.holdout_branch(df, node)
        if leaves_holdout is not None:
            for leaf in leaves_holdout:
                self.holdout_leaf_data(df, leaf)

        self.create_data_table(df)
        self.reset()

    def reset(self):
        self.create_default_tables()
        self.create_avgint_table()
        self.root = '1'
        self.zerosum = False

    def create_node_table(self):
        self.node_table = [{'name': '1', 'parent': ''}]
        self.node_one_level_above_leaves = []
        self.node_leaves = []
        self.node_height = {}
        self.node_depth = {}
        self.node_depth['1'] = 0
        self.node_parent_children = collections.defaultdict(list)
        self.node_child_parent = {}
        self.nodes_all = set()
        self.nodes_size = {}
        self.node_has_leaves = set([])

        def recurse(node_list, path):
            if any([type(x) is int for x in node_list]):  # all children are leaf nodes
                self.node_one_level_above_leaves.append(path)
            K = len(node_list)
            count = 0
            height = 0
            for k in range(K):
                kid_path = path+'_'+str(k+1)
                self.node_depth[kid_path] = self.node_depth[path] + 1
                self.node_table.append({'name': kid_path, 'parent': path})
                self.nodes_all.add(kid_path)
                self.node_parent_children[path].append(kid_path)
                self.node_child_parent[kid_path] = path
                if type(node_list[k]) is list:
                    recurse(node_list[k], kid_path)
                else:
                    self.node_leaves.append(kid_path)
                    self.node_height[kid_path] = 0
                    self.nodes_size[kid_path] = node_list[k]
                count += self.nodes_size[kid_path]
                height = max(height, self.node_height[kid_path])
            self.nodes_size[path] = count
            self.node_height[path] = height + 1

        recurse(self.node_list, '1')

    def create_data_table(self, df):
        self.data_table = list()
        row = {'integrand': 'Sincidence',
               'weight': 'constant',
               'time_lower': 0.0, 'time_upper': 0.0,
               'density': 'gaussian',
               'age_lower': 0.0, 'age_upper': 0.0,
               'one': 1.0, 'a': 1.0}
        for i, r in df.iterrows():
            row['node'] = r['node']
            row['meas_value'] = r['meas_val']
            row['meas_std'] = r['meas_std']
            row['hold_out'] = r['hold_out']
            for j in range(self.n_cov):
                row['cov'+str(j+1)] = r['cov'+str(j+1)]
            self.data_table.append(copy.copy(row))

    def holdout_leaf_data(self, df, leaves):
        node, percent = leaves[0], leaves[1]
        assert node in self.node_leaves
        n = np.round(self.nodes_size[node]*percent)
        count = 0

        for i, row in df.iterrows():
            if row['node'] == node and count < n:
                df.loc[i, 'hold_out'] = True
                count += 1
                df.loc[i, 'hold_out_branch'] = node


    def holdout_branch(self, df, node):
        nodes_to_remove = set()
        stack = [node]
        parent = self.node_child_parent[node]
        self.node_has_leaves.add(parent)
        while stack:
            v = stack.pop()
            if v not in self.node_leaves:
                stack.extend(self.node_parent_children[v])
                if v in self.node_one_level_above_leaves:
                    self.node_one_level_above_leaves.remove(v)
                del self.node_parent_children[v]
                self.node_has_leaves.discard(v)
            else:
                nodes_to_remove.add(v)
            del self.node_child_parent[v]
            del self.node_height[v]
            del self.node_depth[v]
        for v in nodes_to_remove:
            self.node_leaves.remove(v)

        self.node_parent_children[parent].remove(node)
        self.node_height[parent] = max([self.node_height[v] for v in self.node_parent_children[parent]]) + 1

        for i, row in df.iterrows():
            if row['node'] in nodes_to_remove:
                df.loc[i, 'hold_out'] = True
                df.loc[i, 'hold_out_branch'] = node

    def change_meas_density(self, density):
        for i, row in enumerate(self.data_table):
            self.data_table[i].update(density)

    def create_avgint_table(self, use_indicators=False):
        assert self.option_table[0]['name'] == 'parent_node_name'
        node = self.option_table[0]['value']
        self.avgint_table = []
        row = {
              'node': node,
              'weight':      'constant',
              'hold_out':    False,
              'time_lower':  0.0,
              'time_upper':  0.0,
              'one':         1.0,
              'a':          1.0,
              'age_lower': 0.0,
              'age_upper': 0.0,
            }
        for j in range(self.n_cov):
            row['cov'+str(j+1)] = 0.0
        if use_indicators:
            for kid in self.node_parent_children[node]:
                row[kid] = 0.0
        for kid in self.node_parent_children[node]:
            row['node'] = kid
            if use_indicators:
                row[kid] = 1.0
            row['integrand'] = 'Sincidence'
            self.avgint_table.insert(0, copy.copy(row))
        row['node'] = node
        for j in range(self.n_cov):
            row['integrand'] = 'mulcov_'+str(j+2)
            self.avgint_table.append(copy.copy(row))

    def create_default_tables(self):
        self.age_list = [0.0, 5.0]
        self.time_list = [0.0]
        self.rate_table = [{'name': 'iota', 'parent_smooth': 'smooth_iota',
                            'child_smooth': 'smooth_iota_child'}]

        self.covariate_table = [{'name': 'one', 'reference': 0.0},
                                {'name': 'a', 'reference': 0.0}]
        for j in range(self.n_cov):
            self.covariate_table.append({'name': 'cov'+str(j+1), 'reference':0.0})

        self.mulcov_table = [{
            'covariate': 'one',
            'type':      'meas_noise',
            'effected':  'Sincidence',
            'smooth':    'smooth_gamma'
        }, {
            'covariate': 'a',
            'type': 'rate_value',
            'effected': 'iota',
            'smooth': 'smooth_a'
        }]
        for j in range(self.n_cov):
            self.mulcov_table.append({
              'covariate': 'cov'+str(j+1),
              'type':      'rate_value',
              'effected':  'iota',
              'smooth':    'smooth_alpha_'+str(j+1)
             })

        self.smooth_table = [
              {
                   'name':    'smooth_iota',
                   'age_id':   [0],
                   'time_id':  [0],
                   'fun': lambda a, t:('prior_iota_value', None, None)
              }, {
                   'name':    'smooth_iota_child',
                   'age_id':   [0],
                   'time_id':  [0],
                   'fun': lambda a, t:('prior_iota_child', None, None),
                   'mulstd_value_prior_name': 'prior_lambda'
              }, {
                   'name':     'smooth_gamma',
                   'age_id':   [0],
                   'time_id':  [0],
                   'fun': lambda a, t:('prior_gamma', None, None)
              }, {
                   'name':     'smooth_a',
                   'age_id':   [0],
                   'time_id':  [0],
                   'fun': lambda a, t: ('prior_a', None, None)
              }
             ]

        for j in range(self.n_cov):
            self.smooth_table.append({
                 'name':     'smooth_alpha_'+str(j+1),
                 'age_id':   [0],
                 'time_id':  [0],
                 'fun': lambda a, t, j=j: ('prior_alpha_'+str(j+1), None, None)
            })

        self.prior_table = [
             {
                   'name':    'prior_iota_value',
                   'density': 'uniform',
                   'lower':   1e-4,
                   'upper':   1.,
                   'mean':    .05,
              }, {
                   'name':    'prior_iota_child',
                   'density': 'gaussian',
                   'mean':     0.0,
                   'std':      .7,
              }, {
                   'name':    'prior_lambda',
                   'density': 'uniform',
                   'mean':     1.0,
                   'lower':    1.0,
                   'upper':    1.0,
              }, {
                   'name':    'prior_gamma',
                   'density': 'uniform',
                   'mean':     0.0,
                   'lower':    0.0,
                   'upper':   10.0,
              }, {
                    'name': 'prior_a',
                    'density': 'uniform',
                    'lower': 0.0,
                    'upper': 0.,
                    'mean': 0.0
              }
             ]
        for j in range(self.n_cov):
            self.prior_table.append({
                 'name':    'prior_alpha_'+str(j+1),
                 'density': 'uniform',
                 'mean':     0.0,
                 'lower': -1.0,
                 'upper':    1.0,
            })

        self.integrand_table = [{'name': 'Sincidence'}]
        for j in range(self.n_cov):
            self.integrand_table.append({'name': 'mulcov_'+str(j+2)})

        self.weight_table = [{'name': 'constant',  'age_id': [0], 'time_id':[0],
                              'fun':lambda a, t:1.0}]
        self.option_table = [
                  {'name': 'parent_node_name',      'value': '1'},
                  {'name': 'random_seed', 'value': '0'},
                  {'name': 'rate_case',             'value': 'iota_pos_rho_zero'},
                  {'name': 'meas_noise_effect',    'value': 'add_var_scale_all'},
                  {'name': 'quasi_fixed',           'value': 'false'},
                  {'name': 'max_num_iter_fixed',    'value': '300'},
                  {'name': 'print_level_fixed',     'value': '0'},
                  {'name': 'tolerance_fixed',       'value': '1e-4'},
             ]

        self.avgint_table = list()

    def use_lambda(self):
        self.prior_table[2]['lower'] = 1e-4
        self.prior_table[2]['upper'] = 10.

    def disable_lambda(self):
        self.prior_table[2]['lower'] = 1.
        self.prior_table[2]['upper'] = 1.

    def use_gamma(self):
        self.prior_table[3]['upper'] = 10

    def disable_gamma(self):
        self.prior_table[3]['upper'] = 0.0

    def add_intercept(self):
        self.prior_table[4]['lower'] = -10.
        self.prior_table[4]['upper'] = 10.

    def change_re_std(self, u_std):
        self.prior_table[1]['std'] = u_std

    def pass_priors(self, name, mean, std, density):
        start = 0
        n = 1
        if name == 'alpha':
            start = 5
            n = self.n_cov
        assert len(mean) == n and len(std) == n
        for i in range(n):
            self.prior_table[start+i]['density'] = density
            self.prior_table[start+i]['mean'] = mean[i]
            if density == 'gaussian':
                self.prior_table[start+i]['std'] = std[i]
                if name == 'alpha':
                    self.prior_table[start+i]['lower'] = None
                    self.prior_table[start+i]['upper'] = None
            elif density == 'uniform':
                self.prior_table[start+i]['lower'] = mean[i]
                self.prior_table[start+i]['upper'] = mean[i]

    def update_parent_node(self, node):
        self.root = node
        assert self.option_table[0]['name'] == 'parent_node_name'
        self.option_table[0]['value'] = node
        self.create_avgint_table()

    def use_indicators(self, node, use_gamma=False):
        kids = self.node_parent_children[node]
        ancestor = {}
        for kid in kids:
            stack = [kid]
            while stack:
                v = stack.pop()
                if not self.node_parent_children[v]:
                    assert v in self.node_leaves
                    ancestor[v] = kid
                else:
                    assert v not in self.node_leaves
                    stack.extend(self.node_parent_children[v])
            self.covariate_table.append({'name': kid, 'reference': 0.0})
            self.mulcov_table.append({'covariate': kid, 'type': 'rate_value',
                                      'effected': 'iota', 'smooth': 'smooth_ind'})
        self.smooth_table.append({'name': 'smooth_ind', 'age_id': [0],
                                  'time_id': [0], 'fun': lambda a, t: ('prior_ind', None, None)})
        self.prior_table.append({'name': 'prior_ind', 'density': 'uniform', 'mean': 0.0})

        # disable intercept
        self.prior_table[4]['lower'] = 0.
        self.prior_table[4]['upper'] = 0.
        self.prior_table[4]['mean'] = 0.

        # disable base rate est
        self.prior_table[0]['lower'] = 1.
        self.prior_table[0]['mean'] = 1.
        self.prior_table[0]['upper'] = 1.

        # disable gamma
        if not use_gamma:
            self.prior_table[3]['lower'] = 0.
            self.prior_table[3]['mean'] = 0.
            self.prior_table[3]['upper'] = 0.

        for i, row in enumerate(self.data_table):
            self.data_table[i].update({kid: 0.0 for kid in kids})
            self.data_table[i][ancestor[row['node']]] = 1.0

        self.create_avgint_table(use_indicators=True)

    def fix_random_seed(self, seed):
        assert self.option_table[1]['name'] == 'random_seed'
        self.option_table[1]['value'] = seed

    def add_zero_sum(self):
        if not self.zerosum:
            self.zerosum = True
            self.option_table.append({'name': 'zero_sum_random', 'value': 'iota'})

    def initialize(self, file_name, to_csv=False):
        dismod_at.create_database(
          file_name,
          self.age_list,
          self.time_list,
          self.integrand_table,
          self.node_table,
          self.weight_table,
          self.covariate_table,
          self.avgint_table,
          self.data_table,
          self.prior_table,
          self.smooth_table,
          list(),
          self.rate_table,
          self.mulcov_table,
          self.option_table
        )

        if to_csv:
            flag = subprocess.call([program, file_name, 'init'])
            if flag != 0:
                sys.exit('command failed: flag = ' + str(flag))
            dismod_at.db2csv_command(file_name)
