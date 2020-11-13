#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-11-28 下午2:47
import gzip
import os
import pickle
from collections import defaultdict

import numpy as np
from .features.selection import FeatureSelector
from .models.tree import RstTree
from .utils.other import vectorize


class DataHelper(object):
    def __init__(self, max_action_feat_num=-1, max_relation_feat_num=-1,
                 min_action_feat_occur=1, min_relation_feat_occur=1, brown_clusters=None):
        # number of features, feature selection will be triggered if feature num is larger than this
        self.max_action_feat_num = max_action_feat_num
        self.min_action_feat_occur = min_action_feat_occur
        #self.max_relation_feat_num = max_relation_feat_num
        #self.min_relation_feat_occur = min_relation_feat_occur
        self.brown_clusters = brown_clusters
        self.action_feat_template = {}
        #self.relation_feat_template_level_0 = {}
        #self.relation_feat_template_level_1 = {}
        #self.relation_feat_template_level_2 = {}
        self.action_map, self.relation_map = {}, {}
        self.action_cnt, self.relation_cnt = {}, {}
        # train rst trees
        self.rst_tree_instances = []

        
        
    def create_data_helper(self, data_dir):
        # read train data
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        self.action_feat_template = {}
        self.action_feat_counts = {}
        for fdis in files:
            print("Generating tree for", os.path.basename(fdis))
            rst_trees = [self.read_rst_trees(fdis)]
            if rst_trees[0] is None:
                print("Could not find merge file for", os.path.basename(fdis))
                continue
            action_sample = [sample for rst_tree in rst_trees for sample in
                          rst_tree.generate_action_samples(self.brown_clusters)]
            self._build_action_map(action_sample)
            
        self.action_feat_template = self._build_action_feat_template(topn=self.max_action_feat_num,
                                                                     thresh=self.min_action_feat_occur)
        

    def save_data_helper(self, fname):
        print('Save data helper...')
        data_info = {
            'action_feat_template': self.action_feat_template,
            #'relation_feat_template_level_0': self.relation_feat_template_level_0,
            #'relation_feat_template_level_1': self.relation_feat_template_level_1,
            #'relation_feat_template_level_2': self.relation_feat_template_level_2,
            'action_map': self.action_map,
            #'relation_map': self.relation_map
        }
        with open(fname, 'wb') as fout:
            pickle.dump(data_info, fout)

    def load_data_helper(self, fname):
        print('Load data helper ...')
        with open(fname, 'rb') as fin:
            data_info = pickle.load(fin)
        self.action_feat_template = data_info['action_feat_template']
        #self.relation_feat_template_level_0 = data_info['relation_feat_template_level_0']
        #self.relation_feat_template_level_1 = data_info['relation_feat_template_level_1']
        #self.relation_feat_template_level_2 = data_info['relation_feat_template_level_2']
        self.action_map = data_info['action_map']
        #self.relation_map = data_info['relation_map']

    def gen_action_train_data(self, data_dir):
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        for fdis in files:
            rst_tree = self.read_rst_trees(fdis)
            if rst_tree is None:
                continue
            for feats, action in rst_tree.generate_action_samples(self.brown_clusters):
                yield vectorize(feats, self.action_feat_template), self.action_map[action]

                
    def _build_action_feat_template(self, topn=-1, thresh=1):
        print('{} types of actions: {}'.format(len(self.action_map), self.action_map.keys()))
        for action, cnt in self.action_cnt.items():
            print('{}\t{}'.format(action, cnt))
        
        if 0 < topn < len(self.action_feat_template) or self.min_action_feat_occur > 1:
            # Construct freq_table
            nrows, ncols = len(self.action_feat_counts), len(self.action_map)
            freq_table = np.zeros((nrows, ncols))
            for (feat, nrow) in self.action_feat_template.items():
                for (action, ncol) in self.action_map.items():
                    freq_table[nrow, ncol] = self.action_feat_counts[feat][action]
            # Feature selection
            fs = FeatureSelector(topn=topn, thresh=thresh, method='frequency')
            print('Original action_feat_template size: {}'.format(len(self.action_feat_template)))
            self.action_feat_template = fs.select(self.action_feat_template, freq_table)
            print('After feature selection, action_feat_template size: {}'.format(len(self.action_feat_template)))
        else:
            print('Action_feat_template size: {}'.format(len(self.action_feat_template)))
        return self.action_feat_template

    def _build_action_map(self, action_sample):
        for feats, action in action_sample:
            try:
                aidx = self.action_map[action]
                self.action_cnt[action] += 1
            except KeyError:
                naction = len(self.action_map)
                self.action_map[action] = naction
                self.action_cnt[action] = 1
        for feats, action in action_sample:
            for feat in feats:
                try:
                    fidx = self.action_feat_template[feat]
                except KeyError:
                    self.action_feat_counts[feat] = defaultdict(float)
                    nfeats = len(self.action_feat_template)
                    self.action_feat_template[feat] = nfeats
                self.action_feat_counts[feat][action] += 1.0

    @staticmethod
    def save_feature_template(feature_template, fname):
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(feature_template, fout)
        print('Save feature template into file: {}'.format(fname))

    @staticmethod
    def save_map(map, fname):
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(map, fout)
        print('Save map into file: {}'.format(fname))

        
        
    @staticmethod
    def read_rst_trees(fdis):
        fmerge = fdis.replace('.dis', '.merge').replace('/discourse/','/merge/')
        if not os.path.isfile(fmerge):
            return None
        rst_tree = RstTree(fdis, fmerge)
        rst_tree.build()
        return rst_tree
