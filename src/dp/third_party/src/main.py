import argparse
import gzip
import pickle
import scipy
from .data_helper import DataHelper
from .eval.evaluation import Evaluator
from .models.classifiers import ActionClassifier
from .models.parser import RstParser

def train_model(data_helper, data_dir, save_path_model):
     # initialize the parser
    action_clf = ActionClassifier(data_helper.action_feat_template, data_helper.action_map)
    #relation_clf = RelationClassifier(data_helper.relation_feat_template_level_0,
    #                                  data_helper.relation_feat_template_level_1,
    #                                  data_helper.relation_feat_template_level_2,
    #                                  data_helper.relation_map)
    relation_clf = None
    rst_parser = RstParser(action_clf, relation_clf)
    # train action classifier
    action_fvs, action_labels = list(zip(*data_helper.gen_action_train_data(data_dir)))
    rst_parser.action_clf.train(scipy.sparse.vstack(action_fvs), action_labels)
    # train relation classifier
    #for level in [0, 1, 2]:
    #    relation_fvs, relation_labels = list(zip(*data_helper.gen_relation_train_data(level)))
    #    print('{} relation samples at level {}.'.format(len(relation_labels), level))
    #    rst_parser.relation_clf.train(scipy.sparse.vstack(relation_fvs), relation_labels, level)
    rst_parser.save(model_dir=save_path_model)


def prepare(dis_dir, save_path_data_helper):
    # Use brown clusters
    with gzip.open("./dp/third_party/resources/bc3200.pickle.gz") as fin:
        print('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    data_helper = DataHelper(max_action_feat_num=9999999, max_relation_feat_num=999999,
                             min_action_feat_occur=1, min_relation_feat_occur=1,
                             brown_clusters=brown_clusters)
    # Create training data
    data_helper.create_data_helper(data_dir=dis_dir)
    data_helper.save_data_helper(save_path_data_helper)


def train(train_dir, save_path_model, save_path_data_helper):
    # Use brown clusters
    with gzip.open("./dp/third_party/resources/bc3200.pickle.gz") as fin:
        print('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    data_helper = DataHelper(max_action_feat_num=9999999, max_relation_feat_num=999999,
                             min_action_feat_occur=1, min_relation_feat_occur=1,
                             brown_clusters=brown_clusters)

    # Train
    data_helper.load_data_helper(save_path_data_helper)
    train_model(data_helper, train_dir, save_path_model)


def test(dp_rst_test_data, save_path_model, measure):
    # Evaluate models on the RST-DT test set
    with gzip.open("./dp/third_party/resources/bc3200.pickle.gz") as fin:
        #print('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    evaluator = Evaluator(model_dir=save_path_model)
    evaluator.eval_parser(dp_rst_test_data, measure, report=True, bcvocab=brown_clusters, draw=False)
