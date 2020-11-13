#!/usr/bin/env python
# coding: utf-8
import prep.preprocess as prep
import mil.training as training
import mil.eval as eval
import cky.cky as cky
import dp.dp as dp
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete', action='store_true',
                        help='Run complete pipeline')
    parser.add_argument('--prep', action='store_true',
                        help='Run data preprocessing')
    parser.add_argument('--mil', action='store_true',
                        help='Run MIL training')
    parser.add_argument('--eval_mil', action='store_true',
                        help='Run MIL evaluation')
    parser.add_argument('--cky', action='store_true',
                        help='Run CKY tree creation')
    parser.add_argument('--dp_preprocess', action='store_true',
                        help='Run discourse parser preprocessing')
    parser.add_argument('--dp_prepare', action='store_true',
                        help='Run discourse parser preparation')
    parser.add_argument('--dp_train', action='store_true',
                        help='Run discourse parser training')
    parser.add_argument('--dp_test', action='store_true',
                        help='Run discourse parser evaluation')
    parser.add_argument('--lemmatize', action='store_true',
                        help='Activate text lemmatization')
    parser.add_argument('--rm_stopwords', action='store_true',
                        help='Activate stopword removal')
    parser.add_argument('--overwrite', action='store_true',
                        help='If activated, existing models, data-files and outputs are overwritten')
    parser.add_argument('--save_dir',
                        default='../out',
                        help='Top level save directory (Default: ../out)')
    parser.add_argument('--train_data',
                        default='../data/train.txt',
                        help='Train data directory (Default: ./data/train.txt)')
    parser.add_argument('--dev_data',
                        default='../data/dev.txt',
                        help='Dev data directory (Default: ./data/dev.txt)')
    parser.add_argument('--eval_data',
                        default='../data/eval.txt',
                        help='Eval data directory (Default: ./data/eval.txt)')
    parser.add_argument('--edu_eval_data',
                        default='../data/edu_eval.txt',
                        help='EDU level eval data directory (Default: ./data/edu_eval.txt)')
    parser.add_argument('--dp_rst_test_data',
                        default='../data/rst/test',
                        help='''Path to the directory with the RST-DT test-set (Default: ../data/rst/test), \
                                for no evaluation enter "None"''')
    parser.add_argument('--dp_instr_test_data',
                        default='../data/instr/test',
                        help='''Path to the directory with the Instructional-DT test-set (Default: ../data/instr/test)\
                                for no evaluation enter "None"''')
    parser.add_argument('--dp_measures',
                        default='orig, rst',
                        help='''Run discourse parsing evaluation on orig(-parseval) and/or rst(-parseval) \
                                (Default: orig, rst (both))''')
    parser.add_argument('--glove_data',
                        default='../data/glove.txt',
                        help='Glove data directory (Default: ./data/glove.txt)')
    parser.add_argument('--pad_token',
                        default='<pad>',
                        help='Define pad token (Default: <pad>)')
    parser.add_argument('--max_doc_len',
                        default=120,
                        type=int,
                        help='Define max document length (Default: 120)')
    parser.add_argument('--max_edu_len',
                        default=30,
                        type=int,
                        help='Define max edu length (Default: 30)')
    parser.add_argument('--batch_size',
                        default=200,
                        type=int,
                        help='Define the batch size (Default: 200)')
    parser.add_argument('--edu_eval_batch_size',
                        default=100,
                        type=int,
                        help='Define the batch size for the EDU level evaluation on SPOT (Default: 100)')
    parser.add_argument('--cpu_workers',
                        default=4,
                        type=int,
                        help='Define the number of threads (Default: 4)')
    parser.add_argument('--cuda',
                        default=0,
                        type=int,
                        help='Define the GPU to use (Default: 0), -1 for cpu')
    parser.add_argument('--gru_hidden_size',
                        default=100,
                        type=int,
                        help='Define the number of neurons in a GRU layer  (Default: 100)')
    parser.add_argument('--classes',
                        default=5,
                        type=int,
                        help='Define the number of output classes (Default: 5)')
    parser.add_argument('--epochs',
                        default=25,
                        type=int,
                        help='Define the number of training epochs (Default: 25)')
    parser.add_argument('--eval_freq',
                        default=1,
                        type=int,
                        help='Frequency of testing on dev set (epochs) (Default: 1)')
    parser.add_argument('--cky_calc',
                        default='avg',
                        help='max | avg (Default: avg)')
    parser.add_argument('--cky_samples',
                        default=None,
                        type=int,
                        help='Number of samples for CKY, None for unlimited (Default: None)')
    parser.add_argument('--cky_doc_len',
                        default=99999,
                        type=int,
                        help='Max length of CKY documents (Default: 99999)')
    parser.add_argument('--reduce_to',
                        default=10,
                        type=int,
                        help='Beam-search beam size')
    parser.add_argument('--stoch', action='store_true',
                        help='Activate stochastic CKY')
    parser.add_argument('--reduction_at',
                        default='merge',
                        help='merge | sentence (Default: merge)')
    parser.add_argument('--nuc_calc_function',
                        default='ternary',
                        help='binary | ternary (Default: ternary)')
    parser.add_argument('--dp_preprocess_txcm',
                        default='txcm',
                        help='''Which subfunctionality of the preprocessing pipeline to execute.\
                                Generate (t)ext, (x)ml, (c)onll, (m)erge''')
    parser.add_argument('--corenlp_dir',
                        default='/ubc/cs/research/nlp/patrickhuber/2_Tools/CoreNLP/stanford-corenlp-full-2018-10-05/',
                        help='absolute path to the CoreNLP main folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_ext = '/mil_model'
    data_ext = '/preprocessed_data'
    out_ext = '/discourse_structures'
    dp_ext = '/discourse_model'
    allowed_mem_percentage = .75

    if args.prep or args.complete:
        # Data preprocessing shortening/padding EDUs and documents for use in model,
        # converting words into GloVe indexes, doing preprocessing (lemmatization, stop-word removal)
        # and generating an HDF5 file for train/dev/test contatining already batched data-slices
        prep.run_preprocessing(args.save_dir, data_ext, dp_ext,
                               model_ext, out_ext,
                               args.train_data, args.dev_data,
                               args.eval_data, args.edu_eval_data,
                               args.glove_data, args.lemmatize,
                               args.rm_stopwords, args.pad_token,
                               args.max_doc_len, args.max_edu_len,
                               args.batch_size, args.cpu_workers,
                               args.overwrite)

    if args.mil or args.complete:
        # Run Miltiple-Instance-Learning (MIL) to learn sentiment augmentations and attention scores
        # necessary for the fine-grained CKY approach
        training.train(args.save_dir, model_ext, data_ext,
                                  args.cpu_workers, args.cuda,
                                  args.batch_size, args.gru_hidden_size,
                                  args.classes, args.epochs, args.eval_freq,
                                  args.overwrite)

    if args.eval_mil or args.complete:
        # Run intermediate evaluations on the MIL generated model
        # Intermediate evaluations are exectuted on document-level and EDU-level.
        # COmpare scores to the original MILNet paper (https://arxiv.org/pdf/1711.09645.pdf)
        eval.test_doc_lvl(args.save_dir, data_ext, model_ext,
                          args.cpu_workers, args.cuda, args.batch_size,
                          args.gru_hidden_size, args.classes)
        eval.test_edu_lvl(args.save_dir, data_ext, model_ext,
                          args.cpu_workers, args.cuda, args.edu_eval_batch_size,
                          args.gru_hidden_size, args.classes)

    if args.cky or args.complete:
        # Run CKY discourse tree generation approach to aggregate fine-grained EDU-level
        # sentiment and attention scores into constituency trees and select the best performing
        # tree-candidate to represent the document. Both versions (EMNLP2019) - full tree-set and
        # (EMNLP2020) - heuristic subset are available through the "reduction_at", "reduce_to", 
        # "nuc_calc_function" and "stoch" parameters
        cky.create_dp_trees(args.save_dir, data_ext, model_ext,
                            out_ext, args.cpu_workers, args.cuda,
                            args.batch_size, args.gru_hidden_size,
                            args.classes, args.cky_calc, args.cky_samples,
                            args.cky_doc_len, args.reduce_to, args.stoch, args.reduction_at, 
                            args.nuc_calc_function, allowed_mem_percentage, args.overwrite)

    if args.dp_preprocess or args.complete:
        # Prepare the discourse parser for the generated discourse structures through
        # CoreNLP annotations and parser-depended pre-processing. Please note that
        # a CoreNLP server is required for this step, which needs to be set up individually
        dp.preprocessing(args.save_dir, out_ext, args.dp_preprocess_txcm, args.corenlp_dir)

    if args.dp_prepare or args.complete:
        # Run the parser preparation step, generating the data_loader file
        dp.preparation(args.save_dir, out_ext, dp_ext)

    if args.dp_train or args.complete:
        # Train the discourse parser on the generated discourse structures
        dp.training(args.save_dir, out_ext, dp_ext)

    if args.dp_test or args.complete:
        # Test the discourse parser on the generated discourse structures
        dp.testing(args.save_dir, dp_ext, args.dp_rst_test_data, args.dp_instr_test_data, args.dp_measures)
