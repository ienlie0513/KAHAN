import sys
import os
import json
from datetime import datetime
import argparse

import fcntl
import csv

from ast import literal_eval

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from gensim.models.keyedvectors import KeyedVectors
from wikipedia2vec import Wikipedia2Vec

from util.train_util import trainIters
from util.util import show_result, plot_confusion_matrix, calculate_metrics
from util.datahelper import KaDataset, get_preprocessed_data
from IKAHAN import IKAHAN, train, evaluate

now = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

class Timer:
    def __init__(self):
        self.start = datetime.now()

    def get_time(self):
        t_diff = (datetime.now() - self.start).total_seconds()
        mins, secs = divmod(t_diff, 60)
        return '{:02d}:{:02d}:{:02d}'.format(int(mins // 60), int(mins % 60), int(secs))

def log_and_print(s, log):
    print(s)
    log.write(s + '\n')

def init_archive(config, model_type, dimred_method, fusion_method, hid, exclude_with_no_image, ihan, kahan, deep_classifier, clip, ent_att):
    # create log file
    root = ''
    hid = '' if not hid else hid
    if clip:
        if ent_att:
            root = './model_ckpts/{}_clip_ea_{}_{}'.format(config['data_source'], fusion_method, now)
        else:
            root = './model_ckpts/{}_clip_{}_{}'.format(config['data_source'], fusion_method, now)
    if ihan:
        if ent_att:
            root = './model_ckpts/{}_{}_ihan_ea_{}_{}'.format(config['data_source'], model_type, fusion_method, now)
        else:
            root = './model_ckpts/{}_{}_ihan_{}_{}'.format(config['data_source'], model_type, fusion_method, now)
    elif kahan:
        root = './model_ckpts/{}_kahan_{}'.format(config['data_source'], now)
    elif exclude_with_no_image:
        root = './model_ckpts/{}_excluded_no_img_cases_{}_{}{}_{}_{}'.format(config['data_source'], model_type, dimred_method, hid, fusion_method, now)
    else:
        root = './model_ckpts/{}_{}_{}{}_{}_{}'.format(config['data_source'], model_type, dimred_method, hid, fusion_method, now)

    if deep_classifier:
        root = ''.join([root, '_deep_classifier'])
        
    if not os.path.exists(root):
        os.makedirs(root)

    log = open('{}/log.txt'.format(root), 'a')
    json.dump(config, log)
    log.write('\n')

    # create img dir
    img_dir = '{}/img'.format(root)
    os.makedirs(img_dir)

    # create ckpt dir
    ckpt_dir = '{}/ckpts'.format(root)
    os.makedirs(ckpt_dir)

    return log, img_dir, ckpt_dir

def write_result_to_csv(row, results_csv):
    file_exists = os.path.exists(results_csv)

    with open(results_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            fcntl.flock(csvfile, fcntl.LOCK_EX)
            csv_writer.writerow(header)
            fcntl.flock(csvfile, fcntl.LOCK_UN)
            
        fcntl.flock(csvfile, fcntl.LOCK_EX)
        csv_writer.writerow(row)
        fcntl.flock(csvfile, fcntl.LOCK_UN)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', type=str, default='vgg19')
    parser.add_argument('--dimred', type=str, default='maxpooling')
    parser.add_argument('--fusion', type=str, default='cat')
    parser.add_argument('--hid', type=str, default=None)
    parser.add_argument('--platform', type=str, default='politifact_v4')
    parser.add_argument('--exclude_with_no_image', action='store_true')
    parser.add_argument('--kahan', action='store_true')
    parser.add_argument('--deep_classifier', action='store_true')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--ihan', action='store_true')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--ent_att', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--results_csv', type=str, default='results.csv')
    args = parser.parse_args()

    # load config
    config = None
    if args.platform.startswith('politifact'):
        config = json.load(open('./config_p.json'))
    elif args.platform.startswith('gossipcop'):
        config = json.load(open('./config_g.json'))
    else:
        raise ValueError('Invalid platform argument')

    dimred_params = {
        'method': args.dimred,
        'embed_size': config['image_preprocessing']['{}_embed_size'.format(args.cnn)],
        'kernel_size': config['image_preprocessing']['{}_dimred_kernel_size'.format(args.cnn)],
        'hid_layers': literal_eval(args.hid) if args.hid else []
    }

    log, img_dir, ckpt_dir = init_archive(config, args.cnn, args.dimred, args.fusion, args.hid, args.exclude_with_no_image, args.ihan, args.kahan, args.deep_classifier, args.clip, args.ent_att)

    # load data
    contents, comments, entities, clip_entities, images, labels = get_preprocessed_data(config['data_dir'], config['data_source'], args.cnn, args.exclude_with_no_image, args.kahan, args.ihan, args.clip)
    
    # load word2vec, wiki2vec model and add unk vector
    word2vec_cnt = KeyedVectors.load_word2vec_format(config['word2vec_cnt'])
    word2vec_cnt.add_vector('_unk_', np.average(word2vec_cnt.vectors, axis=0))
    word2vec_cnt.add_vector('_pad_', np.zeros(config['word2vec_dim']))
    word2vec_cmt = KeyedVectors.load_word2vec_format(config['word2vec_cmt'])
    word2vec_cmt.add_vector('_unk_', np.average(word2vec_cmt.vectors, axis=0))
    word2vec_cmt.add_vector('_pad_', np.zeros(config['word2vec_dim']))
    wiki2vec = Wikipedia2Vec.load(config['wiki2vec'])

    timer = Timer()

    SEEDS = [i for i in range(args.seeds)]
    seed_avg_total_scores = []
    seed_avg_last_scores = []

    results_csv = args.results_csv
    header = ['DateTime', 'Platform', 'Model', 'Method', 'Fusion', 'Seeds', 'Folds', 'Epochs', 'Time', 'Accuracy', 'Precision', 'Recall', 'Micro F1', 'Macro F1']
    write_result_to_csv(header, results_csv)

    for seed in SEEDS:
        print ('Seed %d start at %s' % (seed, datetime.now().strftime('%Y_%m_%d %H:%M:%S')))
        log.write('Seed %d start at %s\n' % (seed, datetime.now().strftime('%Y_%m_%d %H:%M:%S')))

        # cross validation
        scores = []

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print ('Fold %d start at %s' % (fold, datetime.now().strftime('%Y_%m_%d %H:%M:%S')))
            log.write('Fold %d start at %s\n' % (fold, datetime.now().strftime('%Y_%m_%d %H:%M:%S')))

            x_train, x_val = contents[train_idx], contents[test_idx]
            c_train, c_val = comments[train_idx], comments[test_idx]
            e_train, e_val = entities[train_idx], entities[test_idx]
            ce_train, ce_val = clip_entities[train_idx], clip_entities[test_idx]
            i_train, i_val = images[train_idx], images[test_idx]
            y_train, y_val = labels[train_idx], labels[test_idx]

            print('train size: %d, val size: %d' % (len(y_train), len(y_val)))

            trainset = KaDataset(x_train, c_train, e_train, ce_train, i_train, y_train)
            validset = KaDataset(x_val, c_val, e_val, ce_train, i_val, y_val)

            # training
            model = IKAHAN(config['num_class'], word2vec_cnt, word2vec_cmt, dimred_params, args.kahan, args.deep_classifier, args.fusion, args.ihan, args.clip, args.ent_att, config['image_preprocessing']['clip_embed_size'], config['word2vec_dim'], config['hid_size'], max_sent=config['max_sent'], max_len=config['max_len'], max_cmt=['max_cmt'], dropout=config['dropout'])
            train_accs, test_accs, train_losses, test_losses, model_name = trainIters(model, trainset, validset, train, evaluate,
                epochs=args.epochs, learning_rate=config['lr'], batch_size=config['batch_size'], weight_decay=config['weight_decay'],
                save_info=(fold, ckpt_dir), print_every=config['print_every'], device=config['device'], log=log)
            show_result(train_accs, test_accs, train_losses, test_losses, save=(fold, img_dir, seed))

            # evaluate
            model = IKAHAN(config['num_class'], word2vec_cnt, word2vec_cmt, dimred_params, args.kahan, args.deep_classifier, args.fusion, args.ihan, args.clip, args.ent_att, config['image_preprocessing']['clip_embed_size'], config['word2vec_dim'], config['hid_size'], max_sent=config['max_sent'], max_len=config['max_len'], max_cmt=['max_cmt'], dropout=config['dropout']).to(config['device'])
            model.load_state_dict(torch.load(model_name))
            _, acc, predicts, targets = evaluate(model, validset, device=config['device'])
            acc, precision, recall, microf1, macrof1 = calculate_metrics(acc, targets, predicts, log=log)
            scores.append([acc, precision, recall, microf1, macrof1])
            plot_confusion_matrix(targets, predicts, config['num_class'], save=(fold, img_dir, seed))

        # calculate average score
        avg_total_score = np.mean(scores, axis=0)
        seed_avg_total_scores.append(avg_total_score)
        # calculate last score
        last_score = scores[-1]
        seed_avg_last_scores.append(last_score)

        # log average score of seed to file and print to console
        log_and_print(
            'Average score for seed {}: acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, micro f1: {:.4f}, macro f1: {:.4f}\n'.format(seed, *avg_total_score),
            log
        )
        # log last score of seed to file and print to console
        log_and_print(
            'Last score for seed {}: acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, micro f1: {:.4f}, macro f1: {:.4f}\n'.format(seed, *last_score),
            log
        )

    # calculate and log elapsed time
    elapsed_time = timer.get_time()
    log_and_print('Elapsed time: {}'.format(elapsed_time), log)

    # calculate average score
    avg_total_score = np.mean(seed_avg_total_scores, axis=0)
    avg_last_score = np.mean(seed_avg_last_scores, axis=0)
    
    result_row_total = [now, args.platform, args.cnn, args.dimred, args.fusion, args.seeds, args.folds, args.epochs, *avg_total_score]
    write_result_to_csv(result_row_total, results_csv)
    result_row_last = [now, args.platform, args.cnn, args.dimred, args.fusion, args.seeds, args.folds, args.epochs, *avg_last_score]
    write_result_to_csv(result_row_last, results_csv)

    # log average score for all seeds to file and print to console
    log_and_print(
        'Average total score for all seeds: acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, micro f1: {:.4f}, macro f1: {:.4f}\n'.format(*avg_total_score),
        log
    )
    # log last score for all seeds to file and print to console
    log_and_print(
        'Average last score for all seeds: acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, micro f1: {:.4f}, macro f1: {:.4f}\n'.format(*avg_last_score),
        log
    )

    log.close() 