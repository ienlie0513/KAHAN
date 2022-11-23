import os
import json
# import importlib
from datetime import datetime
import pause

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from gensim.models.keyedvectors import KeyedVectors
from wikipedia2vec import Wikipedia2Vec

from util.train_util import trainIters
from util.util import show_result, plot_confusion_matrix, calculate_metrics, Progressor
from util.datahelper import KaDataset, get_data, get_entity_claim
from KAHAN import KAHAN, train, evaluate

def init_archive(config):
    # create log file
    now = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    root = "./model_ckpts/{}_{}".format(config['data_source'], now)
    if not os.path.exists(root):
        os.makedirs(root)

    log = open("{}/log.txt".format(root), 'a')
    json.dump(config, log)
    log.write('\n')

    # create img dir
    img_dir = "{}/img".format(root)
    os.makedirs(img_dir)

    # create ckpt dir
    ckpt_dir = "{}/ckpts".format(root)
    os.makedirs(ckpt_dir)

    return log, img_dir, ckpt_dir


if __name__ == '__main__':
    config = json.load(open("./config_p.json"))
    # config = json.load(open("./config_g.json"))

    # load word2vec, wiki2vec model and add unk vector
    word2vec_cnt = KeyedVectors.load_word2vec_format(config['word2vec_cnt'])
    word2vec_cnt.add_vector('_unk_', np.average(word2vec_cnt.vectors, axis=0))
    word2vec_cnt.add_vector('_pad_', np.zeros(config['word2vec_dim']))
    word2vec_cmt = KeyedVectors.load_word2vec_format(config['word2vec_cmt'])
    word2vec_cmt.add_vector('_unk_', np.average(word2vec_cmt.vectors, axis=0))
    word2vec_cmt.add_vector('_pad_', np.zeros(config['word2vec_dim']))
    wiki2vec = Wikipedia2Vec.load(config['wiki2vec'])

    log, img_dir, ckpt_dir = init_archive(config)

    # get data and split into 5-fold
    contents, comments, entities, captions, labels = get_data(config['data_dir'], config['data_source'])
    claim_dict = get_entity_claim(config['data_dir'], config['data_source'])
    skf = StratifiedKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print ("Fold %d start at %s" % (fold, datetime.now().strftime("%Y_%m_%d %H:%M:%S")))
        log.write("Fold %d start at %s\n" % (fold, datetime.now().strftime("%Y_%m_%d %H:%M:%S")))
        x_train, x_val = contents[train_idx], contents[test_idx]
        c_train, c_val = comments[train_idx], comments[test_idx]
        e_train, e_val = entities[train_idx], entities[test_idx]
        cap_train, cap_val = captions[train_idx], captions[test_idx]
        y_train, y_val = labels[train_idx], labels[test_idx]

        trainset = KaDataset(x_train, c_train, e_train, cap_train, y_train, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec,
            sb_type=config['sb_type'], max_len=config['max_len'], max_sent=config['max_sent'], max_ent=config['max_ent'], M=config['M'], max_cmt=config['max_cmt'])
        validset = KaDataset(x_val, c_val, e_val, cap_val, y_val, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec,
            sb_type=config['sb_type'], max_len=config['max_len'], max_sent=config['max_sent'], max_ent=config['max_ent'], M=config['M'], max_cmt=config['max_cmt'])

        # training
        model = KAHAN(config['num_class'], word2vec_cnt, word2vec_cmt, config['word2vec_dim'], config['hid_size'], max_sent=config['max_sent'], dropout=config['dropout'])
        train_accs, test_accs, train_losses, test_losses, model_name = trainIters(model, trainset, validset, train, evaluate,
            epochs=config['ep'], learning_rate=config['lr'], batch_size=config['batch_size'], weight_decay=config['weight_decay'],
            save_info=(fold, ckpt_dir), print_every=config['print_every'], device=config['device'], log=log)
        show_result(train_accs, test_accs, train_losses, test_losses, save=(fold, img_dir))

        # evaluate
        model = KAHAN(config['num_class'], word2vec_cnt, word2vec_cmt, config['word2vec_dim'], config['hid_size'], max_sent=config['max_sent'], dropout=config['dropout']).to(config['device'])
        model.load_state_dict(torch.load(model_name))
        _, acc, predicts, targets = evaluate(model, validset, device=config['device'])
        calculate_metrics(acc, targets, predicts, log=log)
        plot_confusion_matrix(targets, predicts, config['num_class'], save=(fold, img_dir))
    
    log.close() 