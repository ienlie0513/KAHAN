import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torchvision.transforms as transforms

from gensim.models.keyedvectors import KeyedVectors
from wikipedia2vec import Wikipedia2Vec

from PIL import Image

# get entity claim from preprocessed tcv file
def get_entity_claim(data_dir, data_source):
    df = pd.read_csv("{}/{}_no_ignore_clm.tsv".format(data_dir, data_source), sep='\t') 
    df = df.fillna('')

    claim_dict = {}
    for idx in range(df.entity.shape[0]):
        clms = [clm for clm in df.claims[idx].split('||')]
        claim_dict[df.entity[idx]] = clms

    return claim_dict

# get news content and comments from preprocessed tcv file
def get_data(data_dir, data_source):
    df = pd.read_csv("{}/{}_no_ignore_en.tsv".format(data_dir, data_source), sep='\t') 
    df = df.fillna('')
    contents = []
    comments = []
    entities = []
    images = []
    labels = []

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    for idx in range(df.id.shape[0]):
        # load news content
        text = df.text[idx]
        text = text.encode('ascii', 'ignore').decode('utf-8')
        contents.append(text)

        # load user comments
        com_text = df.comments[idx]
        com_text = com_text.encode("ascii", "ignore").decode('utf-8')
        tmp_comments = []

        buff = ''
        for ct in com_text.split('::'):
            t = ct.split('<>')
            # handle if not well split
            if len(t) == 1:
                buff = buff+ct
            else:
                tmp_comments.append((buff+t[0], int(t[1])))
                buff = ''
        comments.append(tmp_comments)

        # load entities
        ens = [en for ens in df.entities[idx].split('||') for en in ens.split(' ') if en != '']
        entities.append(ens)

        # load images
        if df.label[idx] == 1:
            path = data_dir + '/' + data_source + '/news_images/' + 'real/politifact_' + str(df.id[idx]) + '.jpg'
        else:
            path = data_dir + '/' + data_source + '/news_images/' + 'fake/politifact_' + str(df.id[idx]) + '.jpg'

        if os.path.exists(path):
            with Image.open(path) as img:
                images.append(transform(img))
        else:
            images.append(None)

        # load labels
        labels.append(df.label[idx])

    contents = np.asarray(contents)
    comments = np.asarray(comments)
    entities = np.asarray(entities)
    images = np.asarray(images)
    labels = np.asarray(labels)

    return contents, comments, entities, images, labels


def get_preprocessed_data(data_dir, data_source, model_type, exclude_with_no_image=False, only_newscontent=False, use_ihan=False):
    loaded_data = None
    try:
        path = ''
        if only_newscontent and exclude_with_no_image:
            path = '{}/{}/preprocessed_only_newscontent_exclude_with_no_image.pt'.format(data_dir, data_source)
        elif only_newscontent:
            path = '{}/{}/preprocessed_only_newscontent.pt'.format(data_dir, data_source)
        else:
            path = '{}/{}/preprocessed_{}.pt'.format(data_dir, data_source, model_type)
        loaded_data = torch.load(path)
    except:
        print ("Preprocessed data not found. Please run preprocess.py first.")
        exit()

    contents = []
    comments = []
    entities = []
    images = []
    labels = []

    for i in range(len(loaded_data['images'])):
        image_repr = loaded_data['images'][i]
        if only_newscontent and exclude_with_no_image:
            contents.append(loaded_data['contents'][i])
            comments.append(loaded_data['comments'][i])
            entities.append(loaded_data['entities'][i])
            images.append(image_repr.numpy())
            labels.append(loaded_data['labels'][i])
        elif exclude_with_no_image:
            if torch.sum(image_repr) != 0:
                contents.append(loaded_data['contents'][i])
                comments.append(loaded_data['comments'][i])
                entities.append(loaded_data['entities'][i])
                images.append(image_repr.numpy())
                labels.append(loaded_data['labels'][i])
        else:
            contents.append(loaded_data['contents'][i])
            comments.append(loaded_data['comments'][i])
            entities.append(loaded_data['entities'][i])
            images.append(image_repr.numpy())
            labels.append(loaded_data['labels'][i])

    print('length of contents: ', len(contents))
    
    contents = np.asarray(contents)
    comments = np.asarray(comments)
    entities = np.asarray(entities)
    images = np.asarray(images)
    labels = np.asarray(labels)

    if use_ihan:
        ihan_images = np.zeros((len(images), 16, 16, 100), dtype=np.float32)
        for i in range(len(images)):
            # Split the images into 16 "sentences" and 16 "words" of length 98
            split_image = np.reshape(images[i], (-1, 16, 98))
            # Add padding of zeroes to the "word" vectors
            split_image = np.pad(split_image, ((0,0), (0,0), (0, 2)), 'constant')
            ihan_images[i] = split_image
        images = ihan_images

    return contents, comments, entities, images, labels


class KaDataset(data.Dataset):
    """
        This Dataset class is for FakeNewsNet data
    """
    def __init__(self, contents, comments, entities, images, labels):
        self.contents = contents
        self.comments = comments
        self.entities = entities
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    # return data
    def __getitem__(self, index):
        content = self.contents[index]
        comment = self.comments[index]
        entity = self.entities[index]
        image = self.images[index]
        label = self.labels[index] 

        cnt, ln, ls = content
        cmt, le, lsb, lc = comment
        ent, lk = entity
        
        return ((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk), image), label


if __name__ == '__main__':
    word2vec_cnt = KeyedVectors.load_word2vec_format('../word2vec/glove-wiki-gigaword-100')
    word2vec_cnt.add_vector('_unk_', np.average(word2vec_cnt.vectors, axis=0))
    word2vec_cnt.add_vector('_pad_', np.zeros(100))
    word2vec_cmt = KeyedVectors.load_word2vec_format('../word2vec/glove-twitter-100')
    word2vec_cmt.add_vector('_unk_', np.average(word2vec_cmt.vectors, axis=0))
    word2vec_cmt.add_vector('_pad_', np.zeros(100))
    wiki2vec = Wikipedia2Vec.load("../word2vec/enwiki_20180420_100d.pkl")

    print ("KaDataset")
    contents, comments, entities, images, labels = get_data("./data", "politifact", "KaDataset")
    claim_dict = get_entity_claim("./data", "politifact")
    dataset = KaDataset(contents, comments, entities, images, labels, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec, sb_type=0)
    print ("politifact dataset: ", len(dataset))
    ((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk)), lb = dataset[1][0], dataset[1][1]
    print (cnt.shape, ln, ls)
    print (ent.shape, lk)
    print (cmt.shape, le, lsb, lc)
    print (lb, end='\n---\n')

    contents, comments, entities, images, labels = get_data("./data", "gossipcop", "KaDataset")
    claim_dict = get_entity_claim("./data", "politifact")
    dataset = KaDataset(contents, comments, entities, images, labels, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec, sb_type=0)
    print ("gossipcop dataset: ", len(dataset))
    ((cnt, ln, ls), (cmt, le, lsb, lc), (ent, lk)), lb = dataset[1][0], dataset[1][1]
    print (cnt.shape, ln, ls)
    print (ent.shape, lk)
    print (cmt.shape, le, lsb, lc)
    print (lb, end='\n---\n')