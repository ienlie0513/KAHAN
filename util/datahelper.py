import re
import string 
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from wikipedia2vec import Wikipedia2Vec

from PIL import Image

from torchvision.models import vgg19, VGG19_Weights

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
    df = pd.read_csv("{}/{}_no_ignore_en_cap.tsv".format(data_dir, data_source), sep='\t') 
    df = df.fillna('')
    contents = []
    comments = []
    entities = []
    images = []
    labels = []

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
            path = data_dir + '/news_images/' + '/real/' + data_source + '_' + str(df.id[idx]) + '.jpg'
        else:
            path = data_dir + '/news_images/' + '/fake/' + data_source + '_' + str(df.id[idx]) + '.jpg'

        if os.path.exists(path):
            images.append(Image.open(path))
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


class KaDataset(data.Dataset):
    """
        This Dataset class is for FakeNewsNet data
        it tokenize the news content and comments, convert each token into word index, padding into max length,
        get entity and claim embed, finally return the tensor of word index and label 
        variables
            sb_type: 0 time-based, 1 count-based subevent build
            max_len: max number of tokens in a sentence
            max_sen: max number of sentences in a document 
            max_ent: max number of entities in a news 
            M: number of subevents if count-based
            max_cmt: max number of comments in a subevent
            intervals: range of time index, for building time-based subevents
    """
    def __init__(self, contents, comments, entities, images, labels, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec, sb_type, max_len=60, max_sent=30, max_ent=100, M=5, max_cmt=50, intervals=100):
        self.contents = contents
        self.comments = comments
        self.entities = entities
        self.images = images
        self.labels = labels

        self.sb_type = sb_type
        
        self.max_len = max_len
        self.max_sent = max_sent
        self.max_ent = max_ent
        self.M = M
        self.max_cmt = max_cmt
        self.intervals = intervals

        self.claim_dict = claim_dict
        self.word2vec_cnt = word2vec_cnt
        self.word2vec_cmt = word2vec_cmt
        self.wiki2vec = wiki2vec
        

    def __len__(self):
        return len(self.labels)

    # convert entity into embed
    def _get_entity_embed(self, ent):
        if self.wiki2vec.get_entity(ent.replace("_", " ")):
            return self.wiki2vec.get_entity_vector(ent.replace("_", " "))
        else:
            return None
    
    # convert entity claim into avg embed
    def _get_claim_embed(self, ent):
        if ent in self.claim_dict:
            claims = self.claim_dict[ent]
        else: 
            return None
        clm_embed = []
        for clm in claims:
            # get wiki2vec
            clm = clm.split(":")
            clm = clm[1] if len(clm)>1 else clm[0]
            
            if self.wiki2vec.get_entity(clm):
                clm_embed.append(self.wiki2vec.get_entity_vector(clm))
            else:
                # get word2vec
                token = clm.split(" ")
                token = [tk.lower() for tk in token if self.wiki2vec.get_word(tk.lower())]
                if token:
                    clm_embed.extend([self.wiki2vec.get_word_vector(tk) for tk in token])
        
        # average claim vector, pad if no word2vec or wiki2vec found 
        if clm_embed:
            return np.average(clm_embed, axis=0)
        else:
            return None

    # given entities in a sentence, return MAXPOOL entity embed and claim embed
    def _knowledge_preprocesss(self, ents):
        ent_vec = []
        for ent in ents:
            ent_embed = self._get_entity_embed(ent)
            clm_embed = self._get_claim_embed(ent)

            if isinstance(ent_embed, np.ndarray) and isinstance(clm_embed, np.ndarray):
                ent_vec.append(np.concatenate((ent_embed, clm_embed), axis=None))

        lk = len(ent_vec) if len(ent_vec)<self.max_ent else self.max_ent
        lk = lk if lk>0 else 1
        
        ent_vec = ent_vec[:self.max_ent]
        if ent_vec:
            ent_vec = np.pad(ent_vec, ((0, self.max_ent-len(ent_vec)),(0, 0)))
        else:
            ent_vec = np.full((self.max_ent, self.word2vec_cnt.vector_size), self.word2vec_cnt['_pad_'])
            ent_vec = np.concatenate((ent_vec, ent_vec), axis=1)

        return ent_vec, lk

    # reorder word2vec to make sure pad is behind
    def _news_reorder(self, word_vec, ls):
        pad_idx = 0
        for i, s in enumerate(ls):
            if s != 0:
                if pad_idx < i:
                    word_vec[pad_idx], word_vec[i] = word_vec[i], word_vec[pad_idx]
                    ls[pad_idx], ls[i] = ls[i], ls[pad_idx]
                pad_idx += 1
        return word_vec, ls

    # return preprocessed news content in sentence level, (max_sentence, max_length)
    def _news_content_preprocess(self, content):
        # convert words to vectors
        content = [re.sub('[^a-zA-Z]', ' ', sentence) for sentence in sent_tokenize(content)]
        content = [word_tokenize(sentence.lower()) for sentence in content]
        content = [[w for w in sentence if w not in stopwords.words('english')] for sentence in content]

        # convert word into index of word embedding
        word_vec = [[self.word2vec_cnt.key_to_index[w] if w in self.word2vec_cnt.key_to_index else self.word2vec_cnt.key_to_index['_unk_'] for w in sentence] for sentence in content if sentence]

        # calculate sentence lengths
        ls = [len(sentence) if len(sentence)<self.max_len else self.max_len for sentence in word_vec]

        word_vec, ls = self._news_reorder(word_vec, ls)

        # pad sentence and calculate number of sentence
        ls = ls[:self.max_sent]
        ls = np.pad(ls, (0, self.max_sent-len(ls))).astype(int)
        ln = np.count_nonzero(ls)

        # if no sent, get one pad sent
        if ln == 0:
            ln = 1
            ls[0] = 1

        # token padding
        word_vec = [sentence[:self.max_len] for sentence in word_vec]
        word_vec = [np.pad(sentence, ((0, self.max_len-len(sentence)))).astype(int) for sentence in word_vec]

        # sentence padding 
        word_vec = word_vec[:self.max_sent]
        # if empty word_vec
        if word_vec:
            word_vec = np.pad(word_vec, ((0, self.max_sent-len(word_vec)), (0, 0)))
        else:
            word_vec = np.full((self.max_sent, self.max_len), self.word2vec_cnt.key_to_index['_pad_'], dtype=int)

        return word_vec, ln, ls

     # reorder word2vec to make sure pad is behind
    def _comment_reorder(self, results):
        word_vec = [sb[0] for sb in results]
        lsb = [sb[1] for sb in results]
        lc = [sb[2] for sb in results]

        pad_idx = 0
        for i, s in enumerate(lsb):
            if s != 0:
                if pad_idx < i:
                    word_vec[pad_idx], word_vec[i] = word_vec[i], word_vec[pad_idx]
                    lsb[pad_idx], lsb[i] = lsb[i], lsb[pad_idx]
                    lc[pad_idx], lc[i] = lc[i], lc[pad_idx]
                pad_idx += 1
        return word_vec, lsb, lc

    # given comments of the subevent, return preprocessed comments, (max_cmt, max_len)
    def _comment_preprocess(self, comments):
        # if empty subevent
        if comments == []:
            return np.full((self.max_cmt, self.max_len), self.word2vec_cmt.key_to_index['_pad_'], dtype=int), 0, [0]*self.max_cmt

        comments = [cmt[0] for cmt in comments]
        comments = [re.sub('[^a-zA-Z]', ' ', cmt) for cmt in comments]
        comments = [word_tokenize(cmt.lower()) for cmt in comments]
        comments = [[w for w in cmt if w not in stopwords.words('english')] for cmt in comments]
        
        # convert word into index of word embedding
        word_vec = [[self.word2vec_cmt.key_to_index[w] if w in self.word2vec_cmt.key_to_index else self.word2vec_cmt.key_to_index['_unk_'] for w in cmt] for cmt in comments if cmt]

        # calculate number of comments and comments lengths
        lsb = len(word_vec) if len(word_vec)<self.max_cmt else self.max_cmt
        lc = [len(cmt) if len(cmt)<self.max_len else self.max_len for cmt in word_vec]
        lc = lc[:self.max_cmt]
        lc = np.pad(lc, (0, self.max_cmt-len(lc))).astype(int)

        # token padding
        word_vec = [cmt[:self.max_len] for cmt in word_vec]
        word_vec = [np.pad(cmt, ((0, self.max_len-len(cmt)))) for cmt in word_vec]

        # comment padding 
        word_vec = word_vec[:self.max_cmt]
        # if empty word_vec
        if word_vec:
            word_vec = np.pad(word_vec, ((0, self.max_cmt-len(word_vec)), (0, 0)))
        else:
            word_vec = np.full((self.max_cmt, self.max_len), self.word2vec_cmt.key_to_index['_pad_'], dtype=int)
        
        return word_vec, lsb, lc

    # given a list of comments, return build time series, (M, max_comment, max_length, word embedding size)
    def _build_subevents(self, comments):
        # count-based subevent build
        if self.sb_type:
            if len(comments) >= self.M*self.max_cmt:
                subevents = [comments[(i-1)*self.max_cmt:i*self.max_cmt] for i in range(1, self.M+1)]
            else:
                avg = int(len(comments)/self.M)
                if avg > 0:
                    subevents = [comments[(i-1)*avg:i*avg] for i in range(1, self.M)]
                    subevents.append(comments[avg*(self.M-1):])
                else:
                    subevents = [comments]
                    subevents.extend([[]] * (self.M-1))
        else:
            # initialization
            l = int(self.intervals/self.M)
            N = self.M
            ordered_comments = [[] for i in range(self.intervals)]
            for cmt in comments:
                ordered_comments[cmt[1]].append(cmt)
            
            last_subevents = []
            while(True):
                subevents = []
                for i in range(N):
                    sb = [cmt for t_cmt in ordered_comments[i*l:(i+1)*l] for cmt in t_cmt]
                    if len(sb) >= 1:
                        subevents.append(sb[:self.max_cmt])
                    
                if len(subevents) >= self.M:
                    subevents = subevents[:self.M]
                    break
                elif len(subevents) <= len(last_subevents):
                    last_subevents.extend([[]]*self.M)
                    subevents = last_subevents[:self.M]
                    break
                else:
                    # shorten length of subevents
                    l = int(0.5 * l)
                    N = 2 * N
                    last_subevents = subevents

        results = [self._comment_preprocess(sb) for sb in subevents]

        le = np.count_nonzero([sb[1] for sb in results])
        word_vec, lsb, lc = self._comment_reorder(results)

        return word_vec, le, lsb, lc

    # create VGG19 embedding of the PIL image object and return
    def _image_preprocess(self, image):
        # initialize the weight transform
        weigths = VGG19_Weights.DEFAULT
        preprocess = weigths.transforms()
        # apply the transform to the image
        image_transformed = preprocess(image) if image else None
        # initialize the model
        model = vgg19(weights=weigths)
        # select the layer to extract features from
        layer = model._modules.get('avgpool')
        # set model to evaluation mode
        model.eval()
        # create empty embedding
        embedding = torch.zeros(25088)
        if image_transformed is not None:
            # create a function that will copy the output of a layer
            def copy_data(m, i, o):
                embedding.copy_(o.flatten())
            # attach that function to our selected layer
            h = layer.register_forward_hook(copy_data)
            # run the model on our transformed image
            model(image_transformed.unsqueeze(0))
            # detach our copy function from the layer
            h.remove()
        # performing max pooling on the embedding to reduce the size
        embedding = F.max_pool1d(embedding.unsqueeze(0), 125).squeeze(0)
        # return the feature vector
        return embedding

    # return data ((contents, comments), label)
    def __getitem__(self, index):
        content = self.contents[index]
        comment = self.comments[index]
        entity = self.entities[index]
        image = self.images[index]
        label = self.labels[index]

        content_vec, ln, ls = self._news_content_preprocess(content)
        comment_vec, le, lsb, lc = self._build_subevents(comment)
        ent_vec, lk = self._knowledge_preprocesss(entity)
        img_vec = self._image_preprocess(image)
        
        return ((torch.tensor(content_vec), torch.tensor(ln), torch.tensor(ls)), (torch.tensor(comment_vec), torch.tensor(le), torch.tensor(lsb), torch.tensor(lc)), (torch.tensor(ent_vec), torch.tensor(lk)), img_vec), torch.tensor(label)


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