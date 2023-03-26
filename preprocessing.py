import torch
import numpy as np
import open_clip

def clip_ent_claims_encoder(ent_clms, tokenizer, model):
    text = tokenizer(ent_clms)

    text_features = None
    ent_embed, clm_embed = None, None

    with torch.no_grad():
        text_features = model.encode_text(text)

    if len(ent_clms) > 2:
        ent_embed = text_features[0].unsqueeze(0)
        clm_embed = torch.mean(text_features[1:], dim=0, keepdim=True)
    elif len(ent_clms) == 2:
        ent_embed, clm_embed = text_features[0], text_features[1]
    elif len(ent_clms) == 1:
        ent_embed, clm_embed = text_features[0], text_features[0]

    return ent_embed, clm_embed #np.concatenate((ent_embed, clm_embed), axis=None)

if __name__ == '__main__':
    import re
    import json
    import argparse

    from multiprocessing import Pool

    import torch.nn as nn
    from torchvision.models import vgg19, resnet50, VGG19_Weights, ResNet50_Weights
    import torchvision.transforms as transforms

    from nltk import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from gensim.models.keyedvectors import KeyedVectors
    from wikipedia2vec import Wikipedia2Vec

    from util.datahelper import get_data, get_entity_claim

    from tqdm import tqdm

    class Preprocess():
        '''
            This class is for FakeNewsNet data
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
        '''
        def __init__(self, contents, comments, entities, images, labels, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec, sb_type, img_embed_params, clip_embed_params, kahan, exclude_with_no_images, use_clip, max_clip_ent=10, max_clip_clms=25,  max_len=60, max_sent=30, max_ent=100, M=5, max_cmt=50, intervals=100):
            self.contents = contents
            self.comments = comments
            self.entities = entities
            self.images = images
            self.labels = labels

            self.indices = np.arange(len(self.labels))

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

            self.img_embed_params = img_embed_params
            self.kahan = kahan
            self.exclude_with_no_images = exclude_with_no_images

            self.use_clip = use_clip
            self.clip_embed_params = clip_embed_params

            self.max_clip_ent = max_clip_ent
            self.max_clip_clms = max_clip_clms
            

        def __len__(self):
            return len(self.labels)
            
        def _img_att_preprocess(self, ents, tokenizer, model, pool, embedding_size=512):
            clip_ent_vec = []
            clip_clm_vec = []

            ent_clms = []

            num_ent_found = 0
            for entity in ents:
                if entity not in claim_dict:
                    continue
                num_ent_found += 1
                
                clms = claim_dict[entity]
                for i, clm in enumerate(clms):
                    clm = clm.split(':')
                    clm = clm[1] if len(clm)>1 else clm[0]
                    clms[i] = clm
                
                clms = clms[:self.max_clip_clms]
                ent_clms.append([entity.replace('_', ' ')] + clms)

            ent_clms = ent_clms[:self.max_clip_ent]

            for (ent_embed, clm_embed) in pool.starmap(clip_ent_claims_encoder, [(ent_clm, tokenizer, model) for ent_clm in ent_clms]):
                clip_ent_vec.append(np.array(ent_embed).flatten())
                clip_clm_vec.append(np.array(clm_embed).flatten())

            clip_lk = len(clip_ent_vec) if len(clip_ent_vec)<self.max_clip_ent else self.max_clip_ent
            clip_lk = clip_lk if clip_lk>0 else 1
            #print('clip_lk: {}'.format(clip_lk))
            
            clip_ent_vec = clip_ent_vec[:self.max_clip_ent]
            clip_clm_vec = clip_clm_vec[:self.max_clip_ent]

            if clip_ent_vec and clip_clm_vec:
                clip_ent_vec = np.pad(clip_ent_vec, ((0, self.max_clip_ent-len(clip_ent_vec)),(0, 0)))
                clip_clm_vec = np.pad(clip_clm_vec, ((0, self.max_clip_ent-len(clip_clm_vec)),(0, 0)))
                #print('clip_ent_vec: {}'.format(clip_ent_vec.shape))
            else:
                clip_ent_vec = np.full((self.max_clip_ent, embedding_size), 0.0)
                clip_clm_vec = np.full((self.max_clip_ent, embedding_size), 0.0)
                #clip_ent_vec = np.concatenate((clip_ent_vec, clip_ent_vec), axis=1)

            #print('clip_ent_vec: {}, clip_lk: {}'.format(clip_ent_vec.shape, clip_lk))
            # print data types

            return clip_ent_vec, clip_clm_vec, clip_lk

        # convert entity into embed
        def _get_entity_embed(self, ent):
            if self.wiki2vec.get_entity(ent.replace('_', ' ')):
                return self.wiki2vec.get_entity_vector(ent.replace('_', ' '))
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
                clm = clm.split(':')
                clm = clm[1] if len(clm)>1 else clm[0]
                
                if self.wiki2vec.get_entity(clm):
                    clm_embed.append(self.wiki2vec.get_entity_vector(clm))
                else:
                    # get word2vec
                    token = clm.split(' ')
                    token = [tk.lower() for tk in token if self.wiki2vec.get_word(tk.lower())]
                    if token:
                        clm_embed.extend([self.wiki2vec.get_word_vector(tk) for tk in token])
            
            # average claim vector, pad if no word2vec or wiki2vec found 
            if clm_embed:
                return np.average(clm_embed, axis=0)
            else:
                return None

        # given entities in a sentence, return MAXPOOL entity embed and claim embed
        def _knowledge_preprocess(self, ents):
            ent_vec = []
            for ent in ents:
                #print('ent: {}'.format(ent))
                ent_embed = self._get_entity_embed(ent)
                #print('ent_embed: {}'.format(ent_embed.shape)) if ent_embed is not None else #print('ent_embed: None')
                clm_embed = self._get_claim_embed(ent)
                #print('clm_embed: {}'.format(clm_embed.shape)) if clm_embed is not None else #print('clm_embed: None')

                if isinstance(ent_embed, np.ndarray) and isinstance(clm_embed, np.ndarray):
                    ent_vec.append(np.concatenate((ent_embed, clm_embed), axis=None))

            lk = len(ent_vec) if len(ent_vec)<self.max_ent else self.max_ent
            lk = lk if lk>0 else 1
            #print('lk: {}'.format(lk))
            
            ent_vec = ent_vec[:self.max_ent]
            #print('ent_vec: {}'.format(len(ent_vec)))
            if ent_vec:
                ent_vec = np.pad(ent_vec, ((0, self.max_ent-len(ent_vec)),(0, 0)))
                #print('ent_vec: {}'.format(ent_vec.shape))
            else:
                ent_vec = np.full((self.max_ent, self.word2vec_cnt.vector_size), self.word2vec_cnt['_pad_'])
                ent_vec = np.concatenate((ent_vec, ent_vec), axis=1)

            #print('ent_vec: {}, lk: {}'.format(ent_vec.shape, lk))

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
            ##print('comments: {}'.format(comments))
            
            # convert word into index of word embedding
            word_vec = [[self.word2vec_cmt.key_to_index[w] if w in self.word2vec_cmt.key_to_index else self.word2vec_cmt.key_to_index['_unk_'] for w in cmt] for cmt in comments if cmt]

            # calculate number of comments and comments lengths
            lsb = len(word_vec) if len(word_vec)<self.max_cmt else self.max_cmt
            lc = [len(cmt) if len(cmt)<self.max_len else self.max_len for cmt in word_vec]
            lc = lc[:self.max_cmt]
            lc = np.pad(lc, (0, self.max_cmt-len(lc))).astype(int)
            ##print('lsb: {} lc: {}'.format(lc, lsb))

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
            ##print('results: {}'.format(results))

            le = np.count_nonzero([sb[1] for sb in results])
            ##print('le: {}'.format(le))
            word_vec, lsb, lc = self._comment_reorder(results)
            ##print('lsb: {} lc: {}'.format(lsb, lc))

            return word_vec, le, lsb, lc
        
        # create a vector space representation of the image using clip
        def _get_clip_img_embed(self, image):
            image_features = None
            if image is None or self.kahan:
                image_features = torch.zeros(self.clip_embed_params['embedding_size'])
            else:
                img_as_pil = self.clip_embed_params['transform'](image)
                image = self.clip_embed_params['preprocess'](img_as_pil).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_embed_params['model'].encode_image(image).squeeze(0)

            return image_features

        # create embedding of the PIL image object and return
        def _get_img_embed(self, image):
            embedding = None
            if image is None or self.kahan:
                embedding = torch.zeros(self.img_embed_params['embedding_size'])
            else:
                with torch.no_grad():
                    # apply the transform to the image
                    image_transformed = self.img_embed_params['preprocess'](image)
                    # extract the image embedding
                    extracted_embedding = self.img_embed_params['model'](image_transformed.unsqueeze(0))
                    embedding = extracted_embedding.squeeze(0)
            # return the feature vector
            return embedding

        # return data
        def preprocess(self):
            contents = []
            comments = []
            entities = []
            clip_entities = []
            images = []
            labels = []

            for index in tqdm(self.indices, desc='Preprocessing'):
                content = self.contents[index]
                comment = self.comments[index]
                entity = self.entities[index]
                image = self.images[index]
                label = self.labels[index]

                content_vec, ln, ls = self._news_content_preprocess(content)
                comment_vec, le, lsb, lc = self._build_subevents(comment)
                ent_vec, lk = self._knowledge_preprocess(entity)

                clip_ent_vec, clip_clm_vec, clip_lk = None, None, None
                if self.use_clip:
                    clip_ent_vec, clip_clm_vec, clip_lk = self._img_att_preprocess(entity, self.clip_embed_params['tokenizer'], self.clip_embed_params['model'], self.clip_embed_params['pool'], self.clip_embed_params['embedding_size']) 
                else:
                    # fill with zeros if not using clip to maintain consistency
                    clip_ent_vec = np.full((self.max_clip_ent, self.clip_embed_params['embedding_size']), 0.0)
                    #clip_ent_vec = np.concatenate((clip_ent_vec, clip_ent_vec), axis=1)
                    clip_lk = self.max_clip_ent

                img_vec = self._get_clip_img_embed(image) if self.use_clip else self._get_img_embed(image)
                
                if self.exclude_with_no_images:
                    if image is not None:
                        contents.append((torch.tensor(content_vec), torch.tensor(ln), torch.tensor(ls)))
                        comments.append((torch.tensor(comment_vec), torch.tensor(le), torch.tensor(lsb), torch.tensor(lc)))
                        entities.append((torch.tensor(ent_vec), torch.tensor(lk)))
                        clip_entities.append((torch.tensor(clip_ent_vec), torch.tensor(clip_clm_vec), torch.tensor(clip_lk)))
                        images.append(img_vec)
                        labels.append(torch.tensor(label))       
                else:
                    contents.append((torch.tensor(content_vec), torch.tensor(ln), torch.tensor(ls)))
                    comments.append((torch.tensor(comment_vec), torch.tensor(le), torch.tensor(lsb), torch.tensor(lc)))
                    entities.append((torch.tensor(ent_vec), torch.tensor(lk)))
                    clip_entities.append((torch.tensor(clip_ent_vec), torch.tensor(clip_clm_vec), torch.tensor(clip_lk)))
                    images.append(img_vec)
                    labels.append(torch.tensor(label))     
            
            return contents, comments, entities, clip_entities, images, labels
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=str, default='politifact')
    parser.add_argument('--cnn', type=str, default='vgg19')
    parser.add_argument('--kahan', action='store_true')
    parser.add_argument('--exclude_with_no_images', action='store_true')
    parser.add_argument('--use_clip', action='store_true')
    args = parser.parse_args()

    # load config
    config = None
    if args.platform.startswith('politifact'):
        config = json.load(open('./config_p.json'))
    elif args.platform == 'gossipcop':
        config = json.load(open('./config_g.json'))
    else:
        raise ValueError('Invalid platform argument')

    # load data
    contents, comments, entities, images, labels = get_data(config['data_dir'], config['data_source'])
    claim_dict = get_entity_claim(config['data_dir'], config['data_source'])

    # load word2vec, wiki2vec model and add unk vector
    word2vec_cnt = KeyedVectors.load_word2vec_format(config['word2vec_cnt'])
    word2vec_cnt.add_vector('_unk_', np.average(word2vec_cnt.vectors, axis=0))
    word2vec_cnt.add_vector('_pad_', np.zeros(config['word2vec_dim']))
    word2vec_cmt = KeyedVectors.load_word2vec_format(config['word2vec_cmt'])
    word2vec_cmt.add_vector('_unk_', np.average(word2vec_cmt.vectors, axis=0))
    word2vec_cmt.add_vector('_pad_', np.zeros(config['word2vec_dim']))
    wiki2vec = Wikipedia2Vec.load(config['wiki2vec'])

    img_embed_params = {}

    if args.cnn == 'vgg19':
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        # remove the last layer (classifier) so that the output is the embedding
        model = nn.Sequential(*list(model.children())[:-1])
        model.add_module('flatten', nn.Flatten(start_dim=1))
        # freeze the model
        model.eval()
        preprocess = VGG19_Weights.DEFAULT.transforms()
        embedding_size = config['image_preprocessing']['{}_embed_size'.format(args.cnn)]
        # add to config
        img_embed_params.update({'model': model, 'preprocess': preprocess, 'embedding_size': embedding_size})
    elif args.cnn == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # remove the last two layers (classifier) so that the output is the embedding
        model = nn.Sequential(*list(model.children())[:-2])
        model.add_module('flatten', nn.Flatten(start_dim=1))
        # freeze the model
        model.eval()
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        embedding_size = config['image_preprocessing']['{}_embed_size'.format(args.cnn)]
        # add to config
        img_embed_params.update({'model': model, 'preprocess': preprocess, 'embedding_size': embedding_size})
    else:
        raise ValueError('CNN model not supported')
    
    model, _, preprocess = open_clip.create_model_and_transforms(config['image_preprocessing']['clip_model'], pretrained=config['image_preprocessing']['clip_pretrained'])
    tokenizer = open_clip.get_tokenizer(config['image_preprocessing']['clip_tokenizer'])
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    pool = Pool(5)
    # add to config
    clip_embed_params = {'model': model, 'transform': transform, 'preprocess': preprocess, 'tokenizer': tokenizer, 'pool': pool, 'embedding_size': config['image_preprocessing']['clip_embed_size']}

    # preprocess data
    preprocessor = Preprocess(contents, comments, entities, images, labels, claim_dict, word2vec_cnt, word2vec_cmt, wiki2vec,
            sb_type=config['sb_type'], img_embed_params=img_embed_params, clip_embed_params=clip_embed_params, kahan=args.kahan, exclude_with_no_images=args.exclude_with_no_images, use_clip=args.use_clip, max_len=config['max_len'], max_sent=config['max_sent'], max_ent=config['max_ent'], M=config['M'], max_cmt=config['max_cmt'], max_clip_ent=config['max_clip_ent'], max_clip_clms=config['max_clip_clms'])
    
    contents, comments, entities, clip_entities, images, labels = preprocessor.preprocess()

    pool.close()
    pool.join()

    # save data
    save_path = ''

    if args.kahan and args.exclude_with_no_images:
        save_path = '{}/{}/preprocessed_kahan_exclude_with_no_image.pt'.format(config['data_dir'], config['data_source'])
    elif args.kahan:
        save_path = '{}/{}/preprocessed_kahan.pt'.format(config['data_dir'], config['data_source'])
    elif args.use_clip:
        save_path = '{}/{}/preprocessed_clip.pt'.format(config['data_dir'], config['data_source'])
    else:
       save_path = '{}/{}/preprocessed_{}.pt'.format(config['data_dir'], config['data_source'], args.cnn)

    torch.save({
        'contents': contents,
        'comments': comments,
        'entities': entities,
        'clip_entities': clip_entities,
        'images': images,
        'labels': labels,
    }, save_path)

