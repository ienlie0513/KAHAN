import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

# input news senteces, output the dict of em model input {'news_id_<sent_idx>':[[text],[]], 'news_id_<sent_idx>':[[text],[]]}
# example input: {"test_doc1": [text, []], "test_doc2": [text, []]}
def input_processing(contents, ids):
    input_list = []

    for cnts, nid in zip(contents, ids):
        for idx, sent in enumerate(cnts):
            key = nid + '_' + str(idx)
            value = [sent, []]
            input_list.append((key,value))

    print ("Total input sents: ", len(input_list), file=sys.stderr)
    return input_list

# convert input dict into a list of batch, each batch is a dict with batch size sents
def batch_input(input_list, batch_size):
    num_batch = int(len(input_list)/batch_size)

    batch_list = []
    for i in range(num_batch):
        batch = {}
        for k,v in input_list[i*batch_size:i*batch_size+batch_size]:
            batch[k]=v
        batch_list.append(batch)

    batch = {}
    for k,v in input_list[num_batch*batch_size:]:
        batch[k]=v
    batch_list.append(batch)

    print ("Total batch: ", len(batch_list), file=sys.stderr)

    return batch_list

# parse batch results
def result_processing(results):
    entity_dict = {} # {'nid': [(sid, 'ent0 ent1 ent2'), (sid, 'ent0 ent1 ent2')]}
    for batch in results:
        for k, ents in batch:
            nid, sid = k.split('_')

            if nid not in entity_dict.keys():
                entity_dict[nid] = []

            entity_dict[nid].append((sid, ' '.join(ents)))

    entity_list = [] # [{'id': k, 'entities': 'ent0 ent1 ent2||ent0 ent1 ent2'}]
    for k,v in entity_dict.items():
        sorted_ent = sorted(entity_dict[k], key = lambda e: e[0])
        entity_list.append({'id': k, 'entities': '||'.join([ent for idx, ent in sorted_ent])})

    return entity_list

# return a list of entities
def extract_entities(input_text, config, md_model, tagger, ed_model):
    mentions_dataset, n_mentions = md_model.find_mentions(input_text, tagger)
    predictions, timing = ed_model.predict(mentions_dataset)

    output_list = []
    for k, ment in mentions_dataset.items():
        ents = []
        if ment != []:
            for ent in predictions[k]:
                if ent['conf_ed'] > 0.65:
                    ents.append(ent['prediction'])

        output_list.append((k, ents))

    return output_list

# get news content from preprocessed tcv file, return in [[sent0, sent1, ...], [sent0, sent1, ...]]
def get_data(df):
    contents = []

    for idx in range(df.id.shape[0]):
        text = df.text[idx]
        text = text.encode('ascii', 'ignore').decode('utf-8')
        contents.append(sent_tokenize(text))

    contents = np.asarray(contents)
    ids = np.asarray(df.id)

    return contents, ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="politifact")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    
    df = pd.read_csv("./data/{}_no_ignore_s.tsv".format(args.platform), sep='\t')
    df['id'] = df['id'].astype(str)
    df = df.fillna('')
    contents, ids = get_data(df)

    input_list = input_processing(contents, ids)
    batch_list = batch_input(input_list, args.batch)

    
    config = {
        "mode": "eval",
        "model_path": "./REL/ed-wiki-2019/model",  # or alias, see also tutorial 7: custom models
    }

    wiki_version = "wiki_2019"
    base_url = "./REL/"

    md_model = MentionDetection(base_url, wiki_version)
    # tagger = Cmns(base_url, wiki_version, n=5) # Using n-gram
    tagger = load_flair_ner("ner-fast") # Using Flair
    ed_model = EntityDisambiguation(base_url, wiki_version, config)

    entities = [extract_entities(batch, config, md_model, tagger, ed_model) for batch in tqdm(batch_list)]
    ent_df = pd.DataFrame(result_processing(entities))
    
    df = pd.merge(df, ent_df, on='id')

    df.to_csv("./data/{}_no_ignore_en.tsv".format(args.platform), sep = '\t', index=False)