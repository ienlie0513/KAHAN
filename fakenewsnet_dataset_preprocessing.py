# TODO: generate two preprocessed TSV-files from the FakeNewsNet dataset in the following format:
#       [id, text, comments, label]
#       where id is the id of the news article, content is the content of the news article, comments is the Twitter comments of the news article, and label is the label of the news article (0 or 1).
#       The TSV-files should be saved in the same directory as this script.
#       The TSV-files should be named "politifact_no_ignore_s.tsv" and "gossipcop_no_ignore_s.tsv".

import os
import re
import json
import pandas as pd

# make a new directory for the preprocessed data
dir_path = os.getcwd() + '/data'
os.makedirs(dir_path, exist_ok=True)

def preprocess_news_data(_dir, sub_dir):
    news_contents = []
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            news_content = {}
            news_content['id'] = re.findall(r'\d+', folder)[0]
            print(folder)
            try:
                with open(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json') as f:
                    file_content = json.load(f)
                    news_content['text'] = file_content['text']
                    news_content['image'] = file_content['top_img']
                    news_content['label'] = 1 if sub_sub_dir == 'real' else 0
                    # TODO: include comments as well
                    news_content['comments'] = ''
                    news_contents.append(news_content)
            except:
                news_content['text'] = ''
                news_content['image'] = ''
                news_content['label'] = 1 if sub_sub_dir == 'real' else 0
                news_content['comments'] = ''
                news_contents.append(news_content)
    return news_contents

gossipcop_news_contents = preprocess_news_data('fakenewsnet_dataset', 'gossipcop')
gossipcop_news_contents_df = pd.DataFrame(gossipcop_news_contents)
gossipcop_news_contents_df.to_csv('data/gossipcop_no_ignore_s.tsv', sep='\t', index=False)

politifact_news_contents = preprocess_news_data('fakenewsnet_dataset', 'politifact')
politifact_news_contents_df = pd.DataFrame(politifact_news_contents)
politifact_news_contents_df.to_csv('data/politifact_no_ignore_s.tsv', sep='\t', index=False)


