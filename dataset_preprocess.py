# TODO: generate two preprocessed TSV-files from the FakeNewsNet dataset in the following format:
#       [id, text, comments, label]
#       where id is the id of the news article, content is the content of the news article, comments is the Twitter comments of the news article, and label is the label of the news article (0 or 1).
#       The TSV-files should be saved in the same directory as this script.
#       The TSV-files should be named "politifact_no_ignore_s.tsv" and "gossipcop_no_ignore_s.tsv".

import os
import shutil
import re
import json
import pandas as pd

# make a new directory for the preprocessed data
dir_path = os.getcwd() + '/data'
os.makedirs(dir_path, exist_ok=True)

# create a log file for datanalysis
log_file = open(dir_path + '/preprocessing_log.txt', 'w')

def get_comments(file_path):
    comments = ''
    try:
        with open(file_path) as f:
            file_content = json.load(f)
            for comment in file_content.values():
                for c in comment:
                    text = re.sub(r'<>+', '', c['text'])
                    text = re.sub(r'::+', '', text)
                    comments += text + '<>' + c['created_at'] + '::'
    except Exception as e:
        print('Error: ', e)
    finally:
        # remove mentions and urls
        comments = re.sub(r'@\w+', '', comments)
        comments = re.sub(r'http\S+', '', comments)
        # remove trailing '::'
        comments = comments[:-2]
        return comments

def get_news_content(file_path):
    text = ''
    image = ''
    all_images = ''
    file_content = {}
    try:
        with open(file_path) as f:
            file_content = json.load(f)
            text = file_content['text']
            image = file_content['top_img']
            all_images = file_content['images']
    except Exception as e:
        print('Error: ', e)
        print(file_path)
    finally:
        return text, image, all_images
    

def preprocess_news_data(_dir, sub_dir):
    contents = []
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            n_id = re.findall(r'\d+', folder)[0]
            text, image, all_images = get_news_content(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news_article.json')
            comments = get_comments(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/replies.json')
            log_file.write('{}: {} \r'.format(n_id, text[:100]))
            contents.append({
                'id': n_id,
                'text': text,
                'comments': comments,
                'image': image,
                'all_images': all_images,
                'label': 1 if sub_sub_dir == 'real' else 0
            })
    return contents

def get_news_article(f_file_path, t_file_folder):
    file_content = {}
    try:
        with open(f_file_path) as f:
            file_content = json.load(f)
    except Exception as e:
        print('Error: ', e)
        if 'No such file or directory' in str(e):
            # delete folder that failed to be imputed due to missing impute data
            print('Deleting: ', t_file_folder)
            shutil.rmtree(t_file_folder)

    return file_content


def dataset_imputer(impt_from, impt_to, dataset):
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(impt_to + '/' + dataset + '/' + sub_sub_dir):
            file_path = impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/news_article.json'
            try:
                with open(file_path) as f:
                    file_content = json.load(f)
                    if not file_content:
                        print('Imputing: ', file_path)
                        file_content = get_news_article(impt_from + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/news content.json', impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder)
                        # write to file
                        with open(file_path, 'w') as f:
                            json.dump(file_content, f)
            except Exception as e:
                print('Error: ', e)


# def preprocess_news_data(_dir, sub_dir):
#     contents = []
#     for sub_sub_dir in ['fake', 'real']:
#         for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
#             news_content = {}
#             news_content['id'] = re.findall(r'\d+', folder)[0]
#             try:
#                 with open(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json') as f:
#                     file_content = json.load(f)
#                     # log_file.write('{} \r {} \r {}...\r'.format(news_content['id'], file_content['url'],  file_content['text'][:100]))
#                     log_file.write('{}: {} \r'.format(news_content['id'], file_content['text'][:100]))
#                     news_content['text'] = file_content['text']
#                     news_content['image'] = file_content['top_img']
#                     news_content['all_images'] = file_content['images']
#                     news_content['label'] = 1 if sub_sub_dir == 'real' else 0
#                     contents.append(news_content)
#             except Exception as e:
#                 print(e)
#                 news_content['text'] = ''
#                 news_content['image'] = ''
#                 news_content['all_images'] = ''
#                 news_content['label'] = 1 if sub_sub_dir == 'real' else 0
#                 news_content['comments'] = ''
#                 news_contents.append(news_content)

#     return news_contents

#dataset_imputer('fakenewsnet_dataset_v2', 'fakenewsnet_dataset_v3', 'gossipcop')

gossipcop_news_contents = preprocess_news_data('fakenewsnet_dataset_v3', 'gossipcop')
gossipcop_news_contents_df = pd.DataFrame(gossipcop_news_contents)
gossipcop_news_contents_df.to_csv('data/gossipcop_no_ignore_s.tsv', sep='\t', index=False)

politifact_news_contents = preprocess_news_data('fakenewsnet_dataset_v3', 'politifact')
politifact_news_contents_df = pd.DataFrame(politifact_news_contents)
politifact_news_contents_df.to_csv('data/politifact_v3_no_ignore_s.tsv', sep='\t', index=False)

