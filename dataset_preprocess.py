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

import argparse

# make a new directory for the preprocessed data
dir_path = os.getcwd() + '/data'
os.makedirs(dir_path, exist_ok=True)

# create a log file for datanalysis
log_file = open(dir_path + '/preprocessing_log.txt', 'w')

def get_time_index(file_content, intervals):
    utc_value = {}
    for comment in file_content.values():
        for c in comment:
            if c['text']:
                if c['text'].isascii() and re.match('.*[a-zA-Z0-9].*', c['text']):
                    utc_value[c['created_at']] = None
    # sort utc_value dict by key (timestamp)
    utc_value = dict(sorted(utc_value.items(), key=lambda item: item[0]))
    # get time index
    indexes = [[] for _ in range(intervals)]
    for i in range(len(utc_value.keys())):
        indexes[i % intervals].append(i % intervals)
    time_index = [i for lst in indexes for i in lst]
    # Assign value of each key (timestamp) to the corresponding time index
    for i, key in enumerate(utc_value.keys()):
        utc_value[key] = time_index[i]
    # return utc_value dict
    return utc_value

def get_comments(file_path, intervals):
    comments = ''
    try:
        with open(file_path) as f:
            file_content = json.load(f)
            time_index = get_time_index(file_content, intervals)
            for comment in file_content.values():
                for c in comment:
                    if c['text']:
                        if c['text'].isascii() and re.match('.*[a-zA-Z0-9].*', c['text']):
                            # remove <> and :: from text
                            text = re.sub(r'<>+', '', c['text']) 
                            text = re.sub(r'::+', '', text)
                            # Add comment texted followed by its timestamp
                            comments += text + '<>' + str(time_index[c['created_at']]) + '::'
    except Exception as e:
        print('Error: ', e)
    finally:
        # remove urls
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
    

def preprocess_news_data(_dir, sub_dir, intervals):
    contents = []
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            if folder == '.DS_Store':
                continue
            n_id = re.findall(r'\d+', folder)[0]
            text, image, all_images = get_news_content(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json')
            comments = get_comments(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/replies.json', intervals)
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

def get_replies(file_path):
    with open(file_path) as f:
        file_content = json.load(f)
        return file_content


def dataset_imputer(impt_from, impt_to, dataset):
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(impt_to + '/' + dataset + '/' + sub_sub_dir):
            if os.path.isdir(impt_from + '/' + dataset + '/' + sub_sub_dir + '/' + folder):
                file_path = impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/news content.json'
                # check if file exists
                if not os.path.isfile(file_path):
                    print('Imputing: ', file_path)
                    file_content = get_news_article(impt_from + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/news_article.json', impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder)
                    # write to file
                    with open(file_path, 'w') as f:
                        json.dump(file_content, f)
                # check if file is empty
                elif os.stat(file_path).st_size == 0:
                    print('Imputing: ', file_path)
                    file_content = get_news_article(impt_from + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/news_article.json', impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder)
                    # write to file
                    with open(file_path, 'w') as f:
                        json.dump(file_content, f)
                # check if text attribute is empty
                elif len(get_news_content(file_path)[0]) == 0:
                    print('Imputing: ', file_path)
                    file_content = get_news_article(impt_from + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/news_article.json', impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder)
                    # write to file
                    with open(file_path, 'w') as f:
                        json.dump(file_content, f)
                # get comments
                comments = get_replies(impt_from + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/replies.json')
                # write to file
                with open(impt_to + '/' + dataset + '/' + sub_sub_dir + '/' + folder + '/replies.json', 'w') as f:
                    json.dump(comments, f)
                

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--platform', type=str, default='politifact', help='politifact or gossipcop')
    argparse.add_argument('--dataset', type=str, default='fakenewsnet_dataset_v4')
    argparse.add_argument('--savename', type=str, default='politifact_v4_no_ignore_s')
    argparse.add_argument('--impute', action='store_true', default=False)
    args = argparse.parse_args()

    if args.platform == 'politifact':
        config = json.load(open('./config_p.json'))
    elif args.platform == 'gossipcop':
        config = json.load(open('./config_g.json'))

    if args.impute:
        dataset_imputer('fakenewsnet_dataset_v3', args.dataset, args.platform)

    news_contents = preprocess_news_data(args.dataset, args.platform, config['intervals'])
    news_contents_df = pd.DataFrame(news_contents)
    news_contents_df.to_csv('data/{}.tsv'.format(args.savename), sep='\t', index=False)



