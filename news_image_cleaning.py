import os
import json
import re

import pandas as pd
import argparse

from waybackpy import WaybackMachineCDXServerAPI
from newspaper import Article

df_e = pd.read_csv("./data/politifact_v2_no_ignore_en.tsv", sep="\t")
news_ids = df_e['id'].tolist()

news_with_factcheck_image = [787, 13052, 12748, 14064, 6932, 780, 4275, 5237, 13283, 4588, 11552, 6603, 11761, 11314, 6646, 13305, 12801, 6939]

fake_news_with_invalid_image = [15371, 15386, 14984, 14265, 13617, 12944]
real_news_with_invalid_image = [14984, 13058, 11855, 11699, 10276, 8118, 8045, 8119, 7714, 3192, 937, 679, 596, 368, 118, 516]

def return_matching_urls(dir_path, source, folders=['real', 'fake'], statement='politifact', negated=False):
    urls = []
    for folder in folders:
        for subfolder in os.listdir(dir_path + '/' + source + '/' + folder):
            try:
                with open(dir_path + '/' + source + '/' + folder + '/' + subfolder + '/news content.json') as f:
                    file_content = json.load(f)
                    news_id = int(re.findall(r'\d+', subfolder)[0])
                    url = file_content['url']
                    if negated:
                        if statement not in url and news_id in news_ids: # and news_id in news_with_factcheck_image:
                            urls.append({'id': news_id, 'url': url})
                    else:
                        if statement in url and news_id in news_ids: # and news_id in news_with_factcheck_image:
                            urls.append({'id': news_id, 'url': url})
            except Exception as e:
                print(e)
    return urls


def update_dataset(dataset, source, df, folders=['real', 'fake']):
    for folder in folders:
        for subfolder in os.listdir(dataset + '/' + source + '/' + folder):
            for _, row in df.iterrows():
                if row['id'] == int(re.findall(r'\d+', subfolder)[0]):
                    try:
                        with open(dataset + '/' + source + '/' + folder + '/' + subfolder + '/news content.json') as f:
                            file_content = json.load(f)
                            file_content['top_img'] = row['img_url']
                            with open(dataset + '/' + source + '/' + folder + '/' + subfolder + '/news content.json', 'w') as f:
                                json.dump(file_content, f)
                    except Exception as e:
                        print(e)

def update_dataframe(df, new_img_df):
    for _, row in new_img_df.iterrows():
        for idx, df_row in df.iterrows():
            if row['id'] == df_row['id']:
                print('Before: {} \r'.format(df.image[idx]))
                df.image[idx] = row['img_url']
                print('After: {} \r'.format(df.image[idx]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='')
    args = parser.parse_args()

    directory = './data'
    platform = 'politifact_v2'
    dataset = 'fakenewsnet_dataset_v2'

    if args.action == 'perform update':
        df_pu = pd.read_csv(directory + '/politifact_urls.csv')
        df_wa = pd.read_csv(directory + '/cases_without_web_archive_url.csv')
        df_ir = pd.read_csv(directory + '/img_replacement.csv')
        df_en = pd.read_csv(directory + '/' + platform + '_no_ignore_en.tsv', sep='\t')
        df_s = pd.read_csv(directory + '/' + platform + '_no_ignore_s.tsv', sep='\t')

        print(df_ir.columns)

        new_img_df = df_pu[df_pu['img_url'].notna()]
        wa_img_df = df_wa[df_wa['img_url'].notna()]

        update_dataset(dataset, 'politifact', df=new_img_df)

        update_dataframe(df_en, new_img_df)
        update_dataframe(df_en, wa_img_df)
        update_dataframe(df_en, df_ir)
        # remove images for news with invalid images
        df_en.loc[df_en['id'].isin(fake_news_with_invalid_image + real_news_with_invalid_image), 'image'] = ''
        df_en.to_csv(directory + '/' + platform + '_no_ignore_en.tsv', sep='\t', index=False)

        update_dataframe(df_s, new_img_df)
        update_dataframe(df_s, wa_img_df)
        update_dataframe(df_s, df_ir)
        # remove images for news with invalid images
        df_s.loc[df_s['id'].isin(fake_news_with_invalid_image + real_news_with_invalid_image), 'image'] = ''
        df_s.to_csv(directory + '/' + platform + '_no_ignore_s.tsv', sep='\t', index=False)

    elif args.action == 'retrieve politifact urls':
        politifact_urls = return_matching_urls(dataset, 'politifact', statement='politifact.com')
        df = pd.DataFrame(politifact_urls)
        df.to_csv(directory + '/politifact_urls.csv', index=False)

    elif args.action == 'retrieve non web archive urls':
        non_web_archive_urls = return_matching_urls(dataset, 'politifact', statement='web.archive.org', negated=True)
        df = pd.DataFrame(non_web_archive_urls)
        df.to_csv(directory + '/cases_without_web_archive_url.csv', index=False)

    elif args.action == 'find web archive images':
        df = pd.read_csv(directory + '/cases_without_web_archive_url.csv')
        for _, row in df.iterrows():
            try:
                cdx_api = WaybackMachineCDXServerAPI(row['url'])
                oldest = cdx_api.oldest()
                archive_url = oldest.archive_url
                article = Article(archive_url)
                article.download()
                article.parse()
                print(str(row['id']) + ': ' + article.top_image + '\r')
                df.loc[df['id'] == row['id'], 'img_url'] = article.top_image
            except Exception as e:
                print(e)
        df.to_csv(directory + '/cases_without_web_archive_url.csv', index=False)

    elif args.action == 'clean gossipcop':
        id_to_remove = [7259603960, 843758, 911401, 923847, 948472, 875314, 8081058062, 3663225246, 3066088888, 4495586215, 2778253719, 2507720803, 4024992311, 6489461221, 1615677123, 5111151830, 5328116357, 9328842988, 4888675652, 6899501174, 8172018375, 3172946389, 5034259222, 843265, 2008641429, 4466125915, 5480976970, 5408886782, 4360448788, 1457904080, 2254004589, 1628848955, 92814312, 4802653039, 267792336, 3893905315, 8410631215, 5241899341, 948328, 887769, 888688, 860652,  919774, 951548, 947692, 948524, 942374]
        df = pd.read_csv(directory + '/gossipcop_no_ignore_en.tsv', sep='\t')
        df = df[~df['id'].isin(id_to_remove)]

        # remove rows with text length less than 1000
        df = df[df['text'].str.len() > 1000]

        df.to_csv(directory + '/gossipcop_no_ignore_en.tsv', sep='\t', index=False)

    elif args.action == 'clean politifact':
        id_to_remove = [14920, 15108, 14498, 15129, 13711, 13576, 7540, 279, 65, 13900, 200, 10332, 320, 7506, 772, 12120, 2624, 773, 11960, 14225, 6267, 954, 7923, 681, 211, 620, 401, 370, 13058, 11399, 8119, 13058, 11399, 8119, 620, 8470, 14498, 15129, 13576, 7540, 279, 65, 13900, 200, 7506, 772, 12120, 773, 2624, 11960, 14225, 6267, 954, 7923, 1213, 681, 211, 8470, 620, 401]
        df = pd.read_csv(directory + '/politifact_v2_no_ignore_en.tsv', sep='\t')
        df = df[~df['id'].isin(id_to_remove)]
        df.to_csv(directory + '/politifact_v2_no_ignore_en.tsv', sep='\t', index=False)

    else:
        print('No action specified')



