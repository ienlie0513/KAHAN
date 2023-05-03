import pandas as pd
import os

from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests

import argparse 

from ast import literal_eval

import concurrent.futures
from tqdm.contrib.concurrent import thread_map

import urllib.parse
import distance

def most_similar(comparator, candidate_list):
    min_distance = float('inf')
    most_similar = ''

    for string in candidate_list:
        curr_distance = distance.levenshtein(comparator, string)
        if curr_distance < min_distance:
            min_distance = curr_distance
            most_similar = string

    return most_similar

def download_and_save_image(url, idx, save_path):
    print('Downloading for: ' + str(idx))
    encoded_url = urllib.parse.quote(url, safe=':/')
    response = requests.get(encoded_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img.save(save_path)

def load_and_store_images(df, source, directory, max_workers=10):
    found_count = 0
    img_number = 0
    os.makedirs(directory, exist_ok=True)

    def process_row(row):
        nonlocal found_count, img_number
        save_path = directory + '/' + source + '_' + str(row[1]['id']) + '.jpg'
        try:
            if os.path.exists(save_path):
                return
            if isinstance(row[1]['image'], str) and row[1]['image'] != '':
                if df.image[row[1].name].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    img_number += 1
                    download_and_save_image(row[1]['image'], row[1]['id'], save_path)
                    found_count += 1
                    print('Image found: {}/{}'.format(found_count, img_number))
        except UnidentifiedImageError:
            try:
                # Try again with the most similar image url
                all_images = literal_eval(row[1]['all_images'])
                # Exclude the image url that was already tried
                all_images.remove(row[1]['image'])
                # Find the most similar image url
                most_similar_image_url = most_similar(row[1]['image'], all_images)
                img_number += 1
                download_and_save_image(most_similar_image_url, row[1]['id'], save_path)
                found_count += 1
                print('Image found: {}/{}'.format(found_count, img_number))
            except Exception as e:
                print('Image not loaded: ', e)
        except Exception as e:
            print('Image not loaded: ', e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        thread_map(process_row, df.itertuples(), max_workers=max_workers, desc="Downloading images")


# def load_and_store_images(df, source, directory):
#     found_count = 0
#     img_number = 0
#     os.makedirs(directory, exist_ok=True)
#     for idx, row in df.iterrows():
#         save_path = directory + '/' + source + '_' + str(row['id']) + '.jpg'
#         try:
#             if os.path.exists(save_path):
#                 continue
#             if isinstance(row['image'], str) and row['image'] != '':
#                 if df.image[idx].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
#                     img_number += 1
#                     download_and_save_image(row['image'], row['id'], save_path)
#                     found_count += 1
#                     print('Image found: {}/{}'.format(found_count, img_number))
#         except UnidentifiedImageError:
#             try:
#                 # Try again with the most similar image url
#                 all_images = literal_eval(row['all_images'])
#                 # Exclude the image url that was already tried
#                 all_images.remove(row['image'])
#                 # Find the most similar image url
#                 most_similar_image_url = most_similar(row['image'], all_images)
#                 img_number += 1
#                 download_and_save_image(most_similar_image_url, row['id'], save_path)
#                 found_count += 1
#                 print('Image found: {}/{}'.format(found_count, img_number))
#             except Exception as e:
#                 print('Image not loaded: ', e)
#         except Exception as e:
#             print('Image not loaded: ', e)

if __name__ == '__main__':

    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('--platform', type=str, default='politifact_v4')
    argsparser.add_argument('--max_workers', type=int, default=10)
    args = argsparser.parse_args()

    data_dir = './data'
    img_dir = 'news_images'

    df_p = pd.read_csv(data_dir + '/' + args.platform + '_no_ignore_s.tsv', sep='\t')
    # df_g = pd.read_csv('./data/gossipcop_data.tsv', sep='\t')

    load_and_store_images(df_p[df_p['label'] == 1], args.platform.split('_')[0], '{}/{}/{}/{}'.format(data_dir, args.platform, img_dir, 'real'), args.max_workers)
    load_and_store_images(df_p[df_p['label'] == 0], args.platform.split('_')[0], '{}/{}/{}/{}'.format(data_dir, args.platform, img_dir, 'fake'), args.max_workers)

    # df_wa = pd.read_csv(data_dir + '/cases_without_web_archive_url.csv', sep=',')
    # df_wa = df_wa[df_wa['img_url'].notna()]
    # df_wa.rename(columns={'img_url': 'image'}, inplace=True)
    # df_p = df_p[df_p['id'].isin(df_wa['id'])]

    # load_and_store_images(df_wa, 'politifact', '{}/{}'.format(data_dir, 'images_with_web_archive_url'))
    # load_and_store_images(df_p, 'politifact', '{}/{}'.format(data_dir, 'images_without_web_archive_url'))

