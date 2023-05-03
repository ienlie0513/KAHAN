import os
import pandas as pd
import argparse

from langdetect import detect
import re

def clean_news_content(platform_df):
    # no content
    platform_df = platform_df.dropna(subset=['text'])
    # no duplicate content
    platform_df = platform_df.drop_duplicates(subset=['text'])
    # no empty content
    platform_df = platform_df[platform_df['text'].str.strip().astype(bool)]
    # less than 50 characters
    platform_df = platform_df[platform_df['text'].str.len() > 50]
    # non english content
    platform_df = platform_df[platform_df['text'].apply(lambda x: detect(x) == 'en')]

    return platform_df

def clean_news_comments(platform_df):
    # no content
    platform_df = platform_df.dropna(subset=['comments'])

    return platform_df

def clean_news_images(remove_lst, img_dir, platform):
    files_to_remove = ['{}_{}.jpg'.format(platform.split('_')[0], i) for i in remove_lst]
    for subdir in ['fake', 'real']:
        for img_file in os.listdir(img_dir + '/' + subdir):
            if img_file in files_to_remove:
                print('Removing {}'.format(img_file))
                os.remove(img_dir + '/' + subdir + '/' + img_file)

if __name__ == '__main__':

    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('--platform', type=str, default='politifact_v4')
    argsparser.add_argument('--remove_images', type=bool, default=False)
    args = argsparser.parse_args()

    df = pd.read_csv('./data/{}_no_ignore_s.tsv'.format(args.platform), sep='\t')

    df_text_cleaned = clean_news_content(df)
    print('Text Content \n {} Real News Removed - {} Fake News Removed'.format(len(df[df['label'] == 1]) - len(df_text_cleaned[df_text_cleaned['label'] == 1]), len(df[df['label'] == 0]) - len(df_text_cleaned[df_text_cleaned['label'] == 0])))
    df = df_text_cleaned

    # df_comments_cleaned = clean_news_comments(df)
    # print('Comments \n {} Real News Removed - {} Fake News Removed'.format(len(df[df['label'] == 1]) - len(df_comments_cleaned[df_comments_cleaned['label'] == 1]), len(df[df['label'] == 0]) - len(df_comments_cleaned[df_comments_cleaned['label'] == 0])))
    # df = df_comments_cleaned

    print('New Size: {}'.format(len(df)))

    if args.remove_images:
        p_f_images_to_remove = [13617, 13936, 14265, 15246]
        p_r_images_to_remove = [118, 368, 596, 937, 1375, 3192, 4555, 4787, 6907, 7714, 8005, 8045, 8557, 8769, 10276, 11069, 12486, 12587, 12944, 13058, 14984, 15133]
        clean_news_images(p_f_images_to_remove + p_r_images_to_remove, './data/{}/news_images'.format(args.platform), args.platform)

    #check if nan exists
    print(df['comments'].isnull().values.any())

    df.to_csv('./data/{}_no_ignore_s.tsv'.format(args.platform), sep='\t', index=False)
    