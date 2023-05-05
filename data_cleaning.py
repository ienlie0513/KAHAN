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
    argsparser.add_argument('--remove_images', action='store_true')
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
        if 'politifact' in args.platform:
            p_f_images_to_remove = [13617, 13936, 14265, 15246]
            p_r_images_to_remove = [118, 368, 596, 937, 1375, 3192, 4555, 4787, 6907, 7714, 8005, 8045, 8557, 8769, 10276, 11069, 12486, 12587, 12944, 13058, 14984, 15133]
            clean_news_images(p_f_images_to_remove + p_r_images_to_remove, './data/{}/news_images'.format(args.platform), args.platform)

        if 'gossipcop' in args.platform:
            g_f_images_to_remove = [20216430, 465550573, 606633573, 756254851, 1435049921, 1885887731, 2365780434, 2738137690, 3282186402, 3325039430, 3358992856, 3571366577, 3726214854, 3738976126, 4241531514, 4298949105, 4341857401, 5049497697, 5480976970, 6344960812, 6489461221, 6798466350, 726695676, 7319293593, 7321484176, 7467879756, 7853811420, 7917898417, 8066502884, 8165063873, 8688894238, 8852162770, 8904022923, 9016605294, 9180203998, 9897229740]
            g_r_images_to_remove = [842508, 843384, 846495, 846867, 847323, 848307, 854510, 860111, 864060, 864371, 864751, 866513, 869516, 870257, 871577, 873934, 877762,  878884, 884397, 886293, 889175, 894734, 895797, 905961, 907328, 908912, 914412, 915457, 917435, 930270, 931604, 947524, 955137]
            clean_news_images(g_f_images_to_remove + g_r_images_to_remove, './data/{}/news_images'.format(args.platform), args.platform)

    #check if nan exists
    print(df['comments'].isnull().values.any())

    df.to_csv('./data/{}_no_ignore_s.tsv'.format(args.platform), sep='\t', index=False)
    