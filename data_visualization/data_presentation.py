import os
import pandas as pd

import argparse

def compute_stats(df, img_dir):
    true_news_count = len(df[df['label'] == 1])
    fake_news_count = len(df[df['label'] == 0])
    total_news_count = len(df)

    # Convert id to string
    df['id'] = df['id'].astype(str)

    # number of images
    true_news_images = len(df[df['id'].isin([f.split('_')[1].split('.')[0] for f in os.listdir(img_dir + '/real')])])
    fake_news_images = len(df[df['id'].isin([f.split('_')[1].split('.')[0] for f in os.listdir(img_dir + '/fake')])])
    total_news_images = true_news_images + fake_news_images
    
    avg_comments_per_news = df['comments'].apply(lambda x: len(x.split('::'))).mean()
    # only compute for non empty entities
    avg_entities_per_news = df[df['entities'].notna()]['entities'].apply(lambda x: len(x.split('||'))).mean()
    
    return true_news_count, fake_news_count, total_news_count, true_news_images, fake_news_images, total_news_images, avg_comments_per_news, avg_entities_per_news

def compute_entity_claims_stats(df):
    avg_entity_claims_per_news = df['claims'].apply(lambda x: len(x.split('||'))).mean()
    return avg_entity_claims_per_news

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Data presentation')
    parser.add_argument('--version', type=str, default='_v4', help='Dataset version to use')
    parser.add_argument('--reduced', action='store_true', help='Use reduced dataset')
    parser.add_argument('--balanced', action='store_true', help='Use balanced dataset')
    args = parser.parse_args()

    # Load the data
    if args.reduced:
        gossipcop_df = pd.read_csv('./data/gossipcop' + args.version + '_no_ignore_en_reduced.tsv', sep='\t')
        politifact_df = pd.read_csv('./data/politifact' + args.version + '_no_ignore_en_reduced.tsv', sep='\t')
    elif args.balanced:
        gossipcop_df = pd.read_csv('./data/gossipcop' + args.version + '_no_ignore_en_balanced.tsv', sep='\t')
        politifact_df = pd.read_csv('./data/politifact' + args.version + '_no_ignore_en.tsv', sep='\t')
    else:
        gossipcop_df = pd.read_csv('./data/gossipcop' + args.version + '_no_ignore_en.tsv', sep='\t')
        politifact_df = pd.read_csv('./data/politifact' + args.version + '_no_ignore_en.tsv', sep='\t')

    politifact_df['comments'] = politifact_df['comments'].fillna('')
    gossipcop_df['comments'] = gossipcop_df['comments'].fillna('')

    gossipcop_claims_df = pd.read_csv('./data/gossipcop' + args.version + '_no_ignore_clm.tsv', sep='\t')
    politifact_claims_df = pd.read_csv('./data/politifact' + args.version + '_no_ignore_clm.tsv', sep='\t')
    # remove the rows with empty claims
    gossipcop_claims_df = gossipcop_claims_df[gossipcop_claims_df['claims'].notna()]
    politifact_claims_df = politifact_claims_df[politifact_claims_df['claims'].notna()]

    # Compute the statistics for both datasets
    gossipcop_stats = compute_stats(gossipcop_df, './data/gossipcop' + args.version + '/news_images')
    politifact_stats = compute_stats(politifact_df, './data/politifact' + args.version + '/news_images')

    gossipcop_entity_claims_stats = compute_entity_claims_stats(gossipcop_claims_df)
    politifact_entity_claims_stats = compute_entity_claims_stats(politifact_claims_df)

    # Combine the statistics into a single DataFrame
    stats_df = pd.DataFrame({
        'Platform': ['Politifact', 'Gossipcop'],
        '# Real news': [politifact_stats[0], gossipcop_stats[0]],
        #'# R Images': [politifact_stats[3], gossipcop_stats[3]],
        '# Fake news': [politifact_stats[1], gossipcop_stats[1]],
        #'# F Images': [politifact_stats[4], gossipcop_stats[4]],
        '# Total news': [politifact_stats[2], gossipcop_stats[2]],
        #'# Total news images' : [politifact_stats[5], gossipcop_stats[5]],
        'avg. # comments per news': [round(politifact_stats[6]), round(gossipcop_stats[6])],
        'avg. # entities per news': [round(politifact_stats[7]), round(gossipcop_stats[7])],
        'avg. # entity claims per news': [round(politifact_entity_claims_stats), round(gossipcop_entity_claims_stats)]
    })

    # Display the table
    print(stats_df.to_string(index=False))


