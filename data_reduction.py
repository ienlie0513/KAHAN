import pandas as pd
import os

import argparse

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Data reduction')
    parser.add_argument('--version', type=str, default='_v4', help='Dataset version to use')
    parser.add_argument('--balance_g', action='store_true', help='balance gossipcop dataset')
    parser.add_argument('--filter_comments', action='store_true', help='filter comments')
    parser.add_argument('--filter_images', action='store_true', help='filter images')
    args = parser.parse_args()

    politifact_df = pd.read_csv('./data/politifact{}_no_ignore_en.tsv'.format(args.version), sep='\t')
    gossipcop_df = pd.read_csv('./data/gossipcop{}_no_ignore_en.tsv'.format(args.version), sep='\t')

    if args.filter_comments or args.filter_images:
        img_path_politifact_real = './data/politifact{}/news_images/real'.format(args.version)
        img_path_politifact_fake = './data/politifact{}/news_images/fake'.format(args.version)

        img_path_gossipcop_real = './data/gossipcop{}/news_images/real'.format(args.version)
        img_path_gossipcop_fake = './data/gossipcop{}/news_images/fake'.format(args.version)

        if args.filter_comments:
            # Remove cases where num comments string (comments seperated by ::) contains at least one comment
            politifact_df = politifact_df[politifact_df['comments'].str.count('::') > 0]
            gossipcop_df = gossipcop_df[gossipcop_df['comments'].str.count('::') > 0]

        if args.filter_images:
            politifact_df['id'] = politifact_df['id'].astype(str)
            gossipcop_df['id'] = gossipcop_df['id'].astype(str)

            # Remove cases where there is no associated article image collected in the folders (match by id)
            politifact_df = politifact_df[politifact_df['id'].isin([f.split('_')[1].split('.')[0] for f in os.listdir(img_path_politifact_real)] + [f.split('_')[1].split('.')[0] for f in os.listdir(img_path_politifact_fake)])]
            gossipcop_df = gossipcop_df[gossipcop_df['id'].isin([f.split('_')[1].split('.')[0] for f in os.listdir(img_path_gossipcop_real)] + [f.split('_')[1].split('.')[0] for f in os.listdir(img_path_gossipcop_fake)])]

        # Write to new tsv files
        politifact_df.to_csv('./data/politifact{}_no_ignore_en_reduced.tsv'.format(args.version), sep='\t', index=False)
        gossipcop_df.to_csv('./data/gossipcop{}_no_ignore_en_reduced.tsv'.format(args.version), sep='\t', index=False)
        
    elif args.balance_g:
        # Undersample the true news articles
        num_fake = len(gossipcop_df[gossipcop_df['label'] == 0])
        
        true_df = gossipcop_df[gossipcop_df['label'] == 1].sample(n=num_fake, random_state=42)
        fake_df = gossipcop_df[gossipcop_df['label'] == 0]

        # Merge the undersampled true news articles and the fake news articles
        balanced_df = pd.concat([true_df, fake_df])
        print(len(balanced_df))

        print(balanced_df['label'].value_counts())
        
        balanced_df.to_csv('./data/gossipcop{}_no_ignore_en_balanced.tsv'.format(args.version), sep='\t', index=False)