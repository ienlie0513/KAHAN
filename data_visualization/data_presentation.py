import os
import pandas as pd

def compute_stats(df, img_dir):
    true_news_count = len(df[df['label'] == 1])
    fake_news_count = len(df[df['label'] == 0])
    total_news_count = len(df)

    # number of images
    true_news_images = len(os.listdir(img_dir + '/real'))
    fake_news_images = len(os.listdir(img_dir + '/fake'))
    total_news_images = true_news_images + fake_news_images
    
    avg_comments_per_news = df['comments'].apply(lambda x: len(x.split('::'))).mean()
    # only compute for non empty entities
    avg_entities_per_news = df[df['entities'].notna()]['entities'].apply(lambda x: len(x.split('||'))).mean()
    
    return true_news_count, fake_news_count, total_news_count, total_news_images, avg_comments_per_news, avg_entities_per_news

def compute_entity_claims_stats(df):
    avg_entity_claims_per_news = df['claims'].apply(lambda x: len(x.split('||'))).mean()
    return avg_entity_claims_per_news

# Load the data
gossipcop_df = pd.read_csv('./data/gossipcop_no_ignore_en.tsv', sep='\t')
politifact_df = pd.read_csv('./data/politifact_v4_no_ignore_en.tsv', sep='\t')

politifact_df['comments'] = politifact_df['comments'].fillna('')

#gossipcop_claims_df = pd.read_csv('./data/gossipcop_v4_no_ignore_clm.tsv', sep='\t')
#politifact_claims_df = pd.read_csv('./data/politifact_v4_no_ignore_clm.tsv', sep='\t')
# remove the rows with empty claims
#gossipcop_claims_df = gossipcop_claims_df[gossipcop_claims_df['claims'].notna()]
#politifact_claims_df = politifact_claims_df[politifact_claims_df['claims'].notna()]

# Compute the statistics for both datasets
gossipcop_stats = compute_stats(gossipcop_df, './data/gossipcop/news_images')
politifact_stats = compute_stats(politifact_df, './data/politifact_v4/news_images')

#gossipcop_entity_claims_stats = compute_entity_claims_stats(gossipcop_claims_df)
#politifact_entity_claims_stats = compute_entity_claims_stats(politifact_claims_df)

# Combine the statistics into a single DataFrame
stats_df = pd.DataFrame({
    'Platform': ['Politifact', 'Gossipcop'],
    '# True news': [politifact_stats[0], gossipcop_stats[0]],
    '# Fake news': [politifact_stats[1], gossipcop_stats[1]],
    '# Total news': [politifact_stats[2], gossipcop_stats[2]],
    '# Images': [politifact_stats[3], gossipcop_stats[3]], # same as total news
    'avg. # comments per news': [round(politifact_stats[4]), round(gossipcop_stats[4])],
    'avg. # entities per news': [round(politifact_stats[5]), round(gossipcop_stats[5])],
    #'avg. # entity claims per news': [round(politifact_entity_claims_stats), round(gossipcop_entity_claims_stats)]
})

# Display the table
print(stats_df.to_string(index=False))


