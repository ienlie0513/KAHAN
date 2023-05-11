from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk

import argparse

def create_wordcloud(text_list, title, filename):
    # Combine all texts into a single string
    text = ' '.join(text_list)

    # Create a set of English stopwords, and add the single letter 's' to it
    stop_words = set(stopwords.words('english'))
    stop_words.add('s')
    stop_words.add('u')

    # Remove stopwords from the text
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)

    # Generate wordcloud
    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words, min_font_size=10).generate(filtered_text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title)

    # Store to file
    plt.savefig(f'./data_visualization/wordclouds/{filename}.png', dpi=300)

if __name__ == '__main__':
    # Load the data
    # Download the stopwords list
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='_v4', help='Dataset version to use')
    parser.add_argument('--reduced', action='store_true', help='Use reduced dataset')
    args = parser.parse_args()

    if args.reduced:
        gossiocop_df = pd.read_csv('./data/gossipcop{}_no_ignore_en_reduced.tsv'.format(args.version), sep='\t')
        politifact_df = pd.read_csv('./data/politifact{}_no_ignore_en_reduced.tsv'.format(args.version), sep='\t')
    else:
        gossiocop_df = pd.read_csv('./data/gossipcop{}_no_ignore_en.tsv'.format(args.version), sep='\t')
        politifact_df = pd.read_csv('./data/politifact{}_no_ignore_en.tsv'.format(args.version), sep='\t')

    print(len(gossiocop_df))
    print(len(politifact_df))

    politifact_text_real = politifact_df[politifact_df['label'] == 1]['text'].tolist()
    politifact_text_fake = politifact_df[politifact_df['label'] == 0]['text'].tolist()

    gossipcop_text_real = gossiocop_df[gossiocop_df['label'] == 1]['text'].tolist()
    gossipcop_text_fake = gossiocop_df[gossiocop_df['label'] == 0]['text'].tolist()

    if args.reduced:
        create_wordcloud(politifact_text_real, 'PolitiFact Real News', 'politifact{}_reduced_real'.format(args.version))
        create_wordcloud(politifact_text_fake, 'PolitiFact Fake News', 'politifact{}_reduced_fake'.format(args.version))
        create_wordcloud(gossipcop_text_real, 'GossipCop Real News', 'gossipcop{}_reduced_real'.format(args.version))
        create_wordcloud(gossipcop_text_fake, 'GossipCop Fake News', 'gossipcop{}_reduced_fake'.format(args.version))
    else:
        create_wordcloud(politifact_text_real, 'PolitiFact Real News', 'politifact{}_real'.format(args.version))
        create_wordcloud(politifact_text_fake, 'PolitiFact Fake News', 'politifact{}_fake'.format(args.version))
        create_wordcloud(gossipcop_text_real, 'GossipCop Real News', 'gossipcop{}_real'.format(args.version))
        create_wordcloud(gossipcop_text_fake, 'GossipCop Fake News', 'gossipcop{}_fake'.format(args.version))


