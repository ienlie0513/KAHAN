import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import nltk

# Download the stopwords list
nltk.download('stopwords')
from nltk.corpus import stopwords


def create_wordcloud(text_list, title):
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
    plt.savefig(f'./data_visualization/wordclouds/{title}.png', dpi=300)

gossiocop_df = pd.read_csv('./data/gossipcop_no_ignore_en.tsv', sep='\t')
politifact_df = pd.read_csv('./data/politifact_v4_no_ignore_s.tsv', sep='\t')

print(len(gossiocop_df))
print(len(politifact_df))

politifact_text_real = politifact_df[politifact_df['label'] == 1]['text'].tolist()
politifact_text_fake = politifact_df[politifact_df['label'] == 0]['text'].tolist()

gossipcop_text_real = gossiocop_df[gossiocop_df['label'] == 1]['text'].tolist()
gossipcop_text_fake = gossiocop_df[gossiocop_df['label'] == 0]['text'].tolist()

create_wordcloud(politifact_text_real, 'PolitiFact Real News')
create_wordcloud(politifact_text_fake, 'PolitiFact Fake News')

create_wordcloud(gossipcop_text_real, 'GossipCop Real News')
create_wordcloud(gossipcop_text_fake, 'GossipCop Fake News')


