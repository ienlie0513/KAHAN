import matplotlib.pyplot as plt
import pandas as pd

def create_bar_chart(dataset_counts, title):
    labels = ['Real News', 'Fake News']
    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, dataset_counts, align='center')
    ax.set_ylabel('Number of Articles')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Store to file
    plt.savefig(f'./data_visualization/barcharts/{title}.png', dpi=300)

def create_pie_chart(dataset_counts, title):
    labels = ['Real News', 'Fake News']

    fig, ax = plt.subplots()
    ax.pie(dataset_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
    ax.set_title(title)

    # Store to file
    plt.savefig(f'./piecharts/{title}.png', dpi=300)

gossiocop_df = pd.read_csv('./data/gossipcop_no_ignore_en.tsv', sep='\t')
politifact_df = pd.read_csv('./data/politifact_v4_no_ignore_s.tsv', sep='\t')



politifact_counts = [len(politifact_df[politifact_df['label'] == 1]), len(politifact_df[politifact_df['label'] == 0])]
gossipcop_counts = [len(gossiocop_df[gossiocop_df['label'] == 1]), len(gossiocop_df[gossiocop_df['label'] == 0])]

create_bar_chart(politifact_counts, 'PolitiFact Dataset')
create_bar_chart(gossipcop_counts, 'GossipCop Dataset')