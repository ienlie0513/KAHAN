import os
import json
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from urllib.parse import urlparse


def get_titles(_dir, sub_dir):
    titles = []
    for folder in os.listdir(_dir + '/' + sub_dir):
        try:
            with open(_dir + '/' + sub_dir + '/' + folder + '/news content.json') as f:
                file_content = json.load(f)
                titles.append(file_content['title'])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {folder}/news content.json: {e}")
    return titles


politifact_path = './fakenewsnet_dataset_v2/politifact'

fake_data = get_titles(politifact_path, 'fake')
real_data = get_titles(politifact_path, 'real')


# Extract the 'title' field of each item in the dataset
politifact_titles = fake_data + real_data

# Vectorize the 'title' field using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(politifact_titles)

# Cluster the titles using K-means
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

# Count the number of items in each cluster
cluster_counts = Counter(kmeans.labels_)

# Print the results
for i in range(n_clusters):
    print(f"Cluster {i}: {cluster_counts[i]} items")

# Function to extract domain name from URL
def get_domain_name(url):
    parsed_uri = urlparse(url)
    domain = '{uri.netloc}'.format(uri=parsed_uri)
    return domain

# Function to get titles and domain names
def get_titles_and_domains(_dir, sub_dir):
    titles = []
    domains = []
    for folder in os.listdir(_dir + '/' + sub_dir):
        try:
            with open(_dir + '/' + sub_dir + '/' + folder + '/news content.json') as f:
                file_content = json.load(f)
                titles.append(file_content['title'])
                domains.append(get_domain_name(file_content['url']))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {folder}/news content.json: {e}")
    return titles, domains

fake_data, fake_domains = get_titles_and_domains(politifact_path, 'fake')
real_data, real_domains = get_titles_and_domains(politifact_path, 'real')

politifact_titles = fake_data + real_data
politifact_domains = fake_domains + real_domains

# Group titles by domain
titles_by_domain = defaultdict(list)
for title, domain in zip(politifact_titles, politifact_domains):
    titles_by_domain[domain].append(title)

# Count the number of items in each domain
domain_counts = {domain: len(titles) for domain, titles in titles_by_domain.items()}

# Print the results
for domain, count in domain_counts.items():
    print(f"Domain {domain}: {count} items")

# Dimensionality reduction with t-SNE
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(politifact_titles)
tsne = TSNE(random_state=42)
X_embedded = tsne.fit_transform(X.todense())

# Assign a cluster ID based on domain
domain_to_cluster_id = {domain: i for i, domain in enumerate(titles_by_domain.keys())}
cluster_ids = [domain_to_cluster_id[domain] for domain in politifact_domains]

# Visualize the clusters
plt.figure(figsize=(12, 8))
for domain, cluster_id in domain_to_cluster_id.items():
    cluster_points = X_embedded[np.array(cluster_ids) == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"{domain}")

plt.legend()
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Visualization of Clusters by Domain")
plt.show()





