import pandas as pd
from urllib.parse import urlparse, parse_qs
from collections import Counter

import os
import json
import re

def get_titles(_dir, sub_dir):
    urls = []
    for folder in os.listdir(_dir + '/' + sub_dir):
        try:
            with open(_dir + '/' + sub_dir + '/' + folder + '/news content.json') as f:
                file_content = json.load(f)
                urls.append(file_content['url'])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {folder}/news content.json: {e}")
    return urls


politifact_path = './fakenewsnet_dataset_v2/politifact'

fake_data = get_titles(politifact_path, 'fake')
real_data = get_titles(politifact_path, 'real')

def extract_domain(url):
    if 'web.archive.org' in url:
        url = re.findall(r'\d{14}/(.*)', url)[0]
    domain = urlparse(url).netloc
    return domain

def get_top_domains(urls, n=10):
    domains = [extract_domain(url) for url in urls]
    domain_counts = Counter(domains)
    return domain_counts.most_common(n)

# Get the top 10 domains for both datasets
real_top_domains = get_top_domains(real_data, 10)
fake_top_domains = get_top_domains(fake_data, 10)

# Display the results
print("Top 10 domains in real dataset:")
for domain, count in real_top_domains:
    print(f"{domain}: {count}")

print("\nTop 10 domains in fake dataset:")
for domain, count in fake_top_domains:
    print(f"{domain}: {count}")
