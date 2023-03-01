import re
import json
import os

import pandas as pd

# create a log file for datanalysis
log_file = open('data_analysis/dataset_summary.txt', 'w')

def log_dataset_content(_dir, sub_dir):
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            idx = re.findall(r'\d+', folder)[0]
            with open(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json') as f:
                try:
                    file_content = json.load(f)
                    log_file.write('{}: {} \r'.format(idx, file_content['text'][:100]))
                except Exception as e:
                    print(e)
                    log_file.write('{}: {} \r'.format(idx, 'Error loading file'))

def log_tsv_file_content(_dir, platform):
    df = pd.read_csv(_dir + '/' + platform + '_no_ignore_en.tsv', sep='\t')
    for _, row in df.iterrows():
        log_file.write('{}: {} \r'.format(row['id'], row['text'][:100]))

log_tsv_file_content('./data', 'politifact_v2')