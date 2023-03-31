import re
import json
import os

import pandas as pd

from textwrap import TextWrapper

log_file_politifact = open('data_analysis/dataset_summary_p.txt', 'w')
log_file_gossipcop = open('data_analysis/dataset_summary_g.txt', 'w')

def log_dataset_content(_dir, sub_dir, log_file):
    # create a log file for datanalysis
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            idx = re.findall(r'\d+', folder)[0]
            with open(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json') as f:
                try:
                    file_content = json.load(f)
                    log_file.write('{}: {} \r'.format(idx, file_content['text'][:1000]))
                except Exception as e:
                    print(e)
                    log_file.write('{}: {} \r'.format(idx, 'Error loading file'))

def log_tsv_file_content(_dir, platform, log_file):
    tw = TextWrapper()
    tw.width = 100

    df = pd.read_csv(_dir + '/' + platform + '_no_ignore_en.tsv', sep='\t')
    for _, row in df.iterrows():
        log_file.write('{}:\r{} \r'.format(row['id'], '\n'.join(tw.wrap(row['text'][:1000]))))
        log_file.write('Comments: \r')

        for cmt in row['comments'].split('::'):
            log_file.write('{} \r'.format(cmt.split('<>')[0]))
            log_file.write('-' * 50 + '\r')

        log_file.write('-' * 100 + '\r')


#log_tsv_file_content('./data', 'politifact_v2', log_file_politifact, True)
log_tsv_file_content('./data', 'gossipcop', log_file_gossipcop)