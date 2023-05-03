import re
import json
import os

from collections import defaultdict
import pandas as pd

from textwrap import TextWrapper

log_file_politifact_1 = open('data_analysis/dataset_summary_real_p.txt', 'w')
log_file_gossipcop_1 = open('data_analysis/dataset_summary_real_g.txt', 'w')


log_file_politifact_0 = open('data_analysis/dataset_summary_fake_p.txt', 'w')
log_file_gossipcop_0 = open('data_analysis/dataset_summary_fake_g.txt', 'w')

def log_dataset_content(_dir, sub_dir, log_file):
    tw = TextWrapper()
    tw.width = 100
    
    # create a log file for datanalysis
    content_freq = defaultdict(list)
    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            if folder == '.DS_Store':
                continue
            idx = re.findall(r'\d+', folder)[0]
            try:
                with open(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json') as f:
                    try:
                        file_content = json.load(f) 
                        content = file_content['text'][:500].strip().replace('\n', ' ')
                        content_freq[content].append(idx)
                    except Exception as e:
                        print(e)
                        log_file.write('{}: {} \r'.format(idx, 'Error loading file'))
            except Exception as e:
                print(e)
                log_file.write('{}: {} \r'.format(idx, 'File does not exist'))
    
    # log the content frequency
    sorted_content_freq = sorted(content_freq.items(), key=lambda x: (-len(x[1]), len(x[0])), reverse=True)
    for i, (content, ids) in enumerate(sorted_content_freq):
        id_str = ', '.join(ids)
        log_file.write('{} ({} occurrences) {}'.format(content, len(ids), id_str))
        if i != len(sorted_content_freq) - 1:
            log_file.write('\n' + '-'*100 + '\n\n')

def log_tsv_file_content(_dir, platform, log_file, label=1):
    tw = TextWrapper()
    tw.width = 100

    df = pd.read_csv(_dir + '/' + platform + '_no_ignore_s.tsv', sep='\t')
    print(len(df))
    df = df.dropna(subset=['text'])
    print(len(df))

    for _, row in df.iterrows():
        if row['label'] == label:
            log_file.write('{}:\r{} \r'.format(row['id'], '\n'.join(tw.wrap(row['text'][:1000]))))
            # log_file.write('Comments: \r')

            # for cmt in row['comments'].split('::'):
            #     log_file.write('{} \r'.format(cmt.split('<>')[0]))
            #     log_file.write('-' * 50 + '\r')

            log_file.write('-' * 100 + '\r')


#log_tsv_file_content('./data', 'politifact_v2', log_file_politifact, True)
# log_tsv_file_content('./data', 'gossipcop', log_file_gossipcop_1, 1)
# #log_tsv_file_content('./data', 'politifact_v2', log_file_politifact_1, 1)

# log_tsv_file_content('./data', 'gossipcop', log_file_gossipcop_0, 0)
#log_tsv_file_content('./data', 'politifact_v2', log_file_politifact_0, 0)


log_file_gossipcop_v1 = open('data_analysis/dataset_summary_gossipcop_v1.txt', 'w')
log_file_politifact_v1 = open('data_analysis/dataset_summary_politifact_v1.txt', 'w')

log_file_gossipcop_v3 = open('data_analysis/dataset_summary_gossipcop_v3.txt', 'w')
log_file_politifact_v3 = open('data_analysis/dataset_summary_politifact_v3.txt', 'w')

log_file_politifact_v4 = open('data_analysis/dataset_summary_politifact_v4.txt', 'w')

# log_dataset_content('./fakenewsnet_dataset', 'politifact', log_file_politifact_v1)
# log_dataset_content('./fakenewsnet_dataset', 'gossipcop', log_file_gossipcop_v1)

# log_dataset_content('./fakenewsnet_dataset_v3', 'politifact', log_file_politifact_v3)
# log_dataset_content('./fakenewsnet_dataset_v3', 'gossipcop', log_file_gossipcop_v3)

log_dataset_content('./fakenewsnet_dataset_v4', 'politifact', log_file_politifact_v4)


# GossipCop news that were removed
gossipcop_removed = [7259603960, 843758, 911401, 923847, 948472, 875314, 8081058062, 3663225246, 3066088888, 4495586215, 2778253719, 2507720803, 4024992311, 6489461221, 1615677123, 5111151830, 5328116357, 9328842988, 4888675652, 6899501174, 8172018375, 3172946389, 5034259222, 843265, 2008641429, 4466125915, 5480976970, 5408886782, 4360448788, 1457904080, 2254004589, 1628848955, 92814312, 4802653039, 267792336, 3893905315, 8410631215, 5241899341, 948328, 887769, 888688, 860652,  919774, 951548, 947692, 948524, 942374]
# Politifact news that were removed
politifact_removed = [14920, 15108, 14498, 15129, 13711, 13576, 7540, 279, 65, 13900, 200, 10332, 320, 7506, 772, 12120, 2624, 773, 11960, 14225, 6267, 954, 7923, 681, 211, 620, 401, 370, 13058, 11399, 8119, 13058, 11399, 8119, 620, 8470, 14498, 15129, 13576, 7540, 279, 65, 13900, 200, 7506, 772, 12120, 773, 2624, 11960, 14225, 6267, 954, 7923, 1213, 681, 211, 8470, 620, 401]