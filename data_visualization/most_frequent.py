import os
import json
import pandas as pd

def get_most_frequent_content(_dir, sub_dir):
    text_freq = {}

    for sub_sub_dir in ['fake', 'real']:
        for folder in os.listdir(_dir + '/' + sub_dir + '/' + sub_sub_dir):
            try:
                with open(_dir + '/' + sub_dir + '/' + sub_sub_dir + '/' + folder + '/news content.json') as f:
                    content = json.load(f)
                    text = content['text']

                    if len(text) > 0:
                        if text in text_freq:
                            text_freq[text] += 1
                        else:
                            text_freq[text] = 1
            except Exception as e:
                print(f"Error parsing JSON in {folder}/news content.json: {e}")

    df = pd.DataFrame.from_dict(text_freq, orient='index', columns=['frequency'])

    df_top10 = df.sort_values(by='frequency', ascending=False).head(10)

    return df_top10

pd.options.display.max_colwidth = 70

df_p = get_most_frequent_content('./fakenewsnet_dataset', 'politifact')
df_g = get_most_frequent_content('./fakenewsnet_dataset', 'gossipcop')

print(df_p)
print(df_g)


