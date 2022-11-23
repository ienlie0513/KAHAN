import pandas as pd 

df_en = pd.read_csv('./data/politifact_no_ignore_en.tsv', sep="\t")
df_cap = pd.read_csv('./data/politifact_data_with_captions.tsv', sep="\t").drop_duplicates('id')
df_s = pd.read_csv('./data/politifact_no_ignore_s.tsv', sep="\t")

# get size of dataframes
print(df_en.shape)
print(df_cap.shape)
print(df_s.shape)

# merge column caption from df_cap to df_en by id and keep size of df_en
df_en = df_en.merge(df_cap[['id', 'caption']], on='id', how='left')
print(df_en.shape)
print(df_en.columns)

print(df_en.iloc[[-1]])

# save the merged dataframe to csv
df_en.to_csv('./data/politifact_no_ignore_en_cap.tsv', sep="\t", index=False)