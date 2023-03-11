import pandas as pd 

df_en_v2 = pd.read_csv('./data/politifact_v2_no_ignore_en.tsv', sep="\t")
df_s_v3 = pd.read_csv('./data/politifact_v3_no_ignore_s.tsv', sep="\t")
df_s = pd.read_csv('./data/politifact_no_ignore_s.tsv', sep="\t")

# get size of dataframes
# print(df_en.shape)
# print(df_cap[["caption"]].notna().sum())
# print(df_s.shape)
# print(df_s.shape)

# merge column caption from df_cap to df_en by id and keep size of df_en
# df_en = df_en_v2.merge(df_s_v3[['id', 'comments']], on='id', how='left')
df_en_v2['comments'] = df_en_v2['comments'].combine_first(df_s_v3['comments']) 
print(df_en_v2.shape)
print(df_en_v2.columns)
print(df_en_v2.isna().sum())

print(df_en_v2['comments'].iloc[[-3]])

# save the merged dataframe to csv
df_en_v2.to_csv('./data/politifact_v2_no_ignore_en.tsv', sep="\t", index=False)