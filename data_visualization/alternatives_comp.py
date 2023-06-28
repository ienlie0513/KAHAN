import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Increasing the default size
plt.rcParams.update({'font.size': 18})

# Define a plotting function
def plot_comparison(df1, df2, title, filename, ylim=(0.80, 1), labels=['Shallow', 'Deep'], colors=['orange', 'blue']):
    plt.figure(figsize=(12, 8), dpi=300)
    bar_width = 0.35

    # Merge the two dataframes and calculate the mean F1 score
    merged = pd.merge(df1, df2, on="Configuration", suffixes=('_1', '_2'))
    merged['mean_F1'] = (merged['F1_1'] + merged['F1_2']) / 2

    # Sort the dataframe by the mean F1 score
    merged.sort_values('Configuration', inplace=True)

    # Create the indices for the bars
    index1 = np.arange(len(merged))
    index2 = [i+bar_width for i in index1]
    
    # Plot the bars with passed colors
    bars1 = plt.bar(index1, merged['F1_1'], bar_width, label=labels[0], color=colors[0], alpha=0.8)
    bars2 = plt.bar(index2, merged['F1_2'], bar_width, label=labels[1], color=colors[1], alpha=0.8)

    # Add the line graph
    line1, = plt.plot(index1 + bar_width/2, merged['F1_1'], marker='o', color=colors[0], linestyle='solid', linewidth=3, markersize=10)
    line2, = plt.plot(index1 + bar_width/2, merged['F1_2'], marker='*', color=colors[1], linestyle='dashed', linewidth=3, markersize=10)

    plt.title(title)
    plt.xlabel('Configuration')
    plt.ylabel('F1 Score')
    plt.ylim(*ylim)
    plt.xticks(index1 + bar_width / 2, merged['Configuration'], rotation=45, ha='right')
    plt.tight_layout()

    # Add a legend
    plt.legend([line1, line2], [f'{labels[0]}', f'{labels[1]}'])

    # Save the figure before showing it
    plt.savefig(f'./data_visualization/barcharts/{filename}.png', format='png', dpi=300)


def make_table(df1, df2, titles, title, filename, colors=['orange', 'blue']):
    df1_final = df1[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1']]
    df2_final = df2[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1']]
    
    df1_final['Accuracy'] = df1_final['Accuracy'].apply(lambda x: f'{x:.4f}')
    df1_final['Precision'] = df1_final['Precision'].apply(lambda x: f'{x:.4f}')
    df1_final['Recall'] = df1_final['Recall'].apply(lambda x: f'{x:.4f}')
    df1_final['F1'] = df1_final['F1'].apply(lambda x: f'{x:.3f}')
    
    df2_final['Accuracy'] = df2_final['Accuracy'].apply(lambda x: f'{x:.4f}')
    df2_final['Precision'] = df2_final['Precision'].apply(lambda x: f'{x:.4f}')
    df2_final['Recall'] = df2_final['Recall'].apply(lambda x: f'{x:.4f}')
    df2_final['F1'] = df2_final['F1'].apply(lambda x: f'{x:.4f}')
    
    df_merged = pd.merge(df1_final, df2_final, on='Configuration', suffixes=('_df1', '_df2'))
    
    for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
        df_merged[col] = df_merged.apply(lambda row: 
                                        f'\\textbf{{{row[col+"_df1"]}}}/{row[col+"_df2"]}' if float(row[col+"_df1"]) > float(row[col+"_df2"]) else 
                                        f'{row[col+"_df1"]}/\\textbf{{{row[col+"_df2"]}}}' if float(row[col+"_df2"]) > float(row[col+"_df1"]) else 
                                        f'{row[col+"_df1"]}/{row[col+"_df2"]}', 
                                        axis=1)

    df_final = df_merged[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1']]

    latex_header = [
        '\\documentclass{article}',
        '\\usepackage{booktabs, tabularx}',  # Add booktabs style
        '\\usepackage[left=0.2in, right=0.2in, top=0.2in]{geometry}',  # Reducing default margins
        '\\begin{document}',
        '\\begin{table}[ht]',
        '\\centering',
        '\\caption{' + title + '}',  # Add title as caption
        '\\begin{tabularx}{\\textwidth}{l *{4}{X}}',  # Use tabularx to make table fit page width
        '\\hline',
        '\\multirow{2}{*}{\\textbf{Configuration}} & \\multicolumn{4}{c}{\\textbf{' + title + '}}',
        '\\\\ \\cline{2-5}',
        '& \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1}',
        '\\\\ \\hline',
    ]
    latex_footer = [
        '\\hline',
        '\\end{tabularx}',
        '\\end{table}',
        '\\end{document}'
    ]

    df_latex = df_final.to_latex(index=False, header=False, escape=False)
    df_latex = df_latex.replace("\\begin{tabular}{lllll}", "").replace("\\end{tabular}", "").strip()
    
    latex_content = '\n'.join(latex_header + [df_latex] + latex_footer)
    
    with open(f'data_visualization/tables/{filename}.tex', 'w') as f:
        f.write(latex_content)



# Load the CSV data into a pandas DataFrame
init_data = pd.read_csv('./data_visualization/results.csv')

file_tags = ['high', 'lastavg', 'totalavg']
result_types = ['Highest', 'LastAVG', 'TotalAVG']

for i in range(len(result_types)):
    # Filter rows where 'Result type' is 'Highest'
    data = init_data[init_data['Result type'] == result_types[i]]
    file_tag = file_tags[i]

    # Add new column for average F1 score
    data['F1'] = (data['Micro F1'] + data['Macro F1']) / 2

    # Filter out KAHAN
    data = data[data['KAHAN'] == '--']

    # Create the 'Configuration' column
    data['Configuration'] = data['Embedding'] + '-' + data['Dim reduction method'] + '-' + data['Fusion']

    # Define a mapping of configuration name replacements
    mapping = {
        'CLIP /w EA----cat': 'CLIP(EA)-Cat',
        'CLIP----cat': 'CLIP-Cat',
        'resnet50-DNN [504,202,202,202,50]-avg': 'Resnet50-DNN-Avg',
        'resnet50-DNN [504,202,202,202,50]-cat': 'Resnet50-DNN-Cat',
        'resnet50-DNN [504,202,202,202,50]-elem_mult': 'Resnet50-DNN-ElemMult',
        'resnet50-IHAN /w EA-avg': 'Resnet50-IHAN(EA)-Avg',
        'resnet50-IHAN /w EA-cat': 'Resnet50-IHAN(EA)-Cat',
        'resnet50-IHAN /w EA-elem_mult': 'Resnet50-IHAN(EA)-ElemMult',
        'resnet50-IHAN-avg': 'Resnet50-IHAN-Avg',
        'resnet50-IHAN-cat': 'Resnet50-IHAN-Cat',
        'resnet50-IHAN-elem_mult': 'Resnet50-IHAN-ElemMult',
        'resnet50-avgpooling-avg': 'Resnet50-AvgPool-Avg',
        'resnet50-avgpooling-cat': 'Resnet50-AvgPool-Cat',
        'resnet50-avgpooling-elem_mult': 'Resnet50-AvgPool-ElemMult',
        'resnet50-fc-avg': 'Resnet50-FC-Avg',
        'resnet50-fc-cat': 'Resnet50-FC-Cat',
        'resnet50-fc-elem_mult': 'Resnet50-FC-ElemMult',
        'resnet50-maxpooling-avg': 'Resnet50-MaxPool-Avg',
        'resnet50-maxpooling-cat': 'Resnet50-MaxPool-Cat',
        'resnet50-maxpooling-elem_mult': 'Resnet50-MaxPool-ElemMult',
        'vgg19-DNN [504,202,202,202,50]-avg': 'VGG19-DNN-Avg',
        'vgg19-DNN [504,202,202,202,50]-cat': 'VGG19-DNN-Cat',
        'vgg19-DNN [504,202,202,202,50]-elem_mult': 'VGG19-DNN-ElemMult',
        'vgg19-IHAN /w EA-avg': 'VGG19-IHAN(EA)-Avg',
        'vgg19-IHAN /w EA-cat': 'VGG19-IHAN(EA)-Cat',
        'vgg19-IHAN /w EA-elem_mult': 'VGG19-IHAN(EA)-ElemMult',
        'vgg19-IHAN-avg': 'VGG19-IHAN-Avg',
        'vgg19-IHAN-cat': 'VGG19-IHAN-Cat',
        'vgg19-IHAN-elem_mult': 'VGG19-IHAN-ElemMult',
        'vgg19-avgpooling-avg': 'VGG19-AvgPool-Avg',
        'vgg19-avgpooling-cat': 'VGG19-AvgPool-Cat',
        'vgg19-avgpooling-elem_mult': 'VGG19-AvgPool-ElemMult',
        'vgg19-fc-avg': 'VGG19-FC-Avg',
        'vgg19-fc-cat': 'VGG19-FC-Cat',
        'vgg19-fc-elem_mult': 'VGG19-FC-ElemMult',
        'vgg19-maxpooling-avg': 'VGG19-MaxPool-Avg',
        'vgg19-maxpooling-cat': 'VGG19-MaxPool-Cat',
        'vgg19-maxpooling-elem_mult': 'VGG19-MaxPool-ElemMult'
    }

    # Create the 'Configuration' column by mapping the names
    data['Configuration'] = data['Configuration'].map(mapping)

    data = data[['Platform', 'Classifier', 'Configuration', 'Accuracy', 'Precision', 'Recall', 'F1']].drop_duplicates(['Platform', 'Classifier', 'Configuration'])
    # remove duplicates in politifact_v4
    data[(data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow')] = data[(data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow')].drop_duplicates(['Configuration'], keep='last')

    ###### EXPERIMENT 2 ######

    # Create pandas dataframes from the data
    politifact = data[data['Platform'] == 'politifact_v4']
    gossipcop = data[data['Platform'] == 'gossipcop_v4']

    p_avg_df = politifact[politifact['Classifier'] == 'Shallow'].sort_values(by='F1', ascending=False)#.head(10)
    g_avg_df = gossipcop[gossipcop['Classifier'] == 'Shallow'].sort_values(by='F1', ascending=False)#.head(10)

    # Create a dataframe for the shallow and deep results
    p_shallow_df = politifact[politifact['Configuration'].isin(p_avg_df['Configuration']) & (politifact['Classifier'] == 'Shallow')].sort_values(by='Configuration')
    p_deep_df = politifact[politifact['Configuration'].isin(p_avg_df['Configuration']) & (politifact['Classifier'] == 'Deep')].sort_values(by='Configuration')

    # Create a dataframe for the shallow and deep results
    g_shallow_df = gossipcop[gossipcop['Configuration'].isin(g_avg_df['Configuration']) & (gossipcop['Classifier'] == 'Shallow')].sort_values(by='Configuration')
    g_deep_df = gossipcop[gossipcop['Configuration'].isin(g_avg_df['Configuration']) & (gossipcop['Classifier'] == 'Deep')].sort_values(by='Configuration')

    ###### EXPERIMENT 3 ######

    p_fake_plus = data[(data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow')]
    p_fake = data[(data['Platform'] == 'politifact') & (data['Classifier'] == 'Shallow')]
    g_fake_plus = data[(data['Platform'] == 'gossipcop_v4') & (data['Classifier'] == 'Shallow')]
    g_fake = data[(data['Platform'] == 'gossipcop') & (data['Classifier'] == 'Shallow')]

    p_fake_plus = p_fake_plus[p_fake_plus['Configuration'].isin(p_avg_df['Configuration'])].sort_values(by='Configuration')
    g_fake_plus = g_fake_plus[g_fake_plus['Configuration'].isin(g_avg_df['Configuration'])].sort_values(by='Configuration')

    p_fake = p_fake[p_fake['Configuration'].isin(p_avg_df['Configuration'])].sort_values(by='Configuration')
    g_fake = g_fake[g_fake['Configuration'].isin(g_avg_df['Configuration'])].sort_values(by='Configuration')

    ###### GossipCop vs PolitiFact ######
    gossipcop_comp = data[((data['Platform'] == 'gossipcop_v4') & (data['Classifier'] == 'Shallow'))]
    politifact_comp = data[((data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow'))]

    combo = pd.concat([gossipcop_comp, politifact_comp])
    # find the top 10 avergage best performing configurations
    avg_df = combo.groupby(['Configuration']).mean().sort_values(by='F1', ascending=False).head(10)
    print('FileTag: {} {}'.format(file_tag, avg_df))

    # Create a dataframe for each platform
    gossipcop_comp = gossipcop_comp[gossipcop_comp['Configuration'].isin(avg_df.index)].sort_values(by='Configuration')
    politifact_comp = politifact_comp[politifact_comp['Configuration'].isin(avg_df.index)].sort_values(by='Configuration')

    # # Generate bar plots
    plot_comparison(g_shallow_df, g_deep_df, 'GossipCop', 'gossipcop_exp2_{}'.format(file_tag), (0.7, 0.90), ['Shallow', 'Deep'], colors=['skyblue', 'navy'])
    plot_comparison(p_shallow_df, p_deep_df, 'PolitiFact', 'politifact_exp2_{}'.format(file_tag), (0.85, 0.95), ['Shallow', 'Deep'], colors=['skyblue', 'navy'])

    plot_comparison(g_fake_plus, g_fake, 'GossipCop', 'gossipcop_exp3_{}'.format(file_tag), (0.7, 0.90), ['FakeNewsNet+', 'FakeNewsNet'], colors=['darkorange', 'purple'])
    plot_comparison(p_fake_plus, p_fake, 'PolitiFact', 'politifact_exp3_{}'.format(file_tag), (0.8, 0.95), ['FakeNewsNet+', 'FakeNewsNet'], colors=['darkorange', 'purple'])

    plot_comparison(politifact_comp, gossipcop_comp, 'GossipCop vs PolitiFact', 'gossipcop_vs_politifact_{}'.format(file_tag), (0.7, 0.95), ['PolitiFact', 'GossipCop'], colors=['blue', 'orange'])

    # Generate tables
    make_table(g_shallow_df, g_deep_df, titles=['Shallow', 'Deep'], title='GossipCop', filename='gossipcop_exp2_table_{}'.format(file_tag))
    make_table(p_shallow_df, p_deep_df, titles=['Shallow', 'Deep'], title='PolitiFact', filename='politifact_exp2_table_{}'.format(file_tag))

    make_table(g_fake_plus, g_fake, titles=['FakeNewsNet+', 'FakeNewsNet'], title='GossipCop', filename='gossipcop_exp3_table_{}'.format(file_tag))
    make_table(p_fake_plus, p_fake, titles=['FakeNewsNet+', 'FakeNewsNet'], title='PolitiFact', filename='politifact_exp3_table_{}'.format(file_tag))

    make_table(politifact_comp, gossipcop_comp, titles=['PolitiFact', 'GossipCop'], title='PolitiFact vs GossipCop', filename='gossipcop_vs_politifact_table_{}'.format(file_tag))