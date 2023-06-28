import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_bar_chart(data_politifact, data_gossipcop, methods, title, ylabel, output_filename):
    x = np.arange(len(methods))
    width = 0.35

    plt.rcParams.update({'font.size': 18})

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.bar(x - width/2, data_politifact, width, label='PolitiFact', color='blue', alpha=0.8)
    ax.bar(x + width/2, data_gossipcop, width, label='GossipCop', color='orange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim([0.50, 0.99])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_filename))

def make_table(df, title, methods, filename):
    df['Method'] = methods
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f'{x:.4f}')
    df['Precision'] = df['Precision'].apply(lambda x: f'{x:.4f}')
    df['Recall'] = df['Recall'].apply(lambda x: f'{x:.4f}')
    df['F1'] = df['F1'].apply(lambda x: f'{x:.4f}')

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
        '\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{4}{c}{\\textbf{' + title + '}}',
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

    df_latex = df.to_latex(index=False, header=False, escape=False)
    df_latex = df_latex.replace("\\begin{tabular}{lllll}", "").replace("\\end{tabular}", "").strip()
    
    latex_content = '\n'.join(latex_header + [df_latex] + latex_footer)
    
    with open(f'data_visualization/tables/{filename}.tex', 'w') as f:
        f.write(latex_content)

output_dir = "data_visualization/barcharts"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('./data_visualization/results.csv')
# Create the 'Configuration' column
data['Configuration'] = data['Embedding'] + '-' + data['Dim reduction method'] + '-' + data['Fusion']
# remove duplicates in politifact_v4
data[(data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow')].drop_duplicates(['Configuration'], keep='last', inplace=True)

# Filter rows to keep only the highest result type
data = data[data['Result type'] == 'TotalAVG']

data['F1'] = (data['Micro F1'] + data['Macro F1']) / 2

embedding_f1_scores_politifact_v4 = []
embedding_f1_scores_gossipcop_v4 = []
reduction_f1_scores_politifact_v4 = []
reduction_f1_scores_gossipcop_v4 = []
fusion_f1_scores_politifact_v4 = []
fusion_f1_scores_gossipcop_v4= []

average_embedding_df_p = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
average_embedding_df_g = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
average_dim_reduction_df_p = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
average_dim_reduction_df_g = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
average_fusion_df_p = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])
average_fusion_df_g = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])

# Iterate over the platforms
for platform in ['politifact_v4', 'gossipcop_v4']:
    platform_data_w_kahan = data[data['Platform'] == platform]
    platform_data_w_kahan = platform_data_w_kahan[platform_data_w_kahan['Classifier'] == 'Shallow']

    print('Platform: {}'.format(platform))

    # drop rows where "KAHAN" is Yes
    platform_data = platform_data_w_kahan[platform_data_w_kahan['KAHAN'] != 'Yes']
    
    # Calculate the average performance for each embedding
    embeddings = platform_data['Embedding'].unique()
    print(embeddings)
    embeddings.sort()
    
    for embedding in embeddings:
        embedding_data = platform_data[platform_data['Embedding'] == embedding]
        
        avg_f1 = embedding_data['F1'].mean()
        avg_accuracy = embedding_data['Accuracy'].mean()
        avg_precision = embedding_data['Precision'].mean()
        avg_recall = embedding_data['Recall'].mean()
        
        average_embedding_df = pd.DataFrame({
            'Method': embedding,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1': avg_f1
        }, index=[0])

        if platform == 'politifact_v4':
            embedding_f1_scores_politifact_v4.append(avg_f1)
            average_embedding_df_p = pd.concat([average_embedding_df_p, average_embedding_df], ignore_index=True)
        else:
            embedding_f1_scores_gossipcop_v4.append(avg_f1)
            average_embedding_df_g = pd.concat([average_embedding_df_g, average_embedding_df], ignore_index=True)

    # Calculate the average performance for each dimensionality reduction method
    dim_reduction_methods = platform_data['Dim reduction method'].unique()
    dim_reduction_methods.sort()
    
    for dim_reduction in dim_reduction_methods:
        dim_reduction_data = platform_data[platform_data['Dim reduction method'] == dim_reduction]

        # exclude rows where the dimensionality reduction method is "--"
        if dim_reduction == '--':
            continue

        avg_f1 = dim_reduction_data['F1'].mean()
        avg_accuracy = dim_reduction_data['Accuracy'].mean()
        avg_precision = dim_reduction_data['Precision'].mean()
        avg_recall = dim_reduction_data['Recall'].mean()

        average_reduction_df = pd.DataFrame({
            'Method': dim_reduction,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1': avg_f1
        }, index=[0])

        if platform == 'politifact_v4':
            reduction_f1_scores_politifact_v4.append(avg_f1)
            average_dim_reduction_df_p = pd.concat([average_dim_reduction_df_p, average_reduction_df], ignore_index=True)
        else:
            reduction_f1_scores_gossipcop_v4.append(avg_f1)
            average_dim_reduction_df_g = pd.concat([average_dim_reduction_df_g, average_reduction_df], ignore_index=True)

    # Calculate the average performance for each fusion method
    fusion_methods = platform_data['Fusion'].unique()
    fusion_methods.sort()
    
    for fusion in fusion_methods:
        fusion_data = platform_data[platform_data['Fusion'] == fusion]

        avg_f1 = fusion_data['F1'].mean()
        avg_accuracy = fusion_data['Accuracy'].mean()
        avg_precision = fusion_data['Precision'].mean()
        avg_recall = fusion_data['Recall'].mean()

        average_fusion_performance = pd.DataFrame({
            'Method': fusion,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1': avg_f1
        }, index=[0])

        if platform == 'politifact_v4':
            fusion_f1_scores_politifact_v4.append(avg_f1)
            average_fusion_df_p = pd.concat([average_fusion_df_p, average_fusion_performance], ignore_index=True)
        else:
            fusion_f1_scores_gossipcop_v4.append(avg_f1)
            average_fusion_df_g = pd.concat([average_fusion_df_g, average_fusion_performance], ignore_index=True)

embedding_methods = ['CLIP', 'CLIP(EA)', 'ResNet50', 'VGG19']
reduction_methods = ['DNN', 'IHAN', 'IHAN(EA)', 'AvgPool', 'FC', 'MaxPool']
fusion_methods = ['Avg', 'Cat', 'ElemMult']

# Image Embedding Methods
plot_bar_chart(embedding_f1_scores_politifact_v4, embedding_f1_scores_gossipcop_v4,
               embedding_methods, 'Image Embedding Methods', 'F1 Score', 'image_embedding_methods.png')
make_table(average_embedding_df_p, 'PolitiFact', embedding_methods, 'average_embedding_performance_p.tex')
make_table(average_embedding_df_g, 'GossipCop', embedding_methods, 'average_embedding_performance_g.tex')

# Dimensionality Reduction Methods
plot_bar_chart(reduction_f1_scores_politifact_v4, reduction_f1_scores_gossipcop_v4,
               reduction_methods, 'Dimensionality Reduction Methods', 'F1 Score', 'dimensionality_reduction_methods.png')
make_table(average_dim_reduction_df_p, 'PolitiFact', reduction_methods, 'average_reduction_performance_p.tex')
make_table(average_dim_reduction_df_g, 'GossipCop', reduction_methods,'average_reduction_performance_g.tex')

# Feature Fusion Methods
plot_bar_chart(fusion_f1_scores_politifact_v4, fusion_f1_scores_gossipcop_v4,
               fusion_methods, 'Feature Fusion Methods', 'F1 Score', 'feature_fusion_methods.png')
make_table(average_fusion_df_p, 'PolitiFact', fusion_methods, 'average_fusion_performance_p.tex')
make_table(average_fusion_df_g, 'GossipCop', fusion_methods, 'average_fusion_performance_g.tex')