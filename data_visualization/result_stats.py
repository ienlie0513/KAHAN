import pandas as pd

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('./data_visualization/results.csv')

# Filter rows to keep only the highest result type
data = data[data['Result type'] == 'Highest']

# Get the unique platforms in the data
platforms = data['Platform'].unique()

# Iterate over the platforms
for platform in platforms:
    platform_data_w_kahan = data[data['Platform'] == platform]

    print('Platform: {}'.format(platform))

    # drop rows where "KAHAN" is Yes
    platform_data = platform_data_w_kahan[platform_data_w_kahan['KAHAN'] != 'Yes']
    
    # Calculate the average performance for each fusion method
    fusion_methods = platform_data['Fusion'].unique()
    average_fusion_performance = pd.DataFrame(columns=['Fusion', 'Accuracy', 'Precision', 'Recall', 'Micro F1', 'Macro F1'])
    
    for fusion in fusion_methods:
        fusion_data = platform_data[platform_data['Fusion'] == fusion]
        avg_accuracy = fusion_data['Accuracy'].mean()
        avg_precision = fusion_data['Precision'].mean()
        avg_recall = fusion_data['Recall'].mean()
        avg_micro_f1 = fusion_data['Micro F1'].mean()
        avg_macro_f1 = fusion_data['Macro F1'].mean()
        
        average_fusion_df = pd.DataFrame({
            'Fusion': fusion,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'Micro F1': avg_micro_f1,
            'Macro F1': avg_macro_f1
        }, index=[0])
        
        average_fusion_performance = pd.concat([average_fusion_performance, average_fusion_df], ignore_index=True)

    print("Average Performance for Fusion Methods")
    print(average_fusion_performance)
    print()
    
    # Calculate the average performance for each embedding
    embeddings = platform_data['Embedding'].unique()
    average_embedding_performance = pd.DataFrame(columns=['Embedding', 'Accuracy', 'Precision', 'Recall', 'Micro F1', 'Macro F1'])
    
    for embedding in embeddings:
        embedding_data = platform_data[platform_data['Embedding'] == embedding]
        avg_accuracy = embedding_data['Accuracy'].mean()
        avg_precision = embedding_data['Precision'].mean()
        avg_recall = embedding_data['Recall'].mean()
        avg_micro_f1 = embedding_data['Micro F1'].mean()
        avg_macro_f1 = embedding_data['Macro F1'].mean()
        
        average_embedding_df = pd.DataFrame({
            'Embedding': embedding,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'Micro F1': avg_micro_f1,
            'Macro F1': avg_macro_f1
        }, index=[0])
        
        average_embedding_performance = pd.concat([average_embedding_performance, average_embedding_df], ignore_index=True)

    print("Average Performance for Embeddings")
    print(average_embedding_performance)
    print()
    
    # Calculate the average performance for each dimensionality reduction method
    dim_reduction_methods = platform_data['Dim reduction method'].unique()
    average_dim_reduction_performance = pd.DataFrame(columns=['Dim reduction method', 'Accuracy', 'Precision', 'Recall', 'Micro F1', 'Macro F1'])
    
    for dim_reduction in dim_reduction_methods:
        dim_reduction_data = platform_data[platform_data['Dim reduction method'] == dim_reduction]

        # exclude rows where the dimensionality reduction method is "--"
        if dim_reduction == '--':
            continue

        avg_accuracy = dim_reduction_data['Accuracy'].mean()
        avg_precision = dim_reduction_data['Precision'].mean()
        avg_recall = dim_reduction_data['Recall'].mean()
        avg_micro_f1 = dim_reduction_data['Micro F1'].mean()
        avg_macro_f1 = dim_reduction_data['Macro F1'].mean()
        
        average_dim_reduction_df = pd.DataFrame({
            'Dim reduction method': dim_reduction,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'Micro F1': avg_micro_f1,
            'Macro F1': avg_macro_f1
        }, index=[0])

        average_dim_reduction_performance = pd.concat([average_dim_reduction_performance, average_dim_reduction_df], ignore_index=True)

    print("Average Performance for Dimensionality Reduction Methods")
    print(average_dim_reduction_performance)
    print()

    # Compute the top 5 configurations
    top_configurations = platform_data.groupby(['Embedding', 'Dim reduction method', 'Fusion']).mean().nlargest(5, 'Accuracy')
    top_configurations = top_configurations[['Accuracy', 'Precision', 'Recall', 'Micro F1', 'Macro F1']]
    print("Top 5 Configurations:")
    print(top_configurations)
    print()

    # Filter rows for KAHAN configuration
    kahan_data = platform_data_w_kahan[(platform_data_w_kahan['KAHAN'] == 'Yes')]

    # Print performance for KAHAN
    print("Performance for KAHAN:")
    print(kahan_data[['Accuracy', 'Precision', 'Recall', 'Micro F1', 'Macro F1']])
    print()
