import pandas as pd
import numpy as np

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('./data_visualization/results.csv')

# Filter rows where 'Result type' is 'Highest'
data = data[data['Result type'] == 'TotalAVG']

# Add new column for average F1 score
data['F1'] = (data['Micro F1'] + data['Macro F1']) / 2

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
data[(data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow')] = data[(data['Platform'] == 'politifact_v4') & (data['Classifier'] == 'Shallow')].drop_duplicates(['Configuration'], keep='last')

politifact = data[(data['Platform'] == 'politifact_v4')]
gossipcop = data[data['Platform'] == 'gossipcop_v4']

politifact_top5 = politifact[politifact['Classifier'] == 'Shallow'].sort_values(by='F1', ascending=False).head(10)
gossipcop_top5 = gossipcop[gossipcop['Classifier'] == 'Shallow'].sort_values(by='F1', ascending=False).head(10)

print('Politifact top 5:')
print(politifact_top5[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1']])
print(politifact[(politifact['KAHAN'] == 'Yes') & (politifact['Classifier'] == 'Shallow')][['Accuracy', 'Precision', 'Recall', 'F1']])
print('Gossipcop top 5:')
print(gossipcop_top5[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1']])
print(gossipcop[(gossipcop['KAHAN'] == 'Yes') & (gossipcop['Classifier'] == 'Shallow')][['Accuracy', 'Precision', 'Recall', 'F1']])