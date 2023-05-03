import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "data_visualization/barcharts"
os.makedirs(output_dir, exist_ok=True)

embedding_methods = ['VGG19', 'ResNet-50', 'CLIP', 'CLIP+EA']
embedding_accuracies_pf = [0.8, 0.85, 0.9, 0.87]
embedding_accuracies_gc = [0.7, 0.75, 0.8, 0.77]

reduction_methods = ['FC', 'DNN', 'Max Pool', 'Avg Pool', 'IHAN', 'IHAN+EA']
reduction_accuracies_pf = [0.8, 0.82, 0.85, 0.87, 0.9, 0.92]
reduction_accuracies_gc = [0.7, 0.72, 0.75, 0.77, 0.8, 0.82]

fusion_methods = ['Avg', 'Concat', 'ElemMult']
fusion_accuracies_pf = [0.8, 0.85, 0.9]
fusion_accuracies_gc = [0.7, 0.75, 0.8]

x = np.arange(len(embedding_methods))
x_1 = np.arange(len(reduction_methods))
width = 0.35

# Image Embedding Methods
fig, ax = plt.subplots()
ax.bar(x - width/2, embedding_accuracies_pf, width, label='PolitiFact')
ax.bar(x + width/2, embedding_accuracies_gc, width, label='GossipCop')
ax.set_xticks(x)
ax.set_xticklabels(embedding_methods)
ax.set_title('Image Embedding Methods')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig(os.path.join(output_dir, 'image_embedding_methods.png'))

# Dimensionality Reduction Methods
fig, ax = plt.subplots()
ax.bar(x_1 - width/2, reduction_accuracies_pf, width, label='PolitiFact')
ax.bar(x_1 + width/2, reduction_accuracies_gc, width, label='GossipCop')
ax.set_xticks(x_1)
ax.set_xticklabels(reduction_methods)
ax.set_title('Dimensionality Reduction Methods')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig(os.path.join(output_dir, 'dimensionality_reduction_methods.png'))

# Feature Fusion Methods
x = np.arange(len(fusion_methods))

fig, ax = plt.subplots()
ax.bar(x - width/2, fusion_accuracies_pf, width, label='PolitiFact')
ax.bar(x + width/2, fusion_accuracies_gc, width, label='GossipCop')
ax.set_xticks(x)
ax.set_xticklabels(fusion_methods)
ax.set_title('Feature Fusion Methods')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig(os.path.join(output_dir, 'feature_fusion_methods.png'))

