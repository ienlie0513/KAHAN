import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "data_visualization/barcharts"
os.makedirs(output_dir, exist_ok=True)

embedding_methods = ['ResNet-50', 'CLIP', 'CLIP+EA', 'VGG19']
embedding_f1_scores_pf = [0.8631, 0.9535, 0.8915, 0.8824]
embedding_f1_scores_gc = [0.7, 0.75, 0.8, 0.77]

reduction_methods = ['Avg Pool', 'Max Pool', 'FC', 'IHAN+EA', 'IHAN', 'DNN']
reduction_f1_scores_pf = [0.8988, 0.8825, 0.8179, 0.8753, 0.8918, 0.8623]
reduction_f1_scores_gc = [0.7, 0.72, 0.75, 0.77, 0.8, 0.82]

fusion_methods = ['Concat', 'ElemMult', 'Avg']
fusion_f1_scores_pf = [0.9143, 0.8230, 0.8898]
fusion_f1_scores_gc = [0.7, 0.75, 0.8]

x = np.arange(len(embedding_methods))
x_1 = np.arange(len(reduction_methods))
width = 0.35

# Increasing the default size
plt.rcParams.update({'font.size': 20})

# Image Embedding Methods
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, embedding_f1_scores_pf, width, label='PolitiFact', color='b', alpha=0.7)
ax.bar(x + width/2, embedding_f1_scores_gc, width, label='GossipCop', color='orange', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(embedding_methods, rotation=45)
ax.set_title('Image Embedding Methods')
ax.set_ylabel('F1 Score')
ax.set_ylim([0.65, 0.96])  # Adjust y-axis limits here
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'image_embedding_methods.png'))

# Dimensionality Reduction Methods
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x_1 - width/2, reduction_f1_scores_pf, width, label='PolitiFact', color='b', alpha=0.7)
ax.bar(x_1 + width/2, reduction_f1_scores_gc, width, label='GossipCop', color='orange', alpha=0.7)
ax.set_xticks(x_1)
ax.set_xticklabels(reduction_methods, rotation=45)
ax.set_title('Dimensionality Reduction Methods')
ax.set_ylabel('F1 Score')
ax.set_ylim([0.65, 0.96])  # Adjust y-axis limits here
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dimensionality_reduction_methods.png'))

# Feature Fusion Methods
x = np.arange(len(fusion_methods))
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, fusion_f1_scores_pf, width, label='PolitiFact', color='b', alpha=0.7)
ax.bar(x + width/2, fusion_f1_scores_gc, width, label='GossipCop', color='orange', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(fusion_methods, rotation=45)
ax.set_title('Feature Fusion Methods')
ax.set_ylabel('F1 Score')
ax.set_ylim([0.65, 0.96])  # Adjust y-axis limits here
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_fusion_methods.png'))
