import torch
import argparse
import pickle
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from random import randint
import numpy as np

# Parse arguments

# em
# # Save embeddings and labels to a pickle file
# data = {'embeddings': embeddings, 'labels': labels}
# with open('./FeyaFSCIL/logs/session1.pkl', 'wb') as f:
#     pickle.dump(data, f)

pkl_path = './FeyaFSCIL/logs/session0.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

embeddings = data['feats']
labels = data['label']

#select_classes=[i for i in range(99)]
#select_classes=[16,29,7,55,47,1,92,93]
select_classes=[1,7,16,29,47,55]
print(select_classes)
select_embeddings = []
select_labels = []
for cls in select_classes:
    for idx, item in enumerate(labels):
        if int(item) == cls:
            select_embeddings.append(embeddings[idx])
            select_labels.append(labels[idx])
embeddings = np.array(select_embeddings)
labels = np.array(select_labels)
#if(lebels=1)
print(embeddings.shape)
print(labels.shape)
print(labels)
# exit()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
embeddings_tsne = tsne.fit_transform(embeddings)

# Visualize t-SNE embeddings
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='tab10', s=10)

legend_labels = ['C-1','C-7','C-16','C-29','C-47', 'C-55']
scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='tab10', s=10)
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='lower right')
plt.title('Session-0 visualization of embeddings')

# Remove horizontal and vertical labels
plt.xticks([])
plt.yticks([])

plt.show()

plt.savefig('./FeyaFSCIL/tsne.png', bbox_inches='tight')
