import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
from collections import Counter

class SignsDataset(Dataset):
    def __init__(self, file, folder, signs_subset=None, transform=None):
        signs = pd.read_csv(file)

        if signs_subset is not None:
            assert isinstance(signs_subset, (list, tuple))
            mask = signs['class_number'].isin(signs_subset)
            signs = signs[mask]
        else:
            signs_subset = np.unique(self.signs['class_number'])

        self.filenames = signs['filename']
        self.labels = signs['class_number']
        self.transform = transform
        self.folder = folder
        self.mapping = {label: i for i, label in enumerate(np.unique(self.labels))}

        N, K = len(self.labels), len(self.mapping)
        counter = Counter()
        counter.update(self.labels)
        counts = [(self.mapping[label], cnt) for label, cnt in counter.items()]
        self.weights = [1] * K


    def __getitem__(self, i):
        filename = self.filenames.iloc[i]
        label = self.labels.iloc[i]
        path = os.path.join(self.folder, filename)
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.mapping[label]

    def __len__(self):
        return len(self.filenames)

    def get_weights(self):
        return torch.FloatTensor(self.weights)
