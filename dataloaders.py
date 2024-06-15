from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler
import random
import numpy as np

class MultiDatasetBatchSampler(Sampler):
    def __init__(self, datasets, batch_size):
        self.batch_size = batch_size
        self.datasets = datasets
        self.total_images = sum([len(d) for d in datasets])
        self.mini_batch_sizes = [int(round(len(d) / self.total_images * batch_size)) for d in datasets]
        self.indices = []
        for i, dataset in enumerate(datasets):
            indices = np.arange(len(dataset))
            if i > 0:
                indices += len(datasets[i - 1])
            self.indices.append(indices)
        for indices in self.indices:
            random.shuffle(indices)
        
    def __len__(self):
        return self.total_images // self.batch_size
    
    def __iter__(self):
        batch = []
        for batch_index in range(len(self)):
            for dataset_i, mini_batch_size in enumerate(self.mini_batch_sizes):
                batch.extend(
                    self.indices[dataset_i][mini_batch_size * batch_index:mini_batch_size * (batch_index + 1)]
                )
            yield batch
            batch = []


def collate_fn(batch):
    batch_by_dataset = defaultdict(lambda: defaultdict(list))
    for item in batch:
        batch_by_dataset[item["dataset_name"]]["x"].append(
            item["image"]
        )
        batch_by_dataset[item["dataset_name"]]["y"].append(
            item["mask"]
        )
    for dataset_name, mini_batch in batch_by_dataset.items():
        batch_by_dataset[dataset_name]["x"] = torch.stack(mini_batch["x"])
        batch_by_dataset[dataset_name]["y"] = torch.stack(mini_batch["y"])
    return dict(batch_by_dataset)