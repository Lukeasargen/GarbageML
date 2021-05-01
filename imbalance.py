import torch
from torch.utils.data.sampler import Sampler


"""https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""
class ImbalancedSampler(Sampler):
    def __init__(self, dataset, indices):
        self.indices = indices
        self.length = len(indices)

        # Calculate weights by inverting the counts
        # Need the dataset to get the labels
        count_per_class = [0]*len(dataset.classes)
        for idx in self.indices:
            label = dataset.targets[idx]
            count_per_class[label] += 1
        # Hopefully no classes have zero counts
        # This is the class count against the total samples
        weight_per_class = [len(indices) / count_per_class[i] for i in range(len(dataset.classes))]
        # Make the class weight align with the class label of the indicies
        weights = [weight_per_class[dataset.targets[self.indices[i]]] for i in range(len(self.indices))]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.length, replacement=True))

    def __len__(self):
        return self.length
