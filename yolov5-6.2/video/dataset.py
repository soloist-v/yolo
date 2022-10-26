import os
import random

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch


def gauss_sample(data, k):
    m = max(data) // 2
    mu = float(np.mean(data))
    sigma = float(np.var(data))
    print(mu, sigma)
    data_set = set()
    for i in range(k):
        while True:
            d = int(random.gauss(mu, sigma)) + m
            if d not in data_set:
                break
        data_set.add(d)
    return sorted(data_set)


def rand_sample(data, k):
    return sorted(random.sample(data, k))


class VideoData(Dataset):
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.names = os.listdir(video_dir)
        self.frame_count = 64

    def load_data(self, name):
        dirname = os.path.join(self.video_dir, name)
        names = os.listdir(dirname)
        temp = []
        for name in names:
            basename, ext = os.path.splitext(name)
            if ext not in (".data",):
                continue
            idx = int(basename)
            temp.append([idx, name])
        temp = sorted(temp, key=lambda x: x[0])
        _, data = zip(*temp)
        idxes = rand_sample(range(len(data)), self.frame_count)
        dataset = []
        for idx in idxes:
            name = data[idx]
            filepath = os.path.join(dirname, name)
            arr = torch.load(open(filepath, 'rb'))
            dataset.append(arr)
        delta = max(len(dataset) - self.frame_count, 0)
        left = delta // 2
        right = delta - left
        t = dataset[0]
        for i in range(left):
            d = torch.zeros_like(t)
            dataset.insert(0, d)
        for i in range(right):
            d = torch.zeros_like(t)
            dataset.append(d)
        return torch.stack(dataset, dim=4)

    def __getitem__(self, index):
        video = self.load_data(self.names[index])
        return video

    @staticmethod
    def collate_fn(batch):
        imgs, labels, src_imgs = zip(*batch)  # transposed
        return torch.stack(imgs, 0), labels, src_imgs


if __name__ == '__main__':
    res = rand_sample(range(128), 64)
    print(res)
