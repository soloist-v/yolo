class BaseDataCache:
    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class BaseAugment:

    def __call__(self, last_res, images: BaseDataCache, labels: BaseDataCache):
        raise NotImplementedError


class AugmentSequential(BaseAugment):

    def __init__(self, augments):
        self.augments = augments

    def __call__(self, last_res, images: BaseDataCache, labels: BaseDataCache):
        last = last_res
        for au in self.augments:
            last = au(last, images, labels)
        return last, images, labels
