from augmentations.base import BaseDataCache


class YoloMemeryCache(BaseDataCache):
    def __init__(self, images, labels):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass


class YoloDiskCache(BaseDataCache):

    def __init__(self, images_path, labels_path):
        self.images = images_path
        self.labels = labels_path

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass


class YoloV5DataCache(BaseDataCache):
    def __init__(self, func_len, func_item, func_iter):
        self.func_len = func_len
        self.func_iter = func_iter
        self.func_item = func_item

    def __getitem__(self, item):
        return self.func_item(item)

    def __len__(self):
        return self.func_len()

    def __iter__(self):
        return self.func_iter()
