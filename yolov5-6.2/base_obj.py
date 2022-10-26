import numpy as np
import time


def calc_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p2) - np.array(p1)) ** 2))


class BaseObj:
    def __init__(self, **kwargs):
        self.time = time.time()
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return self.__dict__.__contains__(key)

    def __repr__(self):
        return f"Baseobj<id:{self.id}, label:{self['label']}, class_type:{self['class_type']}>"

    __str__ = __repr__

    @property
    def class_type(self):
        return self["class_type"] or None

    @property
    def poses(self):
        return self["poses"] or {}

    @property
    def poses_number(self):
        return self["poses_number"] or {}

    @property
    def poses_objs(self):
        return self["poses_objs"] or {}

    @property
    def frame_no(self):
        return self["frame_no"] or 0

    @property
    def id(self):
        return self['id'] or id(self)

    @property
    def id_str(self):
        return self['id'] or ""

    @property
    def label(self):
        return self['label']

    @property
    def bbox(self):
        return self['bbox']

    @bbox.setter
    def bbox(self, bbox):
        self['bbox'] = bbox

    @property
    def bbox_score(self):
        return self['bbox_score']

    @bbox_score.setter
    def bbox_score(self, bbox):
        self['bbox_score'] = bbox

    @property
    def center(self):
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    @property
    def center_int(self):
        return int((self.bbox[0] + self.bbox[2]) / 2), int((self.bbox[1] + self.bbox[3]) / 2)

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    xmin = x_min = x1
    xmax = x_max = x2

    ymin = y_min = y1
    ymax = y_max = y2

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def dist_diagonal(self):
        """计算对角线距离"""
        return calc_dist(self.bbox[:2], self.bbox[2:4])

    @property
    def info(self):
        return self['info']

    def copy(self):
        return BaseObj(**self.__dict__.copy())


if __name__ == '__main__':
    import time

    obj = BaseObj(a=10, b=100, bbox=np.array([1, 2, 3, 4, ]), label='dasdsadas')
    t0 = time.time()
    for i in range(100):
        obj.copy()
    print(time.time() - t0)  # 0.20909404754638672
