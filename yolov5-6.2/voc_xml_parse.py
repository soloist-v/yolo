import os
from pathlib import Path
from xmltodict import unparse, parse
from collections import OrderedDict
from typing import List, Union
import numpy as np

img_formats = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng'}


def calc_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p2) - np.array(p1)) ** 2))


class Base:

    def __init__(self, data):
        self.data = data

    def copy(self):
        cls = self.__class__
        return cls(self.data.copy()) if hasattr(self.data, "copy") else cls(self.data)


class Size(Base):

    @staticmethod
    def create(width: int, height: int):
        data = OrderedDict(width=width, height=height)
        return Size(data)

    @property
    def width(self) -> int:
        return int(self.data['width'])

    @width.setter
    def width(self, val: int):
        self.data['width'] = val

    @property
    def height(self) -> int:
        return int(self.data['height'])

    @height.setter
    def height(self, val: int):
        self.data['height'] = val


class Bndbox(Base):

    @staticmethod
    def create(xmin: Union[float, int], ymin: Union[float, int], xmax: Union[float, int], ymax: Union[float, int]):
        data = OrderedDict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        return Bndbox(data)

    @property
    def xmin(self) -> float:
        return float(self.data['xmin'])

    @xmin.setter
    def xmin(self, val: Union[float, int]):
        self.data['xmin'] = str(val)

    @property
    def ymin(self) -> float:
        return float(self.data['ymin'])

    @ymin.setter
    def ymin(self, val: Union[float, int]):
        self.data['ymin'] = str(val)

    @property
    def xmax(self) -> float:
        return float(self.data["xmax"])

    @xmax.setter
    def xmax(self, val: Union[float, int]):
        self.data['xmax'] = val

    @property
    def ymax(self) -> float:
        return float(self.data['ymax'])

    @ymax.setter
    def ymax(self, val: Union[float, int]):
        self.data['ymax'] = val

    @property
    def bbox(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def diagonal(self):
        return calc_dist(self.bbox[:2], self.bbox[2:4])

    @property
    def x1(self):
        return self.xmin

    @property
    def x2(self):
        return self.xmax

    @property
    def y1(self):
        return self.ymin

    @property
    def y2(self):
        return self.ymax

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    @property
    def cx(self):
        return (self.xmax + self.xmin) / 2

    @property
    def cy(self):
        return (self.ymin + self.ymax) / 2


class Object(Base):

    @staticmethod
    def create(name: str, bndbox: Bndbox) -> "Object":
        data = OrderedDict(name=name, bndbox=bndbox.data)
        return Object(data)

    @property
    def name(self) -> str:
        return self.data['name']

    @name.setter
    def name(self, val: str):
        self.data['name'] = val

    @property
    def bndbox(self):
        return Bndbox(self.data['bndbox'])

    @bndbox.setter
    def bndbox(self, val: Bndbox):
        self.data['bndbox'] = val.data


class Objects(Base):
    def __init__(self, data):
        if not isinstance(data, list):
            data = [data]
        super().__init__(data)
        self.data: list

    @staticmethod
    def create(objects: List[Object] = ()):
        data = []
        o = Objects(data)
        o.extend(objects)
        return o

    def __getitem__(self, idx):
        return Object(self.data[idx])

    def __setitem__(self, idx, value: Object):
        self.data[idx] = value.data

    def __len__(self):
        return len(self.data)

    def append(self, node: Object):
        self.data.append(node.data)

    def pop(self, idx=None):
        self.data.pop(idx)

    def clear(self):
        self.data.clear()

    def extend(self, ls: List[Object]):
        for obj in ls:
            self.append(obj)

    def remove(self, obj: Object):
        rm_s = []
        for i, item in enumerate(self.data):
            if obj.data is item:
                rm_s.append(i)
        for i in reversed(rm_s):
            self.data.pop(i)


class Annotation(Base):

    @staticmethod
    def create(size: Size, objects: Objects):
        data = OrderedDict(size=size.data, object=objects.data)
        return Annotation(data)

    @property
    def size(self):
        return Size(self.data['size'])

    @size.setter
    def size(self, new: Size):
        self.data['size'] = new.data

    @property
    def objects(self):
        if "object" not in self.data:
            self.data['object'] = []
        obj_ls = self.data["object"]
        if isinstance(obj_ls, OrderedDict):
            obj_ls = [obj_ls]
            self.data['object'] = obj_ls
        return Objects(self.data['object'])

    @objects.setter
    def objects(self, new: Objects):
        self.data['object'] = new.data


class VocXML(Base):
    def __init__(self, label: Union[str, OrderedDict]):
        if isinstance(label, str):
            data = parse(open(label, 'rb').read())
        else:
            data = label
        super().__init__(data)

    @staticmethod
    def create(path="", width=0, height=0, depth=3):
        folder, filename = os.path.split(path)
        obj_ls = []  # 如果匹配就创建一个新的
        xml_dict = OrderedDict([('annotation', OrderedDict(
            [('folder', folder),
             ('filename', filename),
             ('path', path),
             ('source', OrderedDict([('database', 'Unknown')])),
             ('size', OrderedDict([('width', width), ('height', height), ('depth', depth)])),
             ('segmented', '0'),
             ('object', obj_ls)]))])
        return VocXML(xml_dict)

    @property
    def annotation(self):
        return Annotation(self.data['annotation'])

    @annotation.setter
    def annotation(self, new: Annotation):
        self.data["annotation"] = new.data

    def save(self, path=None) -> OrderedDict:
        data = unparse(self.data)
        if path:
            open(path, 'wb').write(data.encode('utf8'))
        return data


def walk_dir(dirname):
    for n in os.listdir(dirname):
        path = os.path.join(dirname, n)
        if os.path.isfile(path):
            yield n, path
        else:
            yield from walk_dir(path)


def get_pair_sample_from_dir(img_dir, label_dir):
    img_dir, label_dir = Path(img_dir).as_posix(), Path(label_dir).as_posix()
    img_dir_len = len(img_dir)
    for name, img_path in walk_dir(img_dir):
        prefix, ext = os.path.splitext(img_path)
        if ext not in img_formats:
            continue
        label_path = "%s%s.xml" % (label_dir, prefix[img_dir_len:])
        yield name, img_path, label_path
