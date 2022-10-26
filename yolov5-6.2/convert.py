import argparse
import os
import shutil
from pathlib import Path
import yaml
from PIL import ExifTags, Image
import cv2
import numpy as np
from voc_xml_parse import VocXML, Object, Bndbox


def imread(file):
    data = np.fromfile(file, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def get_ubbox(boxa, boxb):
    return min(boxa[0], boxb[0]), min(boxa[1], boxb[1]), max(boxa[2], boxb[2]), max(boxa[3], boxb[3])


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def bbox2yolo(size, box):
    width, height = size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def walk_dir(dirname):
    if os.path.isfile(dirname):
        yield dirname, dirname
        return
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


def yolo2bbox(size, xywh):
    xcen, ycen, w, h = xywh
    xmin = max(float(xcen) - float(w) / 2, 0)
    xmax = min(float(xcen) + float(w) / 2, 1)
    ymin = max(float(ycen) - float(h) / 2, 0)
    ymax = min(float(ycen) + float(h) / 2, 1)

    xmin = int(size[0] * xmin)
    xmax = int(size[0] * xmax)
    ymin = int(size[1] * ymin)
    ymax = int(size[1] * ymax)

    return xmin, ymin, xmax, ymax


def yolo2xml(img_name, names):
    base_name = os.path.splitext(img_name)[0]
    txt_name = f"{base_name}.txt"
    if not os.path.exists(txt_name):
        return
    img = imread(img_name)
    if img is None:
        return
    size = img.shape[:2][::-1]
    content = open(txt_name, 'rb').read().decode("utf8")
    xml = VocXML.create()
    objs = xml.annotation.objects
    for line in content.split("\n"):
        line = line.strip()
        if not len(line):
            continue
        t = line.split()
        if len(t) != 5:
            continue
        cls, x, y, w, h = t
        cls = int(cls)
        x, y, w, h = map(float, [x, y, w, h])
        box = yolo2bbox(size, (x, y, w, h))
        try:
            objs.append(Object.create(names[cls], Bndbox.create(*box)))
        except:
            print(cls, txt_name, line)
            raise
    xml_path = f"{base_name}.xml"
    xml.save(xml_path)


def get_img_size(im_file):
    im = Image.open(im_file)
    try:
        im.verify()  # PIL verify
    except Exception as e:
        print(e)
        return None
    size = exif_size(im)  # image size
    return size


def xml2yolo(img_name, xml_name, name_map):
    if not Path(xml_name).exists():
        return b''
    size = get_img_size(img_name)
    if size is None:
        return
    xml = VocXML(xml_name)
    lines = []
    for obj in xml.annotation.objects:
        box = obj.bndbox.bbox
        x, y, w, h = convert(size, box)
        name = obj.name
        if name not in name_map:
            continue
        cls = name_map[name]
        line = " ".join(map(str, [cls, x, y, w, h]))
        lines.append(line)
    content = "\n".join(lines).encode("utf8")
    return content


def walk_img(dirname):
    for name in os.listdir(dirname):
        _, ext = os.path.splitext(name)
        if ext.lower() not in img_formats:
            continue
        filepath = os.path.join(dirname, name)
        yield filepath


def create_classes(save_dir, names):
    save_name = os.path.join(save_dir, "classes.txt")
    open(save_name, 'wb').write("\n".join(names).encode('utf8'))


def xml2yolo_dir():
    dirname = r'D:\Workspace\C++\tensorRT_Pro\simple_yolo\workspace\screenshots'
    names = ["head", "person", "zombie"]
    name_map = {k: i for i, k in enumerate(names)}
    for img_path in walk_img(dirname):
        base_name = os.path.splitext(img_path)[0]
        xml_name = f'{base_name}.xml'
        txt_name = f"{base_name}.txt"
        content = xml2yolo(img_path, xml_name, name_map)
        open(txt_name, 'wb').write(content)
    create_classes(dirname, names)


def rm_dir_interactive(dirname, force_clear):
    if Path(dirname).exists() and len(os.listdir(dirname)):
        print(f"目标文件夹: {dirname} 不为空")
        if force_clear:
            shutil.rmtree(dirname)
            return
        choose = input("是否清除 y/n ?").strip()
        if choose.lower() in ["y", "yes"]:
            shutil.rmtree(dirname)


def auto_convert(names, img_dir, xml_dir, save_dir, is_pair, force_clear):
    name_map = {k: i for i, k in enumerate(names)}
    images_dir = os.path.join(save_dir, "images")
    labels_dir = os.path.join(save_dir, "labels")
    img_train_dir = os.path.join(images_dir, "train")
    label_train_dir = os.path.join(labels_dir, "train")
    img_val_dir = os.path.join(images_dir, "val")
    label_val_dir = os.path.join(labels_dir, "val")
    rm_dir_interactive(images_dir, force_clear)
    rm_dir_interactive(labels_dir, force_clear)
    for dirname in [images_dir, labels_dir, img_train_dir, img_val_dir, label_train_dir, label_val_dir]:
        os.makedirs(dirname, exist_ok=True)
    last_img = None
    last_label = None
    for name, img_path, xml_path in get_pair_sample_from_dir(img_dir, xml_dir):
        base_name, ext = os.path.splitext(name)
        if is_pair:
            if not Path(xml_path).exists():
                continue
        content = xml2yolo(img_path, xml_path, name_map)
        if content is None:
            continue
        print(img_path)
        save_txt_path = os.path.join(label_train_dir, f"{base_name}.txt")
        save_image_path = os.path.join(img_train_dir, name)
        open(save_txt_path, 'wb').write(content)
        shutil.copy(img_path, save_image_path)
        last_img = img_path
        last_label = save_txt_path
    # last_base_name_img = os.path.split(last_img)[-1]
    # last_base_name_label = os.path.split(last_label)[-1]
    # shutil.copy(last_img, os.path.join(img_val_dir, last_base_name_img))
    # shutil.copy(last_label, os.path.join(label_val_dir, last_base_name_label))


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def load_names(names_file):
    ext = os.path.splitext(names_file)[-1].lower()
    if ext == ".json":
        import json
        data = json.loads(names_file)
        names = data['names']
    elif ext == ".yaml":
        import yaml
        with open(names_file, 'rb') as f:
            data = yaml.load(f, yaml.FullLoader)
            names = data['names']
    else:
        content = open(names_file, 'rb').read().decode("utf8")
        names = content.split()
    return names


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action='store_true')
    return parser.parse_args()


img_formats = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng'}

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = "convert.yaml"
    cfg = yaml.load(open(config_path, 'rb').read(), yaml.FullLoader)
    dataset_name = cfg['dataset_name']
    names = cfg['names']
    img_dir = cfg['img_dir']
    xml_dir = cfg['xml_dir']
    save_dir = cfg['save_dir']
    is_pair = cfg['is_pair']
    force_clear = cfg['force_clear']
    auto_convert(names, img_dir, xml_dir, save_dir, is_pair, force_clear)
    res = f"""
path: {Path(save_dir).as_posix()}  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/train  # val images (relative to 'path') 128 images
test:  # test images (optional)
nc: {len(names)}  # number of classes
names: {names}
download: # optional
"""
    print(res)
    os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
    open(dataset_name, 'wb').write(res.encode('utf8'))
