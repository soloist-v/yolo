import time

import cv2
import random
from voc_xml_parse import VocXML, get_pair_sample_from_dir
from convert import imread
import numpy as np

'''
0.000997781753540039
0.0009980201721191406
'''


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def rect_in_rect(boxa, boxb, iou_threshold=0.15):
    boxa = np.array(boxa)
    boxb = np.array(boxb)
    wa, ha = np.abs(np.diff(boxa.reshape(-1, 2), axis=0)[0])
    wb, hb = np.abs(np.diff(boxb.reshape(-1, 2), axis=0)[0])
    a_area = ha * wa
    b_area = hb * wb
    inter = (np.min([boxa[2:], boxb[2:]], axis=0) - np.max([boxa[:2], boxb[:2]], axis=0)).clip(0).prod()
    iou = inter / b_area
    return iou >= iou_threshold


def bbox2yolo(size, box):
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


def get_weights(cx, indexes: np.ndarray, width):
    # weights = width / np.maximum(np.abs(indexes - cx), 0.1)
    weights = width / (np.abs(indexes - cx) + 0.01)
    return weights


def softmax(x):
    x -= np.max(x)
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def rand_cut(img, labels, wh, threshold=0.13):
    ih, iw, _ = img.shape
    xi = np.arange(iw)
    yi = np.arange(ih)
    x_weights = np.full(iw, 0.01)
    y_weights = np.full(ih, 0.01)
    for cls, cx, cy, w, h in labels:
        cx = cx * iw
        cy = cy * ih
        w = w * iw
        h = h * ih
        kx = wh[0] / w
        ky = wh[1] / h
        wxi = get_weights(cx, xi, w) * kx
        wyi = get_weights(cy, yi, h) * ky
        x_weights += wxi
        y_weights += wyi
    tx = random.choices(xi, x_weights, k=1)[0]
    ty = random.choices(yi, y_weights, k=1)[0]
    half_w = wh[0] * 0.5
    half_h = wh[1] * 0.5

    dx_min = max(half_w - tx, 0)
    dy_min = max(half_h - ty, 0)
    dx_max = -max(half_w - (iw - tx), 0)
    dy_max = -max(half_h - (ih - ty), 0)
    offset_x = dx_max + dx_min
    offset_y = dy_max + dy_min

    tx += offset_x
    ty += offset_y

    rect_x1 = max(tx - half_w, 0)
    rect_y1 = max(ty - half_h, 0)
    rect_x2 = min(tx + half_w, iw)
    rect_y2 = min(ty + half_h, ih)

    size = (iw, ih)
    new_labels = []
    for cls, cx, cy, w, h in labels:
        x1, y1, x2, y2 = yolo2bbox(size, (cx, cy, w, h))
        if not rect_in_rect([rect_x1, rect_y1, rect_x2, rect_y2], [x1, y1, x2, y2], threshold):
            continue
        x1 = max(x1, rect_x1)
        y1 = max(y1, rect_y1)
        x2 = min(x2, rect_x2)
        y2 = min(y2, rect_y2)
        new_labels.append([cls, *bbox2yolo(size, (x1, y1, x2, y2))])
    new_labels = np.array(new_labels).reshape((-1, 5))
    rect = img[int(rect_y1): int(rect_y2), int(rect_x1): int(rect_x2)]
    return rect, new_labels


if __name__ == '__main__':
    img_dir = r"D:\Workspace\yolo\cv_mix\data\ciping_data"
    label_dir = r"D:\Workspace\yolo\cv_mix\data\ciping_data"
    # res = np.random.choice(np.arange(10 * 10), (2, 3))
    # print(res)
    # exit(0)
    data = list(get_pair_sample_from_dir(img_dir, label_dir))
    index = 0
    while True:
        name, img_path, label_path = data[index]
        index = (index + 1) % len(data)
        img = imread(img_path)
        xml = VocXML(label_path)
        h, w, _ = img.shape
        xi = np.arange(w)
        yi = np.arange(h)
        x_weights = np.zeros(w)
        y_weights = np.zeros(h)

        for o in xml.annotation.objects:
            cx = o.bndbox.cx
            cy = o.bndbox.cy
            kx = 416 / o.bndbox.width
            ky = 416 / o.bndbox.height
            wxi = get_weights(cx, xi, w) * kx
            wyi = get_weights(cy, yi, h) * ky
            x_weights += wxi
            y_weights += wyi
        show = np.zeros_like(img[:, :, 0], dtype=float)
        for i in range(h):
            for j in range(w):
                show[i, j] = y_weights[i] + x_weights[j]
        show = show / np.max(show) * 255
        cv2.imshow("heatmap", auto_resize(show.astype("uint8"), 800, 800)[0])
        cv2.waitKey(0)
        # tx = np.random.choice(xi, 1, p=softmax(x_weights))[0]
        # ty = np.random.choice(yi, 1, p=softmax(y_weights))[0]
        tx = random.choices(xi, x_weights, k=1)[0]
        ty = random.choices(yi, y_weights, k=1)[0]

        cv2.circle(img, (tx, ty), 5, (0, 0, 255), 5)
        cv2.imshow("res", auto_resize(img, 800, 800)[0])
        if cv2.waitKey(0) == 27:
            break
        # break
        # random.choices()
    cv2.destroyAllWindows()
