import os
import random
from typing import List
import cv2
import torch
import numpy as np
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (check_img_size, scale_coords, non_max_suppression)
from utils.torch_utils import select_device


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        label = str(label)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot(frame, labels, boxes, scores, src_shape):
    src_h, src_w, *_ = src_shape
    h, w, d = frame.shape
    mx = w / src_w
    my = h / src_h
    for label, box, score in zip(labels, boxes, scores):
        label = '%s %.2f' % (label, score)
        bbox = box[0] * mx, box[1] * my, box[2] * mx, box[3] * my
        plot_one_box(bbox, frame, label=label, color=(0, 255, 0), line_thickness=1)


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def calc_iou(boxa, boxb):
    boxa = np.array(boxa)
    boxb = np.array(boxb)
    wa, ha = np.abs(np.diff(boxa.reshape(-1, 2), axis=0)[0])
    wb, hb = np.abs(np.diff(boxb.reshape(-1, 2), axis=0)[0])
    a_area = ha * wa
    b_area = hb * wb
    inter = (np.min([boxa[2:], boxb[2:]], axis=0) - np.max([boxa[:2], boxb[:2]], axis=0)).clip(0).prod()
    union = a_area + b_area - inter
    iou = inter / union
    return iou


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def preprocess(img, dw, dh):
    h, w, d = img.shape
    ratio = min(dw / w, dh / h, 1)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h))
    res = cv2.copyMakeBorder(img_resized, 0, dh - new_h, 0, dw - new_w, cv2.BORDER_CONSTANT, value=114)
    return res


def post_process(im, src_shape, data: List[torch.Tensor], iou_threshold):
    data[..., 5:] *= data[..., 4:5]
    data = data[np.max(data[..., 5:], -1) > 0.5]  # [[],[],[]] => [0.8, 0.9, ]

    grid_classes = {}
    for grid in data:
        cls = np.argmax(grid[5:])
        grid[4] = grid[5:][cls]
        grid = grid[:5]
        if cls not in grid_classes:
            grid_classes[cls] = [grid]
        else:
            grid_classes[cls].append(grid)
    labels = []
    boxes = []
    for cls, grids in grid_classes.items():
        grids = sorted(grids, reverse=True, key=lambda x: x[4])
        while True:
            if len(grids) == 0:
                break
            t = grids.pop(0)
            x, y, w, h = t[:4]
            boxa = [x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5]
            labels.append(cls)
            boxes.append(boxa)
            rm_ls = []
            for i, grid in enumerate(grids):
                x, y, w, h = grid[:4]
                boxb = x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5
                iou = calc_iou(boxa, boxb)
                if iou > iou_threshold:
                    rm_ls.append(i)
                print(boxa, boxb, iou)
            for i in reversed(rm_ls):
                grids.pop(i)
    boxes = scale_coords(im.shape[2:], np.array(boxes), src_shape).round()
    return labels, boxes


class Predictor:
    def __init__(self,
                 path,
                 device,
                 imgsz,
                 conf_thres=0.1,
                 iou_thres=0.5,
                 classes=None,
                 max_det=50,
                 stride=None,
                 half=True, ratio=0.99,
                 agnostic_nms=False):
        self.half = half
        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        model = DetectMultiBackend(path, device=self.device, inplace=True, fuse=True)
        self.model = model
        self.stride, self.names, self.pt, self.jit, self.onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        self.stride = stride or self.stride
        print("stride", self.stride)
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        print(self.names)
        self.img_size = check_img_size(imgsz, s=self.stride)  # check image size
        if self.pt:
            model.model.half() if half else model.model.float()
            if half:
                dtype = torch.float16
            else:
                dtype = torch.float32
            model(torch.zeros(1, 3, *self.img_size).to(device).type(dtype))  # warmup
            # model = model.model if hasattr(model, "model") else model
        self.name_map = dict(zip(self.names, range(len(self.names))))
        print(self.name_map)
        self.classes = list(map(lambda name: self.name_map[name], classes))
        self.indexes = set(self.classes)
        self.ratio = ratio

    @torch.no_grad()
    def predict(self, im):
        src_shape = im.shape
        model = self.model
        # Half
        half = self.half
        device = self.device

        img = preprocess(im, *self.img_size)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im)
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                  max_det=self.max_det)[0]
        if not len(det):
            return [], [], []
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], src_shape).round()
        # xyxy, conf, cls
        res = list(map(lambda x: self.names[int(x)], det[:, -1])), \
              det[:, :4].cpu().numpy().astype(int), \
              det[:, 4].cpu().numpy().astype("float32")
        return res


def test_video(video_path, det: Predictor):
    import os
    if not os.path.isdir(video_path):
        ls = [video_path]
    else:
        ls = (os.path.join(p, name) for p, _, names in os.walk(video_path) for name in names if
              os.path.splitext(name)[1].lower() in (".mp4", ".avi", ".3gp", ".webm"))
    for video_path in ls:
        reader = cv2.VideoCapture()
        reader.open(video_path)
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            labels, boxes, scores = det.predict(frame)
            plot(frame, labels, boxes, scores, frame.shape)
            cv2.imshow("res", frame)
            if cv2.waitKey(1) == 27:
                break


def test_images(det: Predictor, img_dir):
    from toolset.image_tools import imread
    for name in os.listdir(img_dir):
        filepath = os.path.join(img_dir, name)
        _, ext = os.path.splitext(name)
        if ext not in [".jpg", ".png"]:
            continue
        img = imread(filepath)
        labels, boxes, scores = det.predict(img)
        plot(img, labels, boxes, scores, img.shape)
        frame, _ = auto_resize(img, 1280, 600)
        cv2.imshow("res", frame)
        if cv2.waitKey(0) == 27:
            break


if __name__ == '__main__':
    predictor = Predictor(
        r"weights/qzdl01.pt",
        "cuda:0", [416, 416], conf_thres=0.3, half=False, ratio=0.999,
        classes=["zombie"])
    # test_video(
    #     r"D:\Videos\youtube\Used To Be - SAD Piano-Orchestral Song Instrumental.webm",
    #     predictor)
    test_images(predictor, r"D:\Pictures\dataset")
