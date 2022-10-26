import os
import pathlib
import random
import threading
import time
from copy import copy
from typing import List
import cv2
import torch
import numpy as np
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (check_img_size, scale_coords, non_max_suppression)
from utils.torch_utils import select_device
from base_model import BaseModel


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


class Predictor(BaseModel):
    def __init__(self,
                 path,
                 device,
                 imgsz,
                 conf_thres=0.6,
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
        model = DetectMultiBackend(path, device=self.device, fuse=True)
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
            for i in range(10):
                model(torch.zeros(1, 3, *self.img_size).to(device).type(dtype))  # warmup
            # model = model.model if hasattr(model, "model") else model
        self.name_map = {v: k for k, v in self.names.items()}
        # self.name_map = self.names

        print(self.name_map)
        if not classes:
            classes = self.names.values()
        self.classes = list(map(lambda name: self.name_map[name], classes))
        self.indexes = set(self.classes)
        self.ratio = ratio

    def get_labels(self):
        return list(self.names.values())

    @torch.no_grad()
    def predict(self, im):
        self.model.model.eval()
        # Load model
        # self.model.model.train()
        src_shape = im.shape
        model = self.model
        # Half
        half = self.half  # half precision only supported by PyTorch on CUDA
        device = self.device
        img = letterbox(im, self.img_size, stride=self.stride, auto=True)[0]
        # img = preprocess(im, *self.img_size)
        # print("img", img.shape)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        # t0 = time.time()
        # print(im.shape)
        # print(im.shape)
        pred = model(im)
        # res = post_process(im, src_shape, pred[1], self.iou_thres)
        # print(time.time() - t0)
        # return res
        # # NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                  max_det=self.max_det)[0]
        # Process predictions
        # print(det[:, :4].shape)
        if not len(det):
            return [], np.array([]), np.array([])
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], src_shape).round()
        # xyxy, conf, cls
        res = list(map(lambda x: self.names[int(x)], det[:, -1])), \
              det[:, :4].cpu().numpy().astype(int), \
              det[:, 4].cpu().numpy().astype("float32")
        # print(time.time() - t0)
        return res

    def predict_original(self, img, rect_w, rect_h, delta_ratio):
        h, w, d = img.shape
        delta_ratio = 1 - delta_ratio
        step_h = int(rect_h * delta_ratio)
        step_w = int(rect_w * delta_ratio)
        bs = []
        ls = []
        ss = []
        for xi in range(0, w, step_w):
            for yi in range(0, h, step_h):
                if xi + rect_w > w:
                    xi = w - rect_w
                    xi = max(xi, 0)
                if yi + rect_h > h:
                    yi = h - rect_h
                    yi = max(yi, 0)
                rect = img[yi:yi + rect_h, xi: xi + rect_w]
                labels, boxes, scores = self.predict(rect)
                boxes = np.array(boxes).reshape((-1, 4))
                boxes += [xi, yi, xi, yi]
                ls.extend(labels)
                ss.extend(scores)
                bs.append(boxes)
        bs = np.concatenate(bs, 0)
        return ls, bs, ss

    def get_heat_map(self, im, size=None):
        # self.model.train()
        model: torch.nn.Module = self.model.model
        # model.eval()
        device = self.device
        if size is None:
            img = letterbox(im, self.img_size, stride=self.stride, auto=True)[0]
        else:
            img = preprocess(im, *size)
        src_img = img
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im: torch.Tensor
        x: torch.Tensor = torch.nn.Parameter(im)
        x.requires_grad_()
        model.train()
        model.zero_grad()
        pred = model(x)
        # print(len(pred))
        # y += pred[2].sum()
        t = []
        for p in pred:
            p: torch.Tensor
            bs, n, h, w, c = p.shape
            p = p.view(bs * n * h * w, c)
            t.append(p)
        pa = torch.concat(t, 0)
        pa = pa.sigmoid()
        y_all = torch.tensor(0., device=self.device)
        items = pa[pa[:, 4] >= self.conf_thres]
        is_empty = True
        for item in items:
            cls = int(torch.argmax(item[5:]))
            if cls not in self.indexes:
                continue
            idx = 5 + cls
            prob = item[idx] * item[4]
            if float(prob) < self.conf_thres:
                print(idx, float(item[idx]), float(item[4]), float(prob), )
                continue
            y_all += prob
            y_all += item[:4].sum()
            is_empty = False
        if is_empty:
            print("empty---------")
            return
        else:
            y_all.backward()
        grad = x.grad[0]
        n = grad.shape[0]
        grad = grad.sum(dim=0, keepdim=True) / n

        grad = grad.abs()
        grad = grad - grad.min()

        # grad /= grad.mean()

        # grad = grad / (torch.max(grad) * 0.8)
        k = torch.flatten(grad).unique()
        top_k_idx = k.argsort(descending=False)
        grad = grad / (k[top_k_idx[int(len(top_k_idx) * self.ratio)]])
        # grad = grad.sigmoid()

        grad = grad * 255

        # k = torch.flatten(grad)
        # top_k_idx = k.argsort(descending=False)
        # thres = k[top_k_idx[int(len(top_k_idx) * 0.9)]]
        # # thres = 10
        # c1 = grad < thres
        # c2 = grad >= thres
        # grad[c1] = torch.from_numpy(gray[None][c1.cpu().numpy()]).float().to(self.device)
        # grad[c2] *= 1

        heatmap_img = grad.cpu().numpy().clip(max=255).astype("uint8")[0]
        show = np.stack([gray, gray, heatmap_img], -1)

        # cv2.imshow("grad", show)
        # cv2.waitKey(1)
        return show


def preprocess(img, dw, dh):
    h, w, d = img.shape
    ratio = min(dw / w, dh / h, 1)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h))
    res = cv2.copyMakeBorder(img_resized, 0, dh - new_h, 0, dw - new_w, cv2.BORDER_CONSTANT, value=114)
    # print(str(res.shape))
    # cv2.imshow("====", res)
    # cv2.waitKey(1)
    return res


def test_video_qzdl(video_path, det: Predictor):
    import os
    if not os.path.isdir(video_path):
        ls = [video_path]
    else:
        ls = (os.path.join(p, name) for p, _, names in os.walk(video_path) for name in names if
              os.path.splitext(name)[1].lower() in (".mp4", ".avi", ".3gp"))
    det_ren = Predictor(r"weights/yolov5s.pt", "cuda:0", 640, conf_thres=0.3, half=False)
    for video_path in ls:
        reader = cv2.VideoCapture()
        reader.open(video_path)
        while True:
            ret, frame = reader.read()
            if not ret:
                break

            src_frame = frame.copy()
            labels, boxes, scores = det_ren.predict(frame)
            plot(frame, labels, boxes, scores, src_frame.shape)
            frame, _ = auto_resize(frame, 1280, 1280)
            cv2.imshow("res", frame)
            if cv2.waitKey(1) == 27:
                break
            for label, box in zip(labels, boxes):
                if label != 'person':
                    continue
                x1, y1, x2, y2 = box
                rect = np.ascontiguousarray(src_frame[y1: y2, x1: x2])
                # preprocess(rect, 320, 608)
                # labels, boxes = det.predict(frame)
                # plot(frame, labels, boxes)
                det.get_heat_map(rect, (320, 608))
                show, _ = auto_resize(src_frame, 640, 640)
                cv2.imshow("res", show)
                if cv2.waitKey(0) == 27:
                    break
                # break


def test_video_vison(video_path, det: Predictor):
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
            heat_map = det.get_heat_map(frame)
            if heat_map is not None:
                t = 0
            else:
                heat_map = letterbox(frame, det.img_size, stride=det.stride, auto=True)[0]
                t = 1
            labels, boxes, scores = det.predict(frame)
            plot(heat_map, labels, boxes, scores, frame.shape)
            cv2.imshow("res", heat_map)
            if cv2.waitKey(t) == 27:
                break


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
            # labels, boxes, scores = det.predict_original(frame, 640, 352, 0.5)
            labels, boxes, scores = det.predict(frame)

            # labels1, boxes1, scores1 = predictor.predict(frame)
            # labels += labels1
            # boxes = np.concatenate([boxes.reshape((-1, 4)), boxes1.reshape((-1, 4))], 0)
            # scores = np.concatenate([scores, scores1], 0)

            plot(frame, labels, boxes, scores, frame.shape)
            frame, _ = auto_resize(frame, 1280, 600)
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
        labels, boxes, scores = det.predict_original(img, 640, 640, 0.5)
        plot(img, labels, boxes, scores, img.shape)
        # det.get_heat_map(img)
        frame, _ = auto_resize(img, 1280, 1280)
        cv2.imshow("res", frame)
        if cv2.waitKey(0) == 27:
            break


from base_obj import BaseObj


def calc_a_v1_v2(ys, ts):
    y0 = ys[0]
    y1 = ys[1]
    y2 = ys[2]
    t0 = ts[0]
    t1 = ts[1]
    t2 = ts[2]
    A1 = y1 - y0
    B1 = t1 - t0
    C1 = 0.5 * ((t1 - t0) ** 2)
    A2 = y2 - y1
    B2 = t2 - t1
    C2 = 0.5 * ((t2 - t1) ** 2)
    a = (A2 - (B2 * A1 / B1)) / (B1 * B2 - (C1 * B2 / B1) + C2)
    v0 = (A1 - a * C1) / B1
    v1 = a * B1 + v0
    return a, v0, v1


def calc_av(x1, x2, t1, t2):
    dx = x2 - x1
    dt = t2 - t1
    v = dx / dt
    return v


class FrameNo:
    def __init__(self, val=0):
        self.val = val

    def __int__(self):
        return self.val

    def __add__(self, other):
        return FrameNo(self.val + other)


class VideoCapture:
    def __init__(self, path):
        self.reader = cv2.VideoCapture(path)
        self.frame_no = 0
        self.ok = True
        self.frame = None
        self._exit = False
        threading.Thread(target=self.__loop).start()

    def __loop(self):
        while True:
            t0 = time.time()
            ok, frame = self.reader.read()
            time.sleep(max(0., 0.02 - (time.time() - t0)))
            self.ok, self.frame = ok, frame
            if not ok:
                break
            self.frame_no += 1
        self._exit = True

    def read(self, frame_no: FrameNo):
        while self.frame_no == frame_no.val and (not self._exit):
            time.sleep(0.)
        frame_no.val = self.frame_no
        return self.ok, self.frame


class Matcher:
    def __init__(self, obj: BaseObj = None, objs=None):
        if objs is not None:
            self.objs = objs
        else:
            self.objs = [obj] if obj is not None else []
        self.__success = False
        self.max_count = 2
        self.max_age = 2
        self.lost_count = 0
        self.__count = -1
        self.last_a = None
        self.id = id(self)
        self.total_h = 0

    @property
    def ffn(self):
        return self.objs[0].frame_no

    @property
    def lfn(self):
        return self.objs[-1].frame_no

    def update(self, obj: BaseObj, frame: np.ndarray):
        self.objs.append(obj)
        self.total_h += obj.height
        frame = frame.copy()
        if len(self.objs) < 3:
            return
        try:
            s_h, s_w, _ = frame.shape
            ts = [i.time for i in self.objs][-3:]
            ys = [i.y_center for i in self.objs][-3:]
            a, v1, v2 = calc_a_v1_v2(ys, ts)
            av1 = calc_av(*ys[:2], *ts[:2])
            av2 = calc_av(*ys[1:3], *ts[1:3])
            success = a > 0
            success &= v2 >= v1 >= 0
            # if av2 > av1:
            #     success &= (av2 / (av1 + 0.00001)) < 1.5
            # else:
            #     success &= (av1 / (av2 + 0.00001)) < 1.5
            if self.last_a is not None:
                pass
                # success &= abs(self.last_a - a) < abs((self.last_a + a) / 2 * 0.5)

                # if self.last_a > a:
                #     success &= (self.last_a / (a + 0.000001)) < 2
                # else:
                #     success &= (a / (self.last_a + 0.000001)) < 2

                # success &= self.objs[-1].y2 - self.objs[0].y1 >= self.total_h * 0.5
            # success &= self.objs[-1].y2 - self.objs[0].y1 > s_h * 0.05
            last_delta = 0
            for i, o in enumerate(self.objs[1:]):
                delta = o.cy - self.objs[i].cy
                if not (0 <= last_delta < delta):
                    success &= False
                    break
                last_delta = delta
            # print(self.objs[-1].y2 - self.objs[0].y1 > s_h * 0.2)
            if success:
                self.__count += 1
                print(f"success: {success}, {self.__count}", "len", len(self.objs), "a:", a, "v1:", v1, "v2:",
                      v2, "av1:", av1, "av2:", av2, abs(av2 - av1), "id:", {self.id}, "a:", a, self.last_a)
            self.last_a = a
            if self.__count >= self.max_count:
                self.__success = True
            #     print("-" * 20)
            #     for o in self.objs:
            #         print(int(o.y_center), end=" ")
            #     print()
            #     print("-" * 20)
            #     if v1 * v2 < 0:
            #         for y, t in zip(ys, ts):
            #             print(y, t)
            # # for o in self.objs:
            #     cv2.circle(frame, o.center_int, 2, (0, 0, 255), 2)
            # cv2.imshow(f"ddd", auto_resize(frame, 1280, 600)[0])
            # cv2.waitKey(0)
            if not success:
                # if self.__count >= 0:
                #     print(f"failed: {success}, {self.__count}", "len", len(self.objs), "a:", a, "v1:", v1, "v2:",
                #           v2, "av1:", av1, "av2:", av2, abs(av2 - av1), "id:", {self.id}, "a:", a, self.last_a)
                if self.lost_count > self.max_age:
                    return False
                self.lost_count += 1
            else:
                self.lost_count = 0
        except:
            return False

    @property
    def is_success(self):
        return self.__success

    def copy(self):
        new = copy(self)
        new.objs = self.objs.copy()
        new.id = id(new)
        return new


def falling_test(predictor: Predictor, video_path):
    import os
    if not os.path.isdir(video_path):
        ls = [video_path]
    else:
        ls = (os.path.join(p, name) for p, _, names in os.walk(video_path) for name in names if
              os.path.splitext(name)[1].lower() in (".mp4", ".avi", ".3gp", ".webm"))
    save_dir = "result"
    for video_path in ls:
        reader = VideoCapture(video_path)
        # reader.open(video_path)
        matchers = []
        frame_no = FrameNo()
        filename = pathlib.Path(video_path).name
        while True:
            ret, frame = reader.read(frame_no)
            if not ret:
                break
            frame_no.val += 1
            labels, boxes, scores = predictor.predict(frame)
            objs = [BaseObj(label=l, bbox=b) for l, b in zip(labels, boxes)]
            success = []
            new_matchers = []
            count = 0
            for m in matchers:
                if count > 200:
                    break
                for obj in objs:
                    t = m.copy()
                    if t.update(obj, frame) is None:
                        if t.is_success:
                            success.append(t)
                        else:
                            new_matchers.append(t)
                            count += 1
            matchers = new_matchers
            for obj in objs:
                matchers.append(Matcher(obj))
            for m in success:
                for o in m.objs:
                    cx = int(o.cx)
                    cy = int(o.cy)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), 3)
                cv2.imwrite(os.path.join(save_dir, f"{filename}_{frame_no.val}.png"), auto_resize(frame, 1280, 600)[0])
                # cv2.waitKey(1)
            plot(frame, labels, boxes, scores, frame.shape)
            frame, _ = auto_resize(frame, 1280, 600)
            cv2.imshow(filename, frame)
            if cv2.waitKey(1) == 27:
                break
            elif cv2.waitKey(1) == 32:
                cv2.waitKey(0)
        cv2.destroyWindow(filename)
        if cv2.waitKey(1) == 27:
            break
    cv2.waitKey(0)


if __name__ == '__main__':
    predictor = Predictor(
        r"weights/best.pt",
        "cuda:0", 640, iou_thres=0.3, conf_thres=0.5, half=False, ratio=0.99,
        classes=None)
    # falling_test(predictor, r"D:\dataset\project\衢州电力\高空抛物_安全帽")
    test_video(r"D:\WorkDir\dataset\src_videos", predictor)
    # test_images(predictor, r"D:\dataset\project\工地项目\gongdi1")
