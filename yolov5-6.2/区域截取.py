import json

import cv2
import os
import math
import numpy as np
import yaml
from pathlib import Path
from toolset.image_tools import imread, imwrite
from voc_xml_parse import VocXML, Objects, Object, Bndbox, get_pair_sample_from_dir
from windows import get_screen_size, get_window_rect, find_window


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


class MyView:
    def __init__(self, img, objs: Objects, target_w,
                 target_h, max_w, max_h, win_name,
                 cache_dir, filename, ):
        h, w = img.shape[:2]
        self.window_name = win_name
        self.cache_dir = Path(cache_dir).as_posix()
        base_name, _ = os.path.splitext(filename)
        self.cache_file = os.path.join(cache_dir, base_name) + ".json"
        self.objs = objs
        self.scale = min(max_w / w, max_h / h, 1)
        self.new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * self.scale))
        self.src_img = img.copy()
        self.canvas = cv2.resize(img, self.new_size)
        self.canvas_src = self.canvas.copy()
        self.rect_ls = self.load_cache()
        self.rect_color = (0, 0, 255)
        self.rect_thickness = 1
        self.rect = [0, 0, 0, 0]
        self.color = (0, 255, 0)
        self.thickness = 2
        self.rect_width = target_w
        self.rect_height = target_h
        self.delta = int(target_h * 0.1)

    def load_cache(self):
        if Path(self.cache_file).exists():
            rect_ls = json.loads(open(self.cache_file, 'rb').read())
        else:
            rect_ls = []
        return rect_ls

    def save_cache(self):
        open(self.cache_file, "wb").write(json.dumps(self.rect_ls).encode("utf8"))

    def redraw(self):
        self.canvas = self.canvas_src.copy()
        scale = self.scale
        for x1, y1, x2, y2 in self.rect_ls:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), self.rect_color, thickness=self.rect_thickness)
        rect_x1, rect_y1, rect_x2, rect_y2 = self.rect
        rect_x1, rect_y1, rect_x2, rect_y2 = map(lambda xx: int(xx / scale),
                                                 (rect_x1, rect_y1, rect_x2, rect_y2))
        for obj in self.objs:
            x1, y1, x2, y2 = obj.bndbox.bbox
            cv2.rectangle(self.canvas, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)),
                          (255, 0, 0))
            if not rect_in_rect([rect_x1, rect_y1, rect_x2, rect_y2], [x1, y1, x2, y2], 0.13):
                # x1, y1, x2, y2 = map(lambda xx: int(xx * scale), (x1, y1, x2, y2))
                # cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (255, 0, 0))
                continue
            x1 = max(x1, rect_x1)
            y1 = max(y1, rect_y1)
            x2 = min(x2, rect_x2)
            y2 = min(y2, rect_y2)
            x1, y1, x2, y2 = map(lambda xx: int(xx * scale), (x1, y1, x2, y2))
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (50, 50, 255), 2)
        cv2.rectangle(self.canvas, tuple(self.rect[:2]), tuple(self.rect[2:]), self.color, self.thickness)
        size = math.ceil((self.rect[2] - self.rect[0]) / scale), \
               math.ceil((self.rect[3] - self.rect[1]) / scale)
        cx = (self.rect[2] + self.rect[0]) // 2
        cy = (self.rect[3] + self.rect[1]) // 2
        size = f"[{size[0]} {size[1]}]"
        tl = round(0.002 * (self.canvas.shape[0] + self.canvas.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(self.canvas, size, (cx - 83, cy - 10), 0, tl / 3, [0, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imshow(self.window_name, self.canvas)

    def save_objs(self, save_fmt_name: str):
        for i, (rect_x1, rect_y1, rect_x2, rect_y2) in enumerate(self.rect_ls):
            rect_x1, rect_y1, rect_x2, rect_y2 = map(lambda xx: math.ceil(xx / self.scale),
                                                     (rect_x1, rect_y1, rect_x2, rect_y2))
            width = rect_x2 - rect_x1
            height = rect_y2 - rect_y1
            rect = self.src_img[rect_y1:rect_y2, rect_x1: rect_x2]
            img_path = save_fmt_name.format(ext="jpg", id=i)
            label_path = save_fmt_name.format(ext='xml', id=i)
            imwrite(img_path, rect)
            labels = VocXML.create(label_path, width, height)
            for obj in self.objs:
                if not rect_in_rect([rect_x1, rect_y1, rect_x2, rect_y2], obj.bndbox.bbox, 0.13):
                    continue
                x1, y1, x2, y2 = obj.bndbox.bbox
                x1 = max(x1 - rect_x1, 0)
                y1 = max(y1 - rect_y1, 0)
                x2 = min(x2 - rect_x1, width)
                y2 = min(y2 - rect_y1, height)
                labels.annotation.objects.append(Object.create(obj.name, Bndbox.create(x1, y1, x2, y2)))
            labels.save(label_path)

    def update_rect(self, x, y):
        w = self.rect_width * self.scale
        h = self.rect_height * self.scale
        w2 = w // 2
        h2 = h // 2
        sh, sw = self.canvas.shape[:2]
        self.rect = [*map(int, [max(x - w2, 0), max(y - h2, 0), min(x + w2, sw), min(y + h2, sh)])]

    def on_mouse(self, event, x, y, flag, param):
        self.update_rect(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
            self.rect_ls.append(self.rect)
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键按下
            if len(self.rect_ls):
                self.rect_ls.pop()
        elif event == cv2.EVENT_MOUSEWHEEL:  # 如果滚动更新橡皮擦矩形
            if flag > 0:
                self.rect_width += self.delta
                self.rect_height += self.delta
            else:
                if self.rect_width >= 4 and self.rect_height >= 4:
                    self.rect_width -= self.delta
                    self.rect_height -= self.delta
                else:
                    self.rect_width = 4
                    self.rect_height = 4
        self.update_rect(x, y)
        self.redraw()

    def run(self, img_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_dir = Path(save_dir).as_posix()
        base_name, _ = os.path.splitext(img_name)
        base_name = f"{base_name}{{id}}.{{ext}}"
        save_fmt_name = os.path.join(save_dir, base_name)
        cv2.imshow(self.window_name, self.canvas)
        ###############################################
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        rect = cv2.getWindowImageRect(self.window_name)
        screen_size = get_screen_size()
        left, top, right, bottom = get_window_rect(find_window(title_name=self.window_name))
        W = right - left
        H = bottom - top
        cv2.moveWindow(self.window_name, int((screen_size[0] - W) // 2), int((screen_size[1] - H) // 2) - 40)
        ###############################################
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        ret_code = 0
        while True:
            #  ord("a"), ord("d")
            key_code = cv2.waitKey()
            if key_code == 27:
                cv2.destroyAllWindows()
                exit()
            elif key_code == 13:  # enter
                continue
            elif key_code == 32:
                self.save_objs(save_fmt_name)
                break
            elif key_code == ord("a"):
                self.save_objs(save_fmt_name)
                ret_code = -1
                break
            elif key_code == ord("d"):
                self.save_objs(save_fmt_name)
                ret_code = 1
                break
        cv2.destroyWindow(self.window_name)
        self.save_cache()
        return ret_code


if __name__ == '__main__':
    # 鼠标左键 选定区域，空格 确认并截取区域图片和标签，
    self_file = Path(__file__)
    print(self_file.with_suffix('.yaml'))
    cfg = yaml.load(open(f"{self_file.with_suffix('.yaml')}"), yaml.FullLoader)
    img_dir = cfg["img_dir"]
    label_dir = cfg["label_dir"]
    cache_dir = cfg["cache_dir"]
    save_dir = cfg["save_dir"]
    t_w = 640
    t_h = 640
    show_max_w = 1300
    show_max_h = 1300
    data = list(get_pair_sample_from_dir(img_dir, label_dir))
    os.makedirs(cache_dir, exist_ok=True)
    i = 0
    while True:
        name, img_path, label_path = data[i]
        img = imread(img_path)
        if img is None:
            continue
        if Path(label_path).exists():
            objs = VocXML(label_path).annotation.objects
        else:
            objs = []
        code = MyView(img, objs, t_w, t_h, show_max_w, show_max_h,
                      f"{name} {i}/{len(data)}", cache_dir, name).run(name, save_dir)
        if code < 0:
            i -= 1
        elif code > 0:
            i += 1
        i = min(len(data), i)
        i = max(0, i)
