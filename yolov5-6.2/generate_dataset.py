import os
import numpy as np
import torch
import cv2
from models.experimental import attempt_load
from models.yolo import Detect
from convert import walk_dir
from pathlib import Path
from utils.dataloaders import letterbox
import torch.nn.functional as F


class GenVideoData:
    def __init__(self, weights_file, img_size, device="cuda:0", fp16=False, fuse=True):
        self.img_size = img_size
        self.device = device
        self.half = fp16
        model = attempt_load(weights_file, device=device, inplace=True, fuse=fuse)
        self.stride = max(int(model.stride.max()), 32)  # model stride
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        for m in model.modules():
            if not isinstance(m, Detect):
                continue
            print(m.video)
            m.video = True
            break

    def run(self, video_dir, save_dir):
        for name, file in walk_dir(video_dir):
            file = Path(file)
            if not file.with_suffix(".mp4"):
                continue
            cap = cv2.VideoCapture(file.as_posix())
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                img = letterbox(frame, new_shape=(352, 640), stride=self.stride, auto=False)[0]
                print(img.shape)
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                im = torch.from_numpy(img).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                pred = self.model(im)  # [B, C, H, W]
                max_w, max_h = 0, 0
                for i in range(len(pred)):
                    b, c, h, w = pred[i].shape
                    if h > max_h:
                        max_h = h
                    if w > max_w:
                        max_w = w
                xs = []
                for i in range(len(pred)):
                    print(pred[i].shape)
                    xs.append(F.interpolate(pred[i], [max_h, max_w], mode="nearest"))
                x = torch.cat(xs, dim=1)
                print(x.shape)
                # break


if __name__ == '__main__':
    video_dir = r"D:\WorkDir\项目\衢州电力1期\dataset\show_videos\all.mp4"
    save_dir = r""
    g = GenVideoData("weights/yolov5s.pt", 640)
    g.run(video_dir, save_dir)
