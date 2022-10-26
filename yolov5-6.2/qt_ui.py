import sys
from typing import List
import random
from PyQt5 import QtCore
from numpy.typing import NDArray
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF
from PyQt5.QtGui import QPaintEvent, QPainter, QPen, QColor, QPolygon, QLinearGradient, QFont, QImage, QPixmap, \
    QMouseEvent, QBrush, QPainterPath
from PyQt5.QtWidgets import QWidget, QLabel, QGraphicsView, QPushButton, QApplication, QHBoxLayout, QVBoxLayout, \
    QHeaderView


def rand_color():
    return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 100)


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


class SlideBar:
    def __init__(self):
        self.is_active = False


class Position:
    def __init__(self, start, width, height: float, color, r=0.15):
        self.default_start = start
        self.start = start
        self.default_color = color
        self.color = color
        self.default_width = width
        self.width = width
        self.default_height = height
        self.height = height
        self.default_r = r
        self.r = r
        self.is_press = False

    def reset(self):
        self.start = self.default_start
        self.color = self.default_color
        self.width = self.default_width
        self.height = self.default_height
        self.r = self.default_r
        self.is_press = False


class EditProgressBar(QWidget):
    def __init__(self, p):
        super().__init__(p)
        self.progress = 0
        self.positions: List[Position] = []
        self._height_ratio = 0.6
        self.is_selected = False
        self.setMouseTracking(True)

    def add_position(self, start, width, height, color):
        self.positions.append(Position(start, width, height, color))

    def paintEvent(self, a0: QPaintEvent) -> None:
        self.setFixedHeight(int(self.width() * 0.05))
        qp = QPainter()
        qp.begin(self)
        width = self.width()
        height = self.height()
        top = int(height * (1 - self._height_ratio) // 2)
        qp.fillRect(2, top, width - 2, height - top, QColor(0, 100, 200, 50))
        for pos in self.positions:
            t_height = pos.height * height
            top = int(max((height - t_height) // 2, 0))
            # qp.fillRect(pos.start, top, pos.width, height - top, pos.color)
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.WindingFill)
            r = t_height * pos.r
            path.addRoundedRect(pos.start, top, pos.width, height - top, r, r)
            path.addRect(pos.start, top + t_height / 2, pos.width, t_height / 2)
            qp.fillPath(path, QBrush(pos.color))
        qp.end()  # 结束整个画图

    def mouseMoveEvent(self, a0: QMouseEvent) -> None:
        x, y = a0.x(), a0.y()
        is_selected = False
        for pos in self.positions:
            if pos.start <= x <= pos.start + pos.width:
                if not is_selected:
                    pos.height = 0.8
                    pos.color = QColor(0, 121, 255, 180)
                    pos.r = 0
                    is_selected = True
                    self.is_selected = True
            else:
                if pos.is_press:
                    continue
                pos.reset()
        self.repaint()

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        x, y = a0.x(), a0.y()
        for pos in self.positions:
            if pos.start <= x <= pos.start + pos.width:
                pos.is_press = not pos.is_press
            else:
                pos.reset()
        self.repaint()

    def underMouse(self) -> bool:
        print("underMouse")
        return True

    # def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
    #     print("mouseReleaseEvent")


class ImgShow(QWidget):
    def __init__(self, p):
        super().__init__(p)
        self.src_img = None
        self.img: QImage = None
        self.show_rect = None

    def set_img(self, img: NDArray):
        self.src_img = img.copy()
        screen_w = self.width()
        screen_h = self.height()
        img = auto_resize(img, screen_w, screen_h)[0]
        h, w = img.shape[:2]
        # print(screen_w, screen_h)
        img: NDArray
        # print(img.shape)
        self.img = QImage(img, w, h, w * 3, QImage.Format_BGR888)
        left = max(screen_w - w, 0) // 2
        top = max(screen_h - h, 0) // 2
        self.show_rect = QRect(left, top, w, h)  # 图片放置坐标，长，高缩放的倍数

    def paintEvent(self, a0: QPaintEvent) -> None:
        if self.img is None:
            return
        qp = QPainter()
        qp.begin(self)
        qp.drawImage(self.show_rect, self.img)  # 画出图片
        qp.end()  # 结束整个画图


class VideoView(QWidget):
    def __init__(self, p):
        super().__init__(p)
        self.lay = QVBoxLayout(self)
        self.progress = EditProgressBar(self)
        self.video = ImgShow(self)
        self.video.setFixedSize(640, 360)
        self.lay.addWidget(self.video)
        self.lay.addWidget(self.progress)

    def set_img(self, img):
        self.video.set_img(img)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(900, 700)
        self.setWindowTitle("视频截取工具")
        self.lay = QVBoxLayout(self)
        self.video = VideoView(self)
        self.video.progress.add_position(10, 20, 0.6, rand_color())
        self.video.progress.add_position(50, 200, 0.6, rand_color())
        self.video.progress.add_position(200, 220, 0.6, rand_color())
        self.lay.addWidget(self.video)
        img = cv2.imread("images/diaowu.png")
        self.video.set_img(img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
