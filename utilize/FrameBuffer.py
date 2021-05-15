import threading
import time
import collections
import cv2
import win32gui, win32ui, win32con, win32api
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 获取视频帧
class FrameBuffer(threading.Thread):
    def __init__(self, threadID, name, width, height, maxlen=5):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.buffer = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

        self.WIDTH = width
        self.HEIGHT = height
        self._stop_event = threading.Event()

        self.hwnd = win32gui.FindWindow(None, 'Hollow Knight')

        rect = win32gui.GetWindowRect(self.hwnd)
        left = rect[0]
        top = rect[1]
        width = rect[2] - left
        height = rect[3] - top

        self.station_size = (left, top, width + left - 1, height + top - 1)

        self.left, self.top, x2, y2 = self.station_size
        self.width = x2 - self.left + 1
        self.height = y2 - self.top + 1


        self.hwindc = win32gui.GetWindowDC(self.hwnd)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)

        # 显示 图片
        # 将截图保存到saveBitMap中
    #    self.memdc.SelectObject(self.bmp)
        # 保存bitmap到内存设备描述表
    #    self.memdc.BitBlt((0, 0), (width, height), self.srcdc, (0, 0), win32con.SRCCOPY)

    #    bmpinfo = self.bmp.GetInfo()
    #    bmpstr = self.bmp.GetBitmapBits(True)
    #    im_PIL = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)#    im_PIL.save("im_PIL.png")  # 保存
    #    im_PIL.show()  # 显示

    def run(self):
        while not self.stopped():
            self.get_frame()
            time.sleep(0.05)
        self.srcdc.DeleteDC()
        self.memdc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwindc)
        win32gui.DeleteObject(self.bmp.GetHandle())

    def get_frame(self):
        self.lock.acquire(blocking=True)
        station = cv2.resize(cv2.cvtColor(self.grab_screen(), cv2.COLOR_RGBA2RGB), (self.WIDTH, self.HEIGHT))
        self.buffer.append(station)
        self.lock.release()

    def get_buffer(self):
        stations = []
        self.lock.acquire(blocking=True)
        for f in self.buffer:
            #plt.imshow(f)
            #plt.show()
            stations.append(f)
        self.lock.release()
        stations = np.transpose(stations, (3, 0, 1, 2))
        return torch.Tensor(stations)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def grab_screen(self):
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (0, 0), win32con.SRCCOPY)

        signedIntsArray = self.bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.height, self.width, 4)

        return img