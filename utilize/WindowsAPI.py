import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from PIL import Image
# get hollow knight hwnd
hwnd = win32gui.FindWindow(None, 'Hollow Knight')


# get windows image of hollow knight
def grab_screen(region=None):
    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        rect = win32gui.GetWindowRect(hwnd)
        left = rect[0]
        top = rect[1]
        width = rect[2] - left
        height = rect[3] - top

    #print(left, top, width, height)
    # 返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    hwindc = win32gui.GetWindowDC(hwnd)
    #color = win32gui.GetPixel(hwindc, 100 , 200)
    # 创建设备描述表
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    # 创建内存设备描述表
    memdc = srcdc.CreateCompatibleDC()
    # 创建位图对象准备保存图片
    bmp = win32ui.CreateBitmap()
    # 为bitmap开辟存储空间
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    # 将截图保存到saveBitMap中
    memdc.SelectObject(bmp)
    # 保存bitmap到内存设备描述表
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)

    # 保存并输出截图
    #bmpinfo = bmp.GetInfo()
    #bmpstr = bmp.GetBitmapBits(True)
    #im_PIL = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)#    im_PIL.save("im_PIL.png")  # 保存
    #im_PIL.show()  # 显示

    img = np.fromstring(signedIntsArray, dtype='uint8')

    # RGBA 的格式，A指透明度
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


# win32 presskey and releasekey, but it has lag, what we need is in SendKey.py
def PressKey(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)


def ReleaseKey(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, win32con.KEYEVENTF_KEYUP, 0)


# check which key is pressed
def key_check():
    operations = []
    if win32api.GetAsyncKeyState(0x41):
        operations.append("A")
    if win32api.GetAsyncKeyState(0x43):
        operations.append("C")
    if win32api.GetAsyncKeyState(0x58):
        operations.append("X")
    if win32api.GetAsyncKeyState(0x5A):
        operations.append("Z")
    # 此为 F1
    if win32api.GetAsyncKeyState(0x70):
        operations.append("T")
    if win32api.GetAsyncKeyState(0x71):
        operations.append("Save")

    direction = []
    if win32api.GetAsyncKeyState(0x25):
        direction.append("Left")
    if win32api.GetAsyncKeyState(0x26):
        direction.append("Up")
    if win32api.GetAsyncKeyState(0x27):
        direction.append("Right")
    if win32api.GetAsyncKeyState(0x28):
        direction.append("Down")

    return operations, direction