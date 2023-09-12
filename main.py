import os
import sys
import random
import PyQt5
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from gui.mainWindow_ui import Ui_MainWindow
from gui.filters_ui import Ui_filterDialog
from gui.specialEffect_ui import Ui_noiseAddDialog
from gui.noiseRemove_ui import Ui_noiseRemoveDialog
from gui.lightness import Ui_lightnessDialog

from backend.histogram import *
from backend.filters import *
from backend.noise import *

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import imageio

#######################################################################


offset = 0


def mousePressEvent(event):
    global offset
    offset = event.pos()


def mouseMoveEvent(event):
    global offset
    try:
        x = event.globalX()
        y = event.globalY()
        x_w = offset.x()
        y_w = offset.y()
        if activeDialog is None:
            MainWindow.move(x - x_w, y - y_w)
        else:
            activeDialog.move(x - x_w, y - y_w)
    except:
        pass


#######################################################################

app = QApplication(sys.argv)
style = open('gui\Eclippy.qss').read()
app.setStyleSheet(style)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
MainWindow.mousePressEvent = mousePressEvent
MainWindow.mouseMoveEvent = mouseMoveEvent
MainWindow.show()

#######################################################################
"""
    Global variables
"""

original_Image = processed_Image = None
underprocessing_Image = None
undo, redo = [], []

#######################################################################
"""
    Helper Functions
"""


def apply_cmap_to_image(img, cmap):
    plt.imsave('temp.jpg', img, cmap=cmap)
    new_image = plt.imread('temp.jpg')
    os.remove('temp.jpg')
    return new_image


def getHistogramImage(img):
    img_gray = getGrayImage(img)
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.hist(img_gray.ravel(), 256, [0, 256])

    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf)
    X = cv.cvtColor(X, cv.COLOR_RGBA2BGR)
    return X


def getPixmap(img=None):
    if img is None:
        return QPixmap()

    cv.imwrite('tempo.jpg', img)
    image_Pixmap = QPixmap('tempo.jpg')
    os.remove('tempo.jpg')

    return image_Pixmap


def getGrayImage(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


#######################################################################
"""
    Main Functions
"""


def loadImage(init=False):
    global original_Image, processed_Image

    if init is False:
        img_path = QFileDialog.getOpenFileName(filter="Image (*.jpg *.png *.jpeg *.tif *.jfif)")[0]
        if img_path != '':
            reset()
            original_Image = processed_Image = cv.imread(img_path)
    else:
        original_Image = processed_Image = cv.imread('images\placeholder.jpg')
    update()


def reset():
    global original_Image, processed_Image
    processed_Image = original_Image
    undo.clear()
    redo.clear()
    ui.undo_Button.setEnabled(False)
    ui.redo_Button.setEnabled(False)
    update()


def update():
    updateImages()


#   updateHF()


def updateImages():
    ui.originalImage_label.setPixmap(getPixmap(original_Image))
    ui.processedImage_label.setPixmap(getPixmap(processed_Image))


def Coding():  # 已经完善
    def distort_image():
        global processed_Image, underprocessing_Image, original_Image
        # 创建一个与原始图像相同的副本
        img_out = processed_Image.copy()

        # 获取图像的行数、列数和通道数
        row, col, channel = processed_Image.shape
        aero = cv2.selectROI("Original", processed_Image, False, False)
        min_x, min_y, width, height = aero

        # 定义局部块的大小的一半
        half_patch = 10

        for i in range(min_y + half_patch, min_y + height - 1 - half_patch, half_patch):
            for j in range(min_x + half_patch, min_x + width - 1 - half_patch, half_patch):
                # 随机生成偏移值
                k1 = random.random() - 0.5
                k2 = random.random() - 0.5
                m = np.floor(k1 * (half_patch * 2 + 1))
                n = np.floor(k2 * (half_patch * 2 + 1))

                # 计算新的位置坐标
                h = int((i + m) % row)
                w = int((j + n) % col)

                # 将局部块的像素值替换为新位置的像素值
                img_out[i - half_patch:i + half_patch, j - half_patch:j + half_patch, :] = processed_Image[h, w, :]
                underprocessing_Image = img_out.copy()

        # 显示原始图像和扰动后的图像
        cv2.imshow('Distorted', img_out)
        applyChanges()
        # 等待按键，然后保存扭曲后的图像并关闭窗口
        cv2.waitKey(0)
        cv2.imwrite(img_out)
        cv2.destroyAllWindows()

    distort_image()  # 调用函数扭曲图像并保存为输出图像


def changeLightness(image, brightness=0.0, contrast=1.0, saturation=1.0, temperature=0.0):  # 滑块界面还没有搞好，无法实现实时更新！！！！（重要）
    if brightness == -255:  # Set image to black if brightness is at its minimum
        return np.zeros(image.shape, image.dtype)

    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    hsv_image = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)
    adjusted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] + temperature * 10, 0, 255)
    adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] - temperature * 10, 0, 255)

    return adjusted

def deletePic():  # 使用时需要先点击画笔大小，否则会报错
    drawing = False
    ix, iy = -1, -1
    radius = 5  # 默认的画笔粗细
    global processed_Image, underprocessing_Image

    def draw_mask(event, x, y, flags, param):
        global ix, iy, drawing, radius

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(newmask, (x, y), radius, 255, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(newmask, (x, y), radius, 255, -1)

    def adjust_radius(val):
        global radius
        radius = val

    img = processed_Image.copy()
    newmask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_mask)

    # 创建滚动条
    cv2.createTrackbar('Radius', 'image', 5, 50, adjust_radius)

    while (1):
        overlay = img.copy()
        red_overlay = cv2.merge([newmask, newmask * 0, newmask * 0])
        cv2.addWeighted(red_overlay, 0.5, overlay, 1 - 0.5, 0, overlay)
        cv2.imshow('image', overlay)

        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # 按下Enter键
            mask[newmask == 255] = 1
            mask[newmask == 0] = 0

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

            mask_output = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img_display = img * mask_output[:, :, np.newaxis]
            cv2.imshow('GrabCut', img_display)
            underprocessing_Image = img_display.copy()
            applyChanges()
        elif k == ord('s'):  # 按下s键
            transparent_img = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2], mask_output * 255))
            cv2.imwrite('saved_image.png', transparent_img)
        elif k == 27:  # 按下ESC键
            transparent_img = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2], mask_output * 255))
            cv2.imwrite('result.png', transparent_img)
            break

    cv2.destroyAllWindows()


def Gif():  # 需要制作ui界面（用户选择输入图像）目前只能实现原图像和用户输入一张图像功能
    global processed_Image, underprocessing_Image

    class make_gif():  # 该类的返回值均为图片序列和图片时间间隔
        def __init__(self, path, t):  # duration为gif图片间的间隔时间,t==1表示该路径为视频路径， t==0表示该路径为图片路径
            self.frames = []
            self.frame_count = 0
            self.duration = 100
            cv2.namedWindow('1')
            cv2.createTrackbar('duration', '1', 100, 200, self.changeduration)
            if t:
                video_cap = cv2.VideoCapture(path)
                all_frames = []
                while True:
                    ret, frame = video_cap.read()
                    if ret is False:
                        break
                    frame = frame[..., ::-1]  # opencv读取BGR，转成RGB
                    self.frames.append(frame)
                    self.frame_count += 1
                video_cap.release()
            else:
                img = cv2.imread(path)
                img = img[..., ::-1]
                self.frames.append(img)
                self.frame_count += 1

        def changeduration(self, x):
            self.duration = cv2.getTrackbarPos('duration', '1')

        def add(self, path):
            img = cv2.imread(path)
            img = img[..., ::-1]
            self.frames.append(img)
            self.frame_count += 1
            return self.frames, self.duration / 100

        def save(self, path):
            imageio.mimsave(path, self.frames, duration=self.duration / 100)
            return self.frames, self.duration / 100

        def show(self):
            for i in range(self.frame_count):
                img = self.frames[i][..., ::-1]
                cv2.imshow("gif", img)
                c = cv2.waitKey(int(1000 / (self.duration / 100)))
                if c == 27:
                    break
            cv2.destroyAllWindows()
            return self.frames, self.duration

    gif_img_path = QFileDialog.getOpenFileName(filter="Image (*.jpg *.png *.jpeg *.tif *.jfif)")[0]
    Gif = make_gif(gif_img_path, 0)
    addgif_img_path = QFileDialog.getOpenFileName(filter="Image (*.jpg *.png *.jpeg *.tif *.jfif)")[0]
    reset()
    Gif.add(addgif_img_path)

    while 1:
        k = cv2.waitKey(1)
        if k == 27:  # 按esc退
            cv2.destroyAllWindows()
        if k == 13:
            Gif.show()
            break


def insertWord():  # 输入文本需要制作ui界面
    def add_text_to_image():
        global processed_Image, underprocessing_Image
        # 初始化绘制状态和文本位置
        drawing = False
        text_position = (0, 0)
        user_text = ""  # 初始化用户输入的文本为空字符串

        # 鼠标事件处理函数，用于记录文本位置和绘制状态
        def draw_text(event, x, y, flags, param):
            nonlocal img_copy, drawing, text_position, user_text
            if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下事件
                drawing = True
                text_position = (x, y)  # 记录鼠标点击位置
            elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开事件
                if drawing:
                    user_text = input("请输入要添加的文本: ")  # 用户输入文本
                    change_text_properties(x)  # 调用修改文本属性函数
                drawing = False

        # 修改文本属性函数，根据滑动条和鼠标位置绘制文本
        def change_text_properties(x):
            nonlocal img_copy, user_text
            r = cv2.getTrackbarPos('R', 'image')  # 获取红色通道值
            g = cv2.getTrackbarPos('G', 'image')  # 获取绿色通道值
            b = cv2.getTrackbarPos('B', 'image')  # 获取蓝色通道值
            text_color = (b, g, r)  # 构建颜色值

            text_size = cv2.getTrackbarPos('Text Size', 'image') / 10.0  # 获取文本大小值

            font = cv2.FONT_HERSHEY_COMPLEX  # 字体类型

            # 在图像副本上绘制新的文本
            img_copy = processed_Image.copy()
            cv2.putText(img_copy, user_text, text_position, font, text_size, text_color, 3)

        # 读取图像
        img_copy = processed_Image.copy()  # 创建图像副本，用于绘制文本

        # 创建图像窗口并创建滑动条
        cv2.namedWindow('image')
        cv2.createTrackbar('R', 'image', 0, 255, change_text_properties)
        cv2.createTrackbar('G', 'image', 0, 255, change_text_properties)
        cv2.createTrackbar('B', 'image', 0, 255, change_text_properties)  # 添加文本颜色滑动条
        cv2.createTrackbar('Text Size', 'image', 0, 50, change_text_properties)  # 添加文本大小滑动条
        cv2.setMouseCallback('image', draw_text)  # 设置鼠标事件回调函数

        # 进入主循环，显示图像
        while True:
            cv2.imshow('image', img_copy)  # 显示图像副本
            if cv2.waitKey(1) & 0xFF == 27:  # 按下 ESC 键退出循环
                break

        underprocessing_Image = img_copy.copy()
        applyChanges()

        # 关闭图像窗口
        cv2.destroyAllWindows()

    add_text_to_image()  # 调用函数添加文本到图像


def insertPic():
    def add_overlay_to_image():
        global processed_Image, underprocessing_Image

        def _changesize(x):
            s = cv.getTrackbarPos('size', '1')  # 获取滑动条值
            nonlocal img2
            img2 = im2  # 重置叠加图像
            while s > 3:
                s -= 1
                img2 = cv.pyrUp(img2)  # 放大叠加图像
            while s < 3:
                s += 1
                img2 = cv.pyrDown(img2)  # 缩小叠加图像

        def _get_address(x, y):
            x2, y2 = x + img2.shape[1], y + img2.shape[0]  # 计算叠加图像的右下角坐标
            if x2 >= img1.shape[1]:
                x2 = img1.shape[1] - 1  # 防止超出源图像的右边界
            if y2 >= img1.shape[0]:
                y2 = img1.shape[0] - 1  # 防止超出源图像的下边界
            x1, y1 = x2 - img2.shape[1], y2 - img2.shape[0]  # 计算叠加图像的左上角坐标
            im = img2 == 0  # 创建叠加图像的掩码
            return x1, y1, x2, y2, im

        def _add(x1, y1, x2, y2, im):
            nonlocal img1
            img1 = img1.copy()  # 创建源图像的副本
            image1_ = img1[int(y1):int(y2), int(x1):int(x2)] * im  # 通过掩码提取源图像的一部分
            image1_1 = cv.add(image1_, img2)  # 将叠加图像与提取的部分相加
            img1[int(y1):int(y2), int(x1):int(x2)] = image1_1  # 将叠加后的图像放回源图像的相应位置

        def _take(event, x, y, flags, param):
            nonlocal img1
            if event == cv.EVENT_LBUTTONDOWN:  # 鼠标左键按下
                x1, y1, x2, y2, im = _get_address(x, y)  # 获取叠加图像的位置和掩码
                _add(x1, y1, x2, y2, im)  # 添加叠加图像到源图像
            if cv.waitKey(10) & 0xFF == 13:  # 按Enter键保存结果图像
                retval = cv.imwrite('new.png', img1)
                if retval:
                    print("保存成功")
                else:
                    print("保存失败")

        im1 = cv.cvtColor(processed_Image, cv.COLOR_BGR2BGRA)  # 将源图像转换为BGRA格式
        cv_img_path = QFileDialog.getOpenFileName(filter="Image (*.jpg *.png *.jpeg *.tif *.jfif)")[0]
        im2 = cv.imread(cv_img_path, cv.IMREAD_UNCHANGED)  # 读取叠加图像（包括alpha通道）
        im2 = cv.cvtColor(im2, cv.COLOR_BGR2BGRA)  # 将叠加图像转换为BGRA格式
        img1 = im1.copy()  # 创建源图像的副本
        img2 = im2.copy()  # 创建叠加图像的副本

        cv.namedWindow('1')  # 创建窗口
        cv.createTrackbar('size', '1', 3, 5, _changesize)  # 创建调整叠加图像大小的滑动条
        cv.imshow('1', img1)  # 显示源图像
        cv.setMouseCallback('1', _take)  # 设置鼠标事件回调函数

        while 1:
            cv.imshow('1', img1)  # 显示源图像
            k = cv.waitKey(1)
            if k == 27:  # 按Esc键退出
                cv.destroyAllWindows()
                break
            elif k == 13:  # 按Enter键保存结果图像
                retval = cv.imwrite('new.png', img1)
                if retval:
                    print("保存成功")
                else:
                    print("保存失败")
                break
        underprocessing_Image = img1.copy()
        applyChanges()
        cv.destroyAllWindows()

    add_overlay_to_image()  # 调用函数将叠加图像添加到源图像上


def saveImage():
    filename = QFileDialog.getSaveFileName(caption="Save Processed Image", filter='JPEG (*.jpg)')

    if filename == ('', ''):
        return

    img = processed_Image.copy()
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imsave(filename[0], img)
    else:
        plt.imsave(filename[0], img, cmap='gray')


####################################################################################
"""
    Mask Filter Functions
"""
number_of_points = 0
point1 = point2 = []
factor_h = factor_w = 0
mf_fourierImage = None


def mf_reset_points():
    global activeDialog, processed_Image, underprocessing_Image
    global number_of_points, point1, point2
    number_of_points = 0
    activeDialog.ui.label_point1MaskFilter.setText("point_1: ?")
    activeDialog.ui.label_point2MaskFilter.setText("point_2: ?")
    activeDialog.ui.apply_button_5.setEnabled(False)
    activeDialog.ui.pushButton_preview.setEnabled(False)
    underprocessing_Image = processed_Image
    mf_set_Fourier_Image(underprocessing_Image)


def mf_tabsChange():
    global activeDialog, processed_Image, underprocessing_Image, number_of_points

    if activeDialog.ui.tabs.currentIndex() == 5:  # mask filter
        number_of_points = 0
        mf_set_Fourier_Image(processed_Image)
        activeDialog.ui.image_label.mousePressEvent = lambda e: mf_mousePressed(e)
        activeDialog.ui.pushButton_reset.clicked.connect(mf_reset_points)
        activeDialog.ui.pushButton_preview.clicked.connect(mf_preview_result)
    else:
        underprocessing_Image = processed_Image
        refresh_dialog()


def mf_mousePressed(event):
    global number_of_points, point1, point2, factor_h, factor_w
    global original_Image

    hi, wi = original_Image.shape[:2]
    hl, wl = 340, 550
    factor_w = wi / wl
    factor_h = hi / hl

    y, x = (int(factor_w * event.pos().x()), int(factor_h * event.pos().y()))

    if number_of_points == 0:
        point1 = x, y
        text = "point_1: " + str(point1)
        activeDialog.ui.label_point1MaskFilter.setText(text)
        number_of_points += 1
    elif number_of_points == 1:
        point2 = (x, y)
        text = "point_2: " + str(point2)
        activeDialog.ui.label_point2MaskFilter.setText(text)
        activeDialog.ui.apply_button_5.setEnabled(True)
        activeDialog.ui.pushButton_preview.setEnabled(True)
        number_of_points += 1
        mf_processImage()
        mf_set_Fourier_Image(underprocessing_Image)


def mf_set_Fourier_Image(image):
    global activeDialog, processed_Image, underprocessing_Image, mf_fourierImage
    dftimg = shifted_dft(getGrayImage(image))
    mf_fourierImage = dftimg
    dftimg_gray = apply_cmap_to_image(dft_magnitude(dftimg), cmap='gray')
    processed_HF_pixmap = getPixmap(dftimg_gray)
    activeDialog.ui.image_label.setPixmap(processed_HF_pixmap)


def mf_processImage():
    global factor_h, factor_w
    global activeDialog, processed_Image, underprocessing_Image, mf_fourierImage
    if number_of_points == 2:
        underprocessing_Image = maskFilter(mf_fourierImage, point1, point2, filter_size=10)
        underprocessing_Image = inverse_shifted_dft(underprocessing_Image)


def mf_preview_result():
    global number_of_points
    global activeDialog, processed_Image, underprocessing_Image, mf_fourierImage
    mf_processImage()

    cv.imshow("After Mask Filter", np.uint8(underprocessing_Image))
    mf_set_Fourier_Image(underprocessing_Image)
    number_of_points = 0


####################################################################################
"""
    Dialogs Functions
"""


def applyChanges():
    global activeDialog, processed_Image, underprocessing_Image
    undo.append(processed_Image.copy())
    ui.undo_Button.setEnabled(True)
    processed_Image = underprocessing_Image.copy()
    update()
    if activeDialog is not None:
        activeDialog.close()


def undoChanges():
    global processed_Image, underprocessing_Image
    redo.append(processed_Image.copy())
    ui.redo_Button.setEnabled(True)
    processed_Image = undo.pop()
    if len(undo) == 0:
        ui.undo_Button.setEnabled(False)
    update()


def redoChanges():
    global processed_Image, underprocessing_Image
    undo.append(processed_Image.copy())
    ui.undo_Button.setEnabled(True)
    processed_Image = redo.pop()
    if len(redo) == 0:
        ui.redo_Button.setEnabled(False)
    update()


def updateFilter():
    global activeDialog, processed_Image, underprocessing_Image
    slider1 = activeDialog.ui.slider1.value()
    slider2 = activeDialog.ui.slider2.value()
    dx = activeDialog.ui.dx_checkBox.isChecked()
    dy = activeDialog.ui.dy_checkBox.isChecked()

    if slider1 & 1 ^ 1:
        slider1 += 1
        activeDialog.ui.slider1.setValue(slider1)
    if slider2 & 1 ^ 1:
        slider2 += 1
        activeDialog.ui.slider2.setValue(slider2)

    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Sobel Filter
        if dx or dy:
            underprocessing_Image = sobel_filter(processed_Image, slider1, dx, dy)
    elif idx == 1:  # Laplace Filter
        underprocessing_Image = laplacian_filter(processed_Image, slider2)

    refresh_dialog()


def updateNoiseAdd():
    global activeDialog, processed_Image, underprocessing_Image

    slider1 = activeDialog.ui.slider1.value() / 100.0
    slider2 = activeDialog.ui.slider2.value() / 100.0

    slider3 = activeDialog.ui.slider3.value() / 100.0
    slider4 = activeDialog.ui.slider4.value() / 100.0

    slider5 = activeDialog.ui.slider5.value()
    slider6 = activeDialog.ui.slider6.value()
    slider7 = activeDialog.ui.slider7.value()
    slider8 = activeDialog.ui.slider8.value()

    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))
    activeDialog.ui.slider3_counter.setText(str(slider3))
    activeDialog.ui.slider4_counter.setText(str(slider4))
    activeDialog.ui.slider5_counter.setText(str(slider5))
    activeDialog.ui.slider6_counter.setText(str(slider6))
    activeDialog.ui.slider7_counter.setText(str(slider7))
    activeDialog.ui.slider8_counter.setText(str(slider8))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Salt and Pepper Noise
        underprocessing_Image = add_salt_and_pepper_noise(processed_Image, slider1, slider2)
    elif idx == 1:  # Gaussian Noise
        underprocessing_Image = gaussianNoise(processed_Image, slider3, slider4)
    elif idx == 2:  # Periodic Noise
        underprocessing_Image = add_periodic_noise(processed_Image, slider5, slider6, slider7, slider8)

    refresh_dialog()

def updateNoiseRemove():
    global activeDialog, processed_Image, underprocessing_Image

    slider1 = activeDialog.ui.slider1.value()
    slider2 = activeDialog.ui.slider2.value()

    slider3 = activeDialog.ui.slider3.value()

    slider4 = activeDialog.ui.slider4.value()
    slider5 = activeDialog.ui.slider5.value()
    slider6 = activeDialog.ui.slider6.value()

    slider7 = activeDialog.ui.slider7.value()

    slider8 = activeDialog.ui.slider8.value()

    if slider3 & 1 ^ 1:
        slider3 += 1
        activeDialog.ui.slider3.setValue(slider3)
    if slider4 & 1 ^ 1:
        slider4 += 1
        activeDialog.ui.slider4.setValue(slider4)

    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))
    activeDialog.ui.slider3_counter.setText(str(slider3))
    activeDialog.ui.slider4_counter.setText(str(slider4))
    activeDialog.ui.slider5_counter.setText(str(slider5))
    activeDialog.ui.slider6_counter.setText(str(slider6))
    activeDialog.ui.slider7_counter.setText(str(slider7))
    activeDialog.ui.slider8_counter.setText(str(slider8))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Average Filter
        if slider1 > 0 and slider2 > 0:
            underprocessing_Image = averaging_filter(processed_Image, slider1, slider2)
    elif idx == 1:  # Median Filter
        underprocessing_Image = median_filter(processed_Image, slider3)
    elif idx == 2:  # Gaussian Blur
        underprocessing_Image = gaussianFilter(processed_Image, slider4, slider5, slider6)
    elif idx == 3:  # Notch Filter
        underprocessing_Image = notch_filter(getGrayImage(processed_Image), slider7)
    elif idx == 4:  # Band Filter
        underprocessing_Image = band_filter(getGrayImage(processed_Image), slider8)
    elif idx == 5:  # Mask Filter
        mf_reset_points()
        pass

    refresh_dialog()

def updateLightness():
    global activeDialog, processed_Image, underprocessing_Image
    slider1 = activeDialog.ui.slider1.value()
    slider2 = activeDialog.ui.slider1.value()
    slider3 = activeDialog.ui.slider1.value()
    slider4 = activeDialog.ui.slider4.value()

    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))
    activeDialog.ui.slider3_counter.setText(str(slider3))
    activeDialog.ui.slider4_counter.setText(str(slider4))

    underprocessing_Image = changeLightness(processed_Image, slider1, slider2,slider3,slider4)

    refresh_dialog()


dialogs = [Ui_filterDialog, Ui_noiseAddDialog, Ui_noiseRemoveDialog,Ui_lightnessDialog]
updateFunctions = [updateFilter, updateNoiseAdd, updateNoiseRemove,updateLightness]

activeDialog = None


def setup_dialog(dialogUI):
    global activeDialog
    activeDialog = QtWidgets.QDialog(None, QtCore.Qt.WindowCloseButtonHint)
    activeDialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    activeDialog.mousePressEvent = mousePressEvent
    activeDialog.mouseMoveEvent = mouseMoveEvent
    activeDialog.ui = dialogUI()
    activeDialog.ui.setupUi(activeDialog)


def refresh_dialog():
    global activeDialog, underprocessing_Image
    activeDialog.ui.image_label.setPixmap(getPixmap(underprocessing_Image))
    activeDialog.ui.image_label.setScaledContents(True)


def show_dialog(idx):
    global activeDialog, underprocessing_Image, processed_Image
    setup_dialog(dialogs[idx])
    underprocessing_Image = processed_Image.copy()
    refresh_dialog()

    components = activeDialog.ui.getComponents()
    for c in components:
        if type(c) is QtWidgets.QTabWidget:
            activeDialog.ui.tabs.currentChanged.connect(mf_tabsChange)
        if type(c) is QtWidgets.QPushButton:
            c.clicked.connect(applyChanges)
        if type(c) is QtWidgets.QSlider:
            c.valueChanged['int'].connect(updateFunctions[idx])
        if type(c) is QtWidgets.QCheckBox:
            c.clicked.connect(updateFunctions[idx])

    activeDialog.ui.close_button.clicked.connect(close_dialog)
    activeDialog.ui.minimize_button.clicked.connect(activeDialog.showMinimized)
    activeDialog.exec_()


def close_dialog():
    global activeDialog
    activeDialog.close()
    activeDialog = None


#######################################################################

loadImage(True)

ui.close_button.clicked.connect(app.exit)
ui.minimize_button.clicked.connect(MainWindow.showMinimized)

ui.loadImage_Button.clicked.connect(loadImage)
ui.reset_Button.clicked.connect(reset)
ui.undo_Button.clicked.connect(undoChanges)
ui.redo_Button.clicked.connect(redoChanges)
ui.saveImage_Button.clicked.connect(saveImage)

# ui.histogram_radioButton.clicked.connect(updateHF)
# ui.fourier_radioButton.clicked.connect(updateHF)

ui.lightness_Button.clicked.connect(lambda: show_dialog(3))
ui.coding_Button.clicked.connect(Coding)
ui.delete_Button.clicked.connect(deletePic)
ui.gif_Button.clicked.connect(Gif)
ui.insertWord_Button.clicked.connect(insertWord)
ui.insertPic_Button.clicked.connect(insertPic)
ui.special_Button.clicked.connect(lambda: show_dialog(1))
ui.change_Button.clicked.connect(lambda: show_dialog(2))


#######################################################################

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


sys.excepthook = except_hook

sys.exit(app.exec_())
