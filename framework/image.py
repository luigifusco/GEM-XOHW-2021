#/******************************************
#*MIT License
#*
# *Copyright (c) [2021] [Luigi Fusco, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# */
import numpy as np
import math
from scipy.ndimage import sobel
import cv2
import pydicom
import os

class SobelGradient():
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        #grad_x = sobel(img, axis=-1)
        #grad_y = sobel(img, axis=0)
        img = img.astype(np.uint16)
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=self.k)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=self.k)

        return grad_x, grad_y

class SimpleGradient():
    def __call__(self, img):
        grad_x = np.zeros(img.shape)
        grad_x[:, :-1] = img[:, 1:] - img[:, :-1]

        grad_y = np.zeros(img.shape)
        grad_y[:-1, :] = img[1:, :] - img[:-1, :]

        return grad_x.flatten(), grad_y.flatten()


def elliptic_paraboloid(width, pad):
    img = np.zeros((width + pad * 2, width + pad * 2))
    for i in range(width):
        for j in range(width):
            img[i + pad, j + pad] = (
                -((i - width / 2) ** 2 * (j - width / 2) ** 2) / ((width / 2) ** 4) + 1
            )
            img[i + pad, j + pad] = img[i + pad, j + pad] * 255
            img[i + pad, j + pad] = math.floor(img[i + pad, j + pad])

    return img


def get_medical_pair(name, size=128, basepath='datasettmp'):
    dcm1 = pydicom.dcmread(os.path.join(basepath, 'SE0', name))
    img1 = cv2.resize(dcm1.pixel_array, dsize=(size,size))
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    dcm2 = pydicom.dcmread(os.path.join(basepath, 'NuovoSE2', name))
    img2 = cv2.resize(dcm2.pixel_array, dsize=(size,size))
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img1, img2