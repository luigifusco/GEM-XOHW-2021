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
import itertools
from functools import partial
from scipy.ndimage.interpolation import affine_transform
import ctypes
import os

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "transforms.lib"))

_lib.rotate_shift_transform_derivatives.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')
]

class Transform():
    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=np.float64)

    def __call__(self, moving, grad=None):
        pass


class ShiftTransform(Transform):
    def __init__(self, parameters=[0, 0]):
        super().__init__(parameters)
        self.__const_gradients = (np.array([[1, 0]]), np.array([[0, 1]]))

    def __call__(self, moving, grad=None):
        #shift = [round(elem) for elem in self.parameters]

        moved = np.zeros(moving.shape)
        for j, i in itertools.product(range(moving.shape[0]), range(moving.shape[1])):
            j_alt = round(j + self.parameters[1])
            i_alt = round(i + self.parameters[0])
            if (
                j_alt >= 0
                and j_alt < moving.shape[0]
                and i_alt >= 0
                and i_alt < moving.shape[1]
            ):
                moved[j, i] = moving[j_alt, i_alt]

        #moved = affine_transform(moving, np.array([[1,0],[0,1]]), np.array([self.parameters[0], self.parameters[1]]))

        if grad is None:
            return moved
        else:
            image_gradient = grad(moving)

            image_transform_gradient = np.stack((image_gradient[0], image_gradient[1]), axis=-1)

            return moved, image_transform_gradient
            

class AffineTransform(Transform):
    '''
    Note that b is flipped, A is flipped in both directions
    '''
    def __init__(self, parameters=[1, 0, 0, 1, 0, 0], alpha=0.001, beta=None):
        super().__init__(parameters)
        self.alpha = alpha
        if beta is None:
            self.beta = alpha
        else:
            self.beta = beta
        self.image_gradient = None

    @property
    def A(self):
        return self.parameters[:4].reshape((2, 2))

    @property
    def b(self):
        return self.parameters[-2:]

    def __call__(self, moving, grad=None):
        moved = affine_transform(moving, self.A, self.b)

        if grad is None:
            return moved
        else:
            if self.image_gradient is None:
                self.image_gradient = grad(moving)


            image_gradient_x = affine_transform(self.image_gradient[0], self.A, self.b)
            image_gradient_y = affine_transform(self.image_gradient[1], self.A, self.b)

            grads = np.array(
                [
                    [y * j * self.alpha, x * j * self.beta, y * i * self.beta, x * i * self.alpha, j, i]
                    for (y, x), i, j in zip(itertools.product(range(moving.shape[0]), range(moving.shape[1])), image_gradient_x.flatten(), image_gradient_y.flatten())
                ]
            )

        return moved, grads


class RotateShiftTransform(Transform):
    def __init__(self, parameters=[0, 0, 0], alpha=0.001):
        super().__init__(parameters)
        self.alpha = alpha
        self.image_gradient = None

    @property
    def A(self):
        theta = self.parameters[0]
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    @property
    def b(self):
        return self.parameters[-2:]

    def __call__(self, moving, grad=None):
        moved = affine_transform(moving, self.A, self.b)

        if grad is None:
            return moved
        else:
            if self.image_gradient is None:
                self.image_gradient = grad(moving)


            image_gradient_x = affine_transform(self.image_gradient[0], self.A, self.b)
            image_gradient_y = affine_transform(self.image_gradient[1], self.A, self.b)

            theta = self.parameters[0]

            grads = np.empty((moving.size, 3))

            _lib.rotate_shift_transform_derivatives(moving.shape[0], moving.shape[1], image_gradient_x.flatten().astype(np.double), image_gradient_y.flatten().astype(np.double), theta, self.alpha, grads)

            #grads = np.array(
            #   [
            #        [self.alpha*(i*(- x*np.sin(theta) - y*np.cos(theta)) + j*(x*np.cos(theta) - y*np.sin(theta))), j, i]
            #        for (y, x), i, j in zip(itertools.product(range(moving.shape[0]), range(moving.shape[1])), image_gradient_x.flatten(), image_gradient_y.flatten())
            #    ]
            #)

        return moved, grads