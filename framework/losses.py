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
from scipy import signal
import math
import ctypes
import os

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "losses.lib"))

_lib.parzen_mutual_information_grad.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]

_lib.parzen_mutual_information_matrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]

_lib.parzen_mutual_information_point.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]

_lib.parzen_mutual_information_point_grad.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]

_lib.parzen_mutual_information_point_matrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]

_lib.get_gradient.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]


epsilon = np.finfo(np.float32).tiny

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


class MeanSquaredErrorLoss():
    def __call__(self, fixed, moving):
        diff = moving.flatten().reshape(1, -1) - fixed.flatten().reshape(
            1, -1
        )
        return np.sum((diff) ** 2), 2 * diff


class CrossCorrelationLoss():
    def __call__(self, fixed, moving):
        return -np.sum(fixed * moving), -fixed.flatten()


class MutualInformationLoss():
    def __init__(self, n_bins=256, padded=False):
        self.n_bins = n_bins
        self.omega = np.array([[1 / 6, 2 / 3, 1 / 6]])
        self.omega_prime = -np.array([[1 / 2, 0, -1 / 2]])
        #self.omega = np.array([[1./9, 2./9, 1./3, 2./9, 1./9]])
        #self.omega_prime = -np.array([[1./9, 1./9, 0, -1./9, -1./9]])
        #self.omega = np.array([[1./16, 2./16, 3./16, 4./16, 3./16, 2./16, 1./16]])
        #self.omega_prime = -np.array([[1./16, 1./16, 1./16, 0, -1./16, -1./16, -1./16]])
        self.filter = np.dot(self.omega.transpose(), self.omega)
        self.filter_prime = np.dot(self.omega_prime.transpose(), self.omega)
        self.filter_prime_sum = np.sum(self.filter_prime)
        self.filter_prime_j = self.omega_prime.transpose() * np.sum(self.omega)
        if padded:
            self.pad_size = self.omega.size//2
        else:
            self.pad_size = 0

    def compute(self, fixed, moving):
        count_matrix = np.zeros((self.n_bins, self.n_bins))
        fixed = np.clip(fixed, 0, self.n_bins-1)
        moving = np.clip(moving, 0, self.n_bins-1)
        for f, m in zip(
            fixed.flatten().astype(int), moving.flatten().astype(int)
        ):
            count_matrix[m, f] += 1

        if self.pad_size > 0:
            count_matrix = np.pad(count_matrix, self.pad_size, pad_with)

        prob_matrix = signal.correlate2d(count_matrix, self.filter, "same")

        prob_matrix /= fixed.size

        prob_k = np.sum(prob_matrix, axis=0)
        prob_j = np.sum(prob_matrix, axis=1)

        bigc = np.sum(self.filter_prime)

        pjk_over_pj = np.copy(prob_matrix)
        for j in range(self.n_bins + self.pad_size*2):
            if prob_j[j] == 0:
                pjk_over_pj[j, :] /= epsilon
            else:
                pjk_over_pj[j, :] /= prob_j[j]

        logs = np.zeros((self.n_bins + self.pad_size*2, self.n_bins + self.pad_size*2))
        for j in range(self.n_bins + self.pad_size*2):
            for k in range(self.n_bins + self.pad_size*2):
                denom = prob_j[j] * prob_k[k]
                if denom == 0:
                    denom = epsilon
                num = prob_matrix[j, k]
                if num == 0:
                    num = epsilon
                frac = num / denom
                logs[j, k] = math.log(frac)

        res = 0

        for j in range(self.n_bins + self.pad_size*2):
            for k in range(self.n_bins + self.pad_size*2):
                res += prob_matrix[j, k] * logs[j, k]

        return -res

    def compute_gradient(self, fixed, moving):
        count_matrix = np.zeros((self.n_bins, self.n_bins))
        fixed = np.clip(fixed, 0, self.n_bins-1)
        moving = np.clip(moving, 0, self.n_bins-1)
        for f, m in zip(
            fixed.flatten().astype(int), moving.flatten().astype(int)
        ):
            count_matrix[m, f] += 1

        if self.pad_size > 0:
            count_matrix = np.pad(count_matrix, self.pad_size, pad_with)

        prob_matrix = signal.correlate2d(count_matrix, self.filter, "same")

        prob_matrix /= fixed.size

        prob_k = np.sum(prob_matrix, axis=0)
        prob_j = np.sum(prob_matrix, axis=1)

        bigc = np.sum(self.filter_prime)

        pjk_over_pj = np.copy(prob_matrix)
        for j in range(self.n_bins + self.pad_size*2):
            if prob_j[j] == 0:
                pjk_over_pj[j, :] /= epsilon
            else:
                pjk_over_pj[j, :] /= prob_j[j]

        logs = np.zeros((self.n_bins + self.pad_size*2, self.n_bins + self.pad_size*2))
        for j in range(self.n_bins + self.pad_size*2):
            for k in range(self.n_bins + self.pad_size*2):
                denom = prob_j[j] * prob_k[k]
                if denom == 0:
                    denom = epsilon
                num = prob_matrix[j, k]
                if num == 0:
                    num = epsilon
                frac = num / denom
                logs[j, k] = math.log(frac)

        alpha_matrix = signal.correlate2d(logs, self.filter_prime, "same")
        beta_matrix = -signal.correlate2d(
            pjk_over_pj, self.filter_prime_j, "same"
        ) + bigc

        derivative_matrix = alpha_matrix + beta_matrix

        derivatives = []
        for f, m in zip(
            fixed.flatten().astype(int), moving.flatten().astype(int)
        ):
            derivatives.append(derivative_matrix[m, f])

        derivatives = -np.array(derivatives)

        return derivatives

    def __call__(self, fixed, moving):
        count_matrix = np.zeros((self.n_bins, self.n_bins))
        fixed = np.clip(fixed, 0, self.n_bins-1)
        moving = np.clip(moving, 0, self.n_bins-1)
        for f, m in zip(
            fixed.flatten().astype(int), moving.flatten().astype(int)
        ):
            count_matrix[m, f] += 1

        if self.pad_size > 0:
            count_matrix = np.pad(count_matrix, self.pad_size, pad_with)

        prob_matrix = signal.correlate2d(count_matrix, self.filter, "same")

        prob_matrix /= fixed.size

        prob_k = np.sum(prob_matrix, axis=0)
        prob_j = np.sum(prob_matrix, axis=1)

        bigc = np.sum(self.filter_prime)

        pjk_over_pj = np.copy(prob_matrix)
        for j in range(self.n_bins + self.pad_size*2):
            if prob_j[j] == 0:
                pjk_over_pj[j, :] /= epsilon
            else:
                pjk_over_pj[j, :] /= prob_j[j]

        logs = np.zeros((self.n_bins + self.pad_size*2, self.n_bins + self.pad_size*2))
        for j in range(self.n_bins + self.pad_size*2):
            for k in range(self.n_bins + self.pad_size*2):
                denom = prob_j[j] * prob_k[k]
                if denom == 0:
                    denom = epsilon
                num = prob_matrix[j, k]
                if num == 0:
                    num = epsilon
                frac = num / denom
                logs[j, k] = math.log(frac)

        res = 0

        for j in range(self.n_bins + self.pad_size*2):
            for k in range(self.n_bins + self.pad_size*2):
                res += prob_matrix[j, k] * logs[j, k]

        alpha_matrix = signal.correlate2d(logs, self.filter_prime, "same")
        beta_matrix = -signal.correlate2d(
            pjk_over_pj, self.filter_prime_j, "same"
        ) + bigc

        derivative_matrix = alpha_matrix + beta_matrix

        derivatives = []
        for f, m in zip(
            fixed.flatten().astype(int), moving.flatten().astype(int)
        ):
            derivatives.append(derivative_matrix[m, f])

        res = -res
        derivatives = -np.array(derivatives)

        return res, derivatives



class MutualInformationLossNative():
    def __init__(self):
        pass

    def compute(self, fixed, moving):
        fixed = np.clip(fixed, 0, 255)
        moving = np.clip(moving, 0, 255)
        fixed = fixed.flatten().astype(np.uint8)
        moving = moving.flatten().astype(np.uint8)

        res = np.empty(1, dtype=np.float32)

        _lib.parzen_mutual_information_point(fixed, moving, len(fixed), res)

        return res[0]


    def compute_gradient(self, fixed, moving):
        fixed = np.clip(fixed, 0, 255)
        moving = np.clip(moving, 0, 255)
        fixed = fixed.flatten().astype(np.uint8)
        moving = moving.flatten().astype(np.uint8)

        derivs = np.empty(len(fixed), dtype=np.float32)

        _lib.parzen_mutual_information_grad(fixed, moving, len(fixed), derivs)

        return derivs


    def compute_gradient_matrix(self, fixed, moving):
        fixed = np.clip(fixed, 0, 255)
        moving = np.clip(moving, 0, 255)
        fixed = fixed.flatten().astype(np.uint8)
        moving = moving.flatten().astype(np.uint8)

        matrix = np.empty(256*256, dtype=np.float32)

        _lib.parzen_mutual_information_matrix(fixed, moving, len(fixed), matrix)

        return matrix


    def __call__(self, fixed, moving):
        fixed = np.clip(fixed, 0, 255)
        moving = np.clip(moving, 0, 255)
        fixed = fixed.flatten().astype(np.uint8)
        moving = moving.flatten().astype(np.uint8)

        res = np.empty(1, dtype=np.float32)
        derivs = np.empty(len(fixed), dtype=np.float32)

        _lib.parzen_mutual_information_point_grad(fixed, moving, len(fixed), res, derivs)

        return res[0], derivs


class MutualInformationLossFPGA():
    def __init__(self, mi_ip, fixed_buf, moving_buf, res_buf):
        self.mi_ip = mi_ip
        self.fixed_buf = fixed_buf
        self.moving_buf = moving_buf
        self.res_buf = res_buf

    def __call__(self, fixed, moving):
        fixed = np.clip(fixed, 0, 255)
        moving = np.clip(moving, 0, 255)
        fixed = fixed
        moving = moving
        fixed = fixed.flatten().astype(np.uint8)
        moving = moving.flatten().astype(np.uint8)
        self.moving_buf[:] = moving
        self.moving_buf.flush()

        self.mi_ip.write(0x10, self.fixed_buf.physical_address)
        self.mi_ip.write(0x18, self.moving_buf.physical_address)
        self.mi_ip.write(0x20, self.res_buf.physical_address)
        self.mi_ip.write(0x00, 1)
        while self.mi_ip.read(0x00) & 0x04 != 0x04:
            pass

        self.res_buf.invalidate()
        
        derivs = np.empty(len(fixed), dtype=np.float32)
        
        _lib.get_gradient(moving, fixed, len(fixed), 256, self.res_buf, derivs)

        #derivs = [self.res_buf[256*m+f] for m, f in zip(self.moving_buf, self.fixed_buf)]

        return 0, derivs

