/******************************************
*MIT License
*
*Copyright (c) [2021] [Luigi Fusco, Eleonora D'Arnese, Marco Domenico Santambrogio]
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/
#include <math.h>
#include <stdio.h>
#include <float.h>

#define PRECOMPUTE

#define VERTICAL true
#define HORIZONTAL false

const int F = 3;

extern "C" {
    void parzen_mutual_information_grad(unsigned char* I_m, unsigned char* I_f, int N, float *mi_deriv);
    void parzen_mutual_information_matrix(unsigned char* I_m, unsigned char* I_f, int N, float *mi_deriv);
    void parzen_mutual_information_point(unsigned char* I_m, unsigned char* I_f, int N, float *mi);
    void parzen_mutual_information_point_grad(unsigned char* I_m, unsigned char* I_f, int N, float *mi, float *mi_deriv);
    void parzen_mutual_information_point_matrix(unsigned char* I_m, unsigned char* I_f, int N, float *mi, float *mi_deriv);
    void get_gradient(unsigned char* I_m, unsigned char* I_f, int N, int W, float* matrix, float *grad);
}

/*
    function accelerating the pixel wise gradient extraction procedure from the gradient matrix
*/
void get_gradient(unsigned char* I_m, unsigned char* I_f, int N, int W, float* matrix, float *grad) {
    for (int i = 0; i < N; ++i) {
        int curr_m = I_m[i];
        int curr_f = I_f[i];
        
        grad[i] = matrix[curr_m*W+curr_f];
    }
}

/*
    T: type of input (int or float)
    H: height of matrix
    W: width of matrix
    K: size of kernel
    ver: true -> vertical, false -> horizontal
    m: pointer to matrix (as line vector of size H*W)
    k: pointer to kernel (vector of size K)
    out: pointer to output (output, line vector of size H*W)
*/
template <typename T, int H, int W, int K, bool ver>
void convolution(T* m, float* k, float* out) {
    int pad = K/2;
    for (int j = 0; j < H; ++j) for (int i = 0; i < W; ++i) {
            float acc = 0;
            for (int b = -pad; b < pad+1; ++b) {
                if (ver) {
                    if (j+b >= 0 && j+b < H) acc += m[(j+b)*W + i]*k[b+pad];
                }
                else {
                    if (i+b >= 0 && i+b < W) acc += m[j*W + i + b]*k[b+pad];
                }
            }
            out[j*W + i] = acc;
        }
}

/*
    sum of omega must be zero for normalization to work!
    I_f: pointer to fixed image data (array of N)
    I_m: pointer to moving image data (array of N)
    N: size of input
    mi: pointer to mutual information value (output, single float)
    mi_deriv: pointer to mutual information derivatives (output, array of N floats)

    B: number of bins
    POINT: if point mutual information is returned
    GRAD: if gradients are returned
    MATRIX: if matrix of gradients is returned instead of pixel wise gradients
*/
template <int B, bool POINT, bool GRAD, bool MATRIX>
void mutual_information_backend(unsigned char* I_f, unsigned char* I_m, int N, float* mi, float *mi_deriv) {
    int counting_matrix[B][B] = { 0 };
    static float buffer_matrix[B][B]; // used only for partial calculation, probably skippable if output of convolution can be same vector as input
    static float prob_matrix[B][B];
    float omega[F] = { 1./6., 2./3., 1./6. };
    float omega_deriv[F] = { -1./2., 0., 1./2. };
    float omega_deriv_k[F] = { -1./2., 0., 1./2. };


    for (int i = 0; i < N; ++i) {
        counting_matrix[I_m[i]][I_f[i]]++;
    }


    convolution<int, B, B, F, VERTICAL>((int*)counting_matrix, omega, (float*)buffer_matrix);
    convolution<float, B, B, F, HORIZONTAL>((float*)buffer_matrix, omega, (float*)prob_matrix);

    for (int j = 0; j < B; ++j) for (int k = 0; k < B; ++k)
            prob_matrix[j][k] /= (float)N;

    float prob_j[B] = { 0 };
    for (int j = 0; j < B; ++j) for (int k = 0; k < B; ++k)
            prob_j[j] += prob_matrix[j][k];

    float prob_k[B] = { 0 };
    for (int j = 0; j < B; ++j) for (int k = 0; k < B; ++k)
            prob_k[k] += prob_matrix[j][k];

    float pjk_over_pk[B][B];
    for (int j = 0; j < B; ++j) for (int k = 0; k < B; ++k)
            pjk_over_pk[j][k] = prob_matrix[j][k] / prob_k[k];

    float logs_matrix[B][B];
    for (int j = 0; j < B; ++j) for (int k = 0; k < B; ++k) {
            float denom = prob_j[j] * prob_k[k];
            if (denom == 0)
                denom = DBL_MIN;

            float num = prob_matrix[j][k];
            if (num == 0)
                num = DBL_MIN;

            float l = logf(num/denom);
            logs_matrix[j][k] = isinf(l) ? -DBL_MAX : l;
        }

    if (POINT) {

        float res = 0;
        for (int j = 0; j < B; ++j) for (int k = 0; k < B; ++k)
                res += prob_matrix[j][k] * logs_matrix[j][k]; 

        *mi = -res;

    }

    if (GRAD) {

        float bigc = 0;

        /* needed for other window functions
        for (int i = 0; i < F; ++i) for (int j = 0; j < F; ++j)
                bigc += omega[i] * omega_deriv[j];
        */
        
        #ifdef PRECOMPUTE
        // precompute all possible derivative values through a full convolution

            static float alpha_matrix[B][B];
            convolution<float, B, B, F, VERTICAL>((float*)logs_matrix, omega_deriv, (float*)buffer_matrix);
            convolution<float, B, B, F, HORIZONTAL>((float*)buffer_matrix, omega, (float*)alpha_matrix);
            
            static float beta_matrix[B][B];
            convolution<float, B, B, F, VERTICAL>((float*)pjk_over_pk, omega_deriv_k, (float*)beta_matrix);

            if (MATRIX) {
                for (int i = 0; i < B; ++i) {
                    for (int j = 0; j < B; ++j) {
                        mi_deriv[i*B+j] = beta_matrix[i][j] - bigc - alpha_matrix[i][j];
                    }
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    int m_idx = I_m[i],
                        f_idx = I_f[i];

                    mi_deriv[i] = beta_matrix[m_idx][f_idx] - bigc - alpha_matrix[m_idx][f_idx];
                }
            }


        #else
        // computes derivative values only for actual pixels of the image.
        // for every pair of pixels computes a single step of the convolution above.
        // probably optimizable with cache

        int pad = F/2;

        for (int i = 0; i < N; ++i) {
            int m_idx = I_m[i],
                f_idx = I_f[i];

            float alpha = 0;
            for (int a = -pad; a < pad+1; ++a) for (int b = -pad; b < pad+1; ++b)
                    if (m_idx+a>0 && m_idx+a<B && f_idx+b>0 && f_idx+b<B)
                        alpha += logs_matrix[m_idx+a][f_idx+b] * omega[b+pad] * omega_deriv[a+pad];

            float beta = 0;
            for (int a = -pad; a < pad+1; ++a)
                if (m_idx+a>0 && m_idx+a<B)
                    beta += pjk_over_pk[m_idx+a][f_idx] * omega_deriv_k[a+pad];

            mi_deriv[i] = beta - bigc - alpha;
        }

        #endif

    }
}

void parzen_mutual_information_grad(unsigned char* I_m, unsigned char* I_f, int N, float *mi_deriv) {
    mutual_information_backend<256, false, true, false>(I_m, I_f, N, NULL, mi_deriv);
}

void parzen_mutual_information_matrix(unsigned char* I_m, unsigned char* I_f, int N, float *mi_deriv) {
    mutual_information_backend<256, false, true, true>(I_m, I_f, N, NULL, mi_deriv);
}

void parzen_mutual_information_point(unsigned char* I_m, unsigned char* I_f, int N, float *mi) {
    mutual_information_backend<256, true, false, false>(I_m, I_f, N, mi, NULL);
}

void parzen_mutual_information_point_grad(unsigned char* I_m, unsigned char* I_f, int N, float *mi, float *mi_deriv) {
    mutual_information_backend<256, true, true, false>(I_m, I_f, N, mi, mi_deriv);
}

void parzen_mutual_information_point_matrix(unsigned char* I_m, unsigned char* I_f, int N, float *mi, float *mi_deriv) {
    mutual_information_backend<256, true, true, true>(I_m, I_f, N, mi, mi_deriv);
}