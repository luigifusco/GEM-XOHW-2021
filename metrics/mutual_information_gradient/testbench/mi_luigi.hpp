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

#include <iostream>
#include <fstream>

typedef float type_t;

#define PRECOMPUTE

// IMPORTANT: used all over the place to reduce verbosity
#define range(I,S,N) for(int I=S;I<N;++I)

#define VERTICAL true
#define HORIZONTAL false

const int F = 3;

/*
    T: type of input (int or double)
    H: height of matrix
    W: width of matrix
    K: size of kernel
    ver: true -> vertical, false -> horizontal
    m: pointer to matrix (as line vector of size H*W)
    k: pointer to kernel (vector of size K)
    out: pointer to output (output, line vector of size H*W)
*/
template <typename T, int H, int W, int K, bool ver>
void convolution(T* m, type_t k[K], type_t* out) {
    int pad = K/2;
    range(j, 0, H) range(i, 0, W) {
            type_t acc = 0;
            range(b, -pad, pad+1) {
                if (ver) {
                    if (j+b >= 0 && j+b < H) acc += ((type_t)(m[(j+b)*W + i]))*k[b+pad];
                }
                else {
                    if (i+b >= 0 && i+b < W) acc += ((type_t)(m[j*W + i + b]))*k[b+pad];
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
    mi: pointer to mutual information value (output, single double)
    mi_deriv: pointer to mutual information derivatives (output, array of N doubles)
*/
template <int B>
void mutual_information(unsigned char* I_f, unsigned char* I_m, int N, type_t* mi, type_t *mi_deriv) {
    int counting_matrix[B+2][B+2] = { 0 }; 
    static type_t buffer_matrix[B+2][B+2]; // used only for partial calculation, probably skippable if output of convolution can be same vector as input
    static type_t prob_matrix[B+2][B+2];
    type_t omega[F] = { 1./6., 4./6., 1./6. };
    type_t omega_deriv[F] = { -1./2., 0., 1./2. };
    type_t omega_deriv_k[F] = { -1./2., 0., 1./2. };


    range(i, 0, N)
        counting_matrix[I_m[i]+1][I_f[i]+1]++;

    convolution<int, B+2, B+2, F, VERTICAL>((int*)counting_matrix, omega, (type_t*)buffer_matrix);
    convolution<type_t, B+2, B+2, F, HORIZONTAL>((type_t*)buffer_matrix, omega, (type_t*)prob_matrix);

    range(j, 0, B+2) range(k, 0, B+2)
        prob_matrix[j][k] /= (type_t)N;    

    type_t prob_j[B+2] = { 0 };
    range(j, 0, B+2) range(k, 0, B+2)
            prob_j[j] += prob_matrix[j][k];


    type_t prob_k[B+2] = { 0 };
    range(j, 0, B+2) range(k, 0, B+2)
            prob_k[k] += prob_matrix[j][k];

    type_t pjk_over_pk[B+2][B+2];
    range(j, 0, B+2) range(k, 0, B+2)
            pjk_over_pk[j][k] = prob_matrix[j][k] / prob_k[k];

    type_t logs_matrix[B+2][B+2];
    range(j, 0, B+2) range(k, 0, B+2) {
            /*type_t denom = prob_j[j] * prob_k[k];
            if (denom == 0)
                denom = DBL_MIN;

            type_t num = prob_matrix[j][k];
            if (num == 0)
                num = DBL_MIN;

            type_t l = log2(num/denom);
            logs_matrix[j][k] = isinf(l) ? -DBL_MAX : l;*/
            type_t denom = prob_j[j] * prob_k[k];
            type_t num = prob_matrix[j][k];
            type_t l;

            if (denom == 0) {
                l = 0;
            } else if (num == 0) {
                l = -1000000000;
            } else {
                l = log2(num/denom);
            }

            logs_matrix[j][k] = l;
        }

    type_t res = 0;
    range(j, 0, B+2) range(k, 0, B+2)
            res += prob_matrix[j][k] * logs_matrix[j][k]; 

    if (mi) *mi = -res;

    if (mi_deriv == NULL) return;

    type_t bigc = 0;
    range(i, 0, F) range(j, 0, F)
            bigc += omega[i] * omega_deriv[j];
    
    #ifdef PRECOMPUTE
    // precompute all possible derivative values through a full convolution

        static type_t alpha_matrix[B+2][B+2];
        convolution<type_t, B+2, B+2, F, HORIZONTAL>((type_t*)logs_matrix, omega, (type_t*)buffer_matrix);
        convolution<type_t, B+2, B+2, F, VERTICAL>((type_t*)buffer_matrix, omega_deriv, (type_t*)alpha_matrix);
        
        static type_t beta_matrix[B+2][B+2];
        convolution<type_t, B+2, B+2, F, VERTICAL>((type_t*)pjk_over_pk, omega_deriv_k, (type_t*)beta_matrix);

        range(i, 0, N) {
            int m_idx = I_m[i]+1,
                f_idx = I_f[i]+1;

            mi_deriv[i] = beta_matrix[m_idx][f_idx] - bigc - alpha_matrix[m_idx][f_idx];
        }

    #else
    // computes derivative values only for actual pixels of the image.
    // for every pair of pixels computes a single step of the convolution above.
    // probably optimizable with cache

    int pad = F/2;

    range(i, 0, N) {
        int m_idx = I_m[i],
            f_idx = I_f[i];

        type_t alpha = 0;
        range(a, -pad, pad+1) range(b, -pad, pad+1)
                if (m_idx+a>0 && m_idx+a<B && f_idx+b>0 && f_idx+b<B)
                    alpha += logs_matrix[m_idx+a][f_idx+b] * omega[b+pad] * omega_deriv[a+pad];

        type_t beta = 0;
        range(a, -pad, pad+1)
            if (m_idx+a>0 && m_idx+a<B)
                beta += pjk_over_pk[m_idx+a][f_idx] * omega_deriv_k[a+pad];

        mi_deriv[i] = beta - bigc - alpha;
    }

    #endif
}