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

#include "mutual_information_derived.hpp"

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

    #ifndef PADDED
    const int SIZE = B;
    #else
    const int SIZE = B+2;
    #endif

    int counting_matrix[SIZE][SIZE] = { 0 }; 
    static type_t buffer_matrix[SIZE][SIZE]; // used only for partial calculation, probably skippable if output of convolution can be same vector as input
    static type_t prob_matrix[SIZE][SIZE];
    type_t omega[F] = { 1./6., 4./6., 1./6. };
    type_t omega_deriv[F] = { -1./2., 0., 1./2. };
    type_t omega_deriv_k[F] = { -1./2., 0., 1./2. };


    range(i, 0, N)
    #ifndef PADDED
        counting_matrix[I_m[i]][I_f[i]]++;
    #else
        counting_matrix[I_m[i]+1][I_f[i]+1]++;
    #endif

    convolution<int, SIZE, SIZE, F, VERTICAL>((int*)counting_matrix, omega, (type_t*)buffer_matrix);
    convolution<type_t, SIZE, SIZE, F, HORIZONTAL>((type_t*)buffer_matrix, omega, (type_t*)prob_matrix);

    range(j, 0, SIZE) range(k, 0, SIZE)
        prob_matrix[j][k] /= (type_t)N;    

    type_t prob_j[SIZE] = { 0 };
    range(j, 0, SIZE) range(k, 0, SIZE)
            prob_j[j] += prob_matrix[j][k];


    type_t prob_k[SIZE] = { 0 };
    range(j, 0, SIZE) range(k, 0, SIZE)
            prob_k[k] += prob_matrix[j][k];

    type_t pjk_over_pk[SIZE][SIZE];
    range(j, 0, SIZE) range(k, 0, SIZE)
            pjk_over_pk[j][k] = prob_matrix[j][k] / prob_k[k];

    type_t logs_matrix[SIZE][SIZE];
    range(j, 0, SIZE) range(k, 0, SIZE) {
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
    range(j, 0, SIZE) range(k, 0, SIZE)
            res += prob_matrix[j][k] * logs_matrix[j][k]; 

    if (mi) *mi = -res;

    if (mi_deriv == NULL) return;

    type_t bigc = 0;
    range(i, 0, F) range(j, 0, F)
            bigc += omega[i] * omega_deriv[j];
    
    #ifdef PRECOMPUTE
    // precompute all possible derivative values through a full convolution

        static type_t alpha_matrix[SIZE][SIZE];
        convolution<type_t, SIZE, SIZE, F, HORIZONTAL>((type_t*)logs_matrix, omega, (type_t*)buffer_matrix);
        convolution<type_t, SIZE, SIZE, F, VERTICAL>((type_t*)buffer_matrix, omega_deriv, (type_t*)alpha_matrix);
        
        static type_t beta_matrix[SIZE][SIZE];
        convolution<type_t, SIZE, SIZE, F, VERTICAL>((type_t*)pjk_over_pk, omega_deriv_k, (type_t*)beta_matrix);

        /*std::ofstream file;
        file.open("sw.txt");
        for (int i = 1; i < B+1; ++i)
            for (int j = 1; j < B+1; ++j)
                file <<  beta_matrix[i][j] - alpha_matrix[i][j] << '\t' << beta_matrix[i][j] << '\t' << alpha_matrix[i][j] << std::endl;
        file.close();*/

        #ifndef DERIV_MATRIX

        range(i, 0, N) {
            int m_idx = I_m[i]+1,
                f_idx = I_f[i]+1;

            mi_deriv[i] = beta_matrix[m_idx][f_idx] - bigc - alpha_matrix[m_idx][f_idx];
        }

        #else

        range(i, 0, SIZE)
            range(j, 0, SIZE)
                mi_deriv[i*(SIZE)+j] = beta_matrix[i][j] - bigc - alpha_matrix[i][j];

        #endif

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
