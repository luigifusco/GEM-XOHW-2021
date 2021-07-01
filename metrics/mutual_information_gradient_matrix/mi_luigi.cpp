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

#define B 256
#define F 3
#define SMALLFLOAT 1.175494e-38

// IMPORTANT: used all over the place to reduce verbosity
#define range(I,S,N) for(int I=S;I<N;++I)

#define VERTICAL true
#define HORIZONTAL false

//#define PRECOMPUTE

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
    range(j, 0, H) range(i, 0, W) {
            float acc = 0;
            range(b, -pad, pad+1) {
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
    I_m: pointer to moving image data (array of N)
    I_f: pointer to fixed image data (array of N)
    N: size of input
    mi: pointer to mutual information value (output, single float)
    mi_deriv: pointer to mutual information derivatives (output, array of N floats)
*/

void mutual_information(unsigned char* I_m, unsigned char* I_f, int N, float* mi, float *mi_deriv) {
    int counting_matrix[B][B] = { 0 };
    float omega[F] = { 1./6., 2./3., 1./6. };
    float omega_deriv[F] = { 1./2., 0., -1./2. };
    float omega_deriv_k[F] = { 1./2., 0., -1./2. };


    range(i, 0, N)
        counting_matrix[I_m[i]][I_f[i]]++;
    
///////////////////////////////////////////////////////////////////////
    //1 Joint histogram + calcolo degli indici
///////////////////////////////////////////////////////////////////////

    float buffer_matrix[B][B]; // used only for partial calculation, probably skippable if output of convolution can be same vector as input
    float prob_matrix[B][B];
    convolution<int, B, B, F, VERTICAL>((int*)counting_matrix, omega, (float*)buffer_matrix);
    convolution<float, B, B, F, HORIZONTAL>((float*)buffer_matrix, omega, (float*)prob_matrix);

///////////////////////////////////////////////////////////////////////
    //2 convoluzioni (con padding prima)
///////////////////////////////////////////////////////////////////////

    range(j, 0, B) range(k, 0, B)
            prob_matrix[j][k] /= (float)N;
///////////////////////////////////////////////////////////////////////
    //3 scaling
///////////////////////////////////////////////////////////////////////

    float prob_j[B] = { 0 };
    range(j, 0, B) range(k, 0, B)
            prob_j[j] += prob_matrix[j][k];
//righe 
    float prob_k[B] = { 0 };
    range(j, 0, B) range(k, 0, B)
            prob_k[k] += prob_matrix[j][k];
//colonne
///////////////////////////////////////////////////////////////////////
    //4 histo row e histo col
///////////////////////////////////////////////////////////////////////

    float pjk_over_pk[B][B];
    range(j, 0, B) range(k, 0, B)
            pjk_over_pk[j][k] = prob_matrix[j][k] / prob_k[k];
///////////////////////////////////////////////////////////////////////
    //5 histo conj normalizzato divide per histo colonne
///////////////////////////////////////////////////////////////////////


    float logs_matrix[B][B];
    range(j, 0, B) range(k, 0, B) {
            float denom = prob_j[j] * prob_k[k];
            if (denom == 0)
                denom = SMALLFLOAT;

            float num = prob_matrix[j][k];
            if (num == 0)
                num = SMALLFLOAT;

            logs_matrix[j][k] = logf(num/denom);
        }

///////////////////////////////////////////////////////////////////////
    //6 creazione matrice logaritmi
///////////////////////////////////////////////////////////////////////

    float res = 0;
    range(j, 0, B) range(k, 0, B)
            res += prob_matrix[j][k] * logs_matrix[j][k]; 

    *mi = -res;

    float bigc = 0;
    range(i, 0, F) range(j, 0, F)
            bigc += omega[i] * omega_deriv[j];
    
    #ifdef PRECOMPUTE
    // precompute all possible derivative values through a full convolution

        float alpha_matrix[B][B];
        convolution<float, B, B, F, VERTICAL>((float*)logs_matrix, omega_deriv, (float*)buffer_matrix);
        convolution<float, B, B, F, HORIZONTAL>((float*)buffer_matrix, omega, (float*)alpha_matrix);
        
        float beta_matrix[B][B];
        convolution<float, B, B, F, VERTICAL>((float*)pjk_over_pk, omega_deriv_k, (float*)beta_matrix);

        range(i, 0, N) {
            int m_idx = I_m[i],
                f_idx = I_f[i];

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

        float alpha = 0;
        range(a, -pad, pad+1) range(b, -pad, pad+1)
                if (m_idx+a>0 && m_idx+a<B && f_idx+b>0 && f_idx+b<B)
                    alpha += logs_matrix[m_idx+a][f_idx+b] * omega[b+pad] * omega_deriv[a+pad];

        float beta = 0;
        range(a, -pad, pad+1)
            if (m_idx+a>0 && m_idx+a<B)
                beta += pjk_over_pk[m_idx+a][f_idx] * omega_deriv_k[a+pad];

        mi_deriv[i] = beta - bigc - alpha;
    }

    #endif
}

#define IMSIZE 10000

int main() {
    unsigned char moving[IMSIZE], fixed[IMSIZE];
    float mi, mi_deriv[IMSIZE];
    int in;

    freopen("test.dat", "r", stdin);

    range(i, 0, IMSIZE) {
        scanf("%d\n", &in);
        moving[i] = (unsigned char)in;
    }

    range(i, 0, IMSIZE) {
        scanf("%d\n", &in);
        fixed[i] = (unsigned char)in;
    }

    mutual_information(moving, fixed, IMSIZE, &mi, mi_deriv);

    printf("%f\n", mi);
    range(i, 0, 10) printf("%f ", mi_deriv[i]);
    putchar('\n');
}