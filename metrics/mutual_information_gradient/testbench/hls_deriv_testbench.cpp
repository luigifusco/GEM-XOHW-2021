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
#include <iostream>
#include <cmath>
#include <random>
#include <stdio.h>
#include "mutual_information_derived.hpp"
#include "mi_luigi.hpp"

typedef ap_uint<8> MY_PIXEL;
#define MAX_RANGE 255
#define PADDING 1

const data_t kernel_host[KERNEL_SIZE] = { 1./6., 2./3., 1./6. };

int main(){
   MY_PIXEL ref[DIMENSION * DIMENSION];
   MY_PIXEL flt[DIMENSION * DIMENSION];

   data_t estimators[J_HISTO_ROWS+PADDING*2][J_HISTO_ROWS+PADDING*2];
   data_t partial_k_estimators[J_HISTO_COLS];
   data_t partial_i_estimators[J_HISTO_ROWS];
   data_t cache[KERNEL_SIZE];

   float nmi_sw[DIMENSION*DIMENSION];
   data_t nmi_hw_0[1], nmi_hw_1[DIMENSION*DIMENSION], nmi_hw_2[DIMENSION*DIMENSION];

   int myseed = 1234;

   std::default_random_engine rng(myseed);
   std::uniform_int_distribution<unsigned int> rng_dist(0, MAX_RANGE);

   for(int i=0;i<DIMENSION*DIMENSION;i++){
      ref[i]= static_cast<unsigned char>(rng_dist(rng));
      flt[i]= static_cast<unsigned char>(rng_dist(rng));
   }

#ifdef CACHING
   int status = 0;
   printf("Loading images...\n");
   parzen_master((INPUT_DATA_TYPE*)ref, nmi_hw_0, 0, &status);
   printf("Status %d\n", status);
   parzen_master((INPUT_DATA_TYPE*)flt, nmi_hw_0, 1, &status);
   printf("Status %d\n", status);
#endif

   mutual_information<256>((unsigned char *)flt, (unsigned char *)ref, DIMENSION*DIMENSION, NULL, nmi_sw);
   printf("Software NMI: ");
   for (int i = 0; i < 20; ++i) printf("%f ", nmi_sw[i]);
   printf("\n");


#ifndef CACHING
   parzen_master((INPUT_DATA_TYPE*)flt, (INPUT_DATA_TYPE*)ref, (INPUT_DATA_TYPE*)flt, (INPUT_DATA_TYPE*)ref, nmi_hw_0);
   printf("First Hardware NMI: ");
   for (int i = 0; i < 20; ++i) printf("%f ", nmi_hw_0[i]);
   printf("\n");

   parzen_master((INPUT_DATA_TYPE*)flt, (INPUT_DATA_TYPE*)ref, (INPUT_DATA_TYPE*)flt, (INPUT_DATA_TYPE*)ref, nmi_hw_1);
   printf("Second Hardware NMI: ");
   for (int i = 0; i < 20; ++i) printf("%f ", nmi_hw_1[i]);
   printf("\n");

   for (int i = 0; i < 20; ++i) printf("%d %d | ", (int)flt[i], (int)ref[i]);
   printf("\n");
#else
   parzen_master(NULL, nmi_hw_0, 2, &status);
   printf("First Hardware NMI: ");
   for (int i = 0; i < 20; ++i) printf("%f ", nmi_hw_0[i]);
      printf("\n");
   printf("Status %d\n", status);

   parzen_master(NULL, nmi_hw_1, 2, &status);
   printf("Second Hardware NMI: ");
   for (int i = 0; i < 20; ++i) printf("%f ", nmi_hw_1[i]);
      printf("\n");
   printf("Status %d\n", status);

   parzen_master(NULL, nmi_hw_2, 3, &status);
   printf("Status %d\n", status);
#endif

   return 0;
}
