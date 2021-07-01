#include <iostream>
#include <cmath>
#include <random>
#include <stdio.h>
#include "parzen.hpp"
#include "mi_luigi.hpp"
#include <time.h>

typedef ap_uint<8> MY_PIXEL;
#define MAX_RANGE 255
#define PADDING 1

const double kernel_host[KERNEL_SIZE] = { 1./6., 2./3., 1./6. };

int main(){


   MY_PIXEL ref[DIMENSION * DIMENSION];
   MY_PIXEL flt[DIMENSION * DIMENSION];

   double estimators[J_HISTO_ROWS+PADDING*2][J_HISTO_ROWS+PADDING*2];
   double partial_k_estimators[J_HISTO_COLS];
   double partial_i_estimators[J_HISTO_ROWS];
   double cache[KERNEL_SIZE];

   double partial_estimator_nmi, full_estimator_nmi, nmi_sw;
   data_t nmi_hw_0, nmi_hw_1, nmi_hw_2;

   int myseed = 1234;

   std::default_random_engine rng(myseed);
   std::uniform_int_distribution<unsigned int> rng_dist(0, MAX_RANGE);

   for(int i=0;i<DIMENSION*DIMENSION;i++){
      ref[i]= static_cast<unsigned char>(rng_dist(rng));
      flt[i]= static_cast<unsigned char>(rng_dist(rng));
   }

#ifdef CACHING
   int status = 0;
   printf("Loading image...\n");
   parzen_master((INPUT_DATA_TYPE*)ref, &nmi_hw_0, 0, &status);
   printf("Status %d\n", status);
#endif

   for (int row = 0; row < J_HISTO_ROWS+2*PADDING; ++row) {
      for (int col = 0; col < J_HISTO_ROWS+2*PADDING; ++col) {
         estimators[row][col] = 0;
      }
   }

   // counts the number of occurrence of intensity pairs in the input images
   estimators:for(int i = 0; i < DIMENSION * DIMENSION; ++i) {
      estimators[ref[i]+PADDING][flt[i]+PADDING]++;
   }

   printf("estimated: %f\n", (estimators[3][82]));

   // horizontal pass
   hconv:for (int row = PADDING; row < J_HISTO_ROWS+2*PADDING; ++row) {
      for (int col = 0; col < J_HISTO_ROWS+2*PADDING; ++col) {
         double in_val = estimators[row][col];
         double out_val = 0;
         for (int i = 0; i < KERNEL_SIZE; ++i) {
            cache[i] = i == KERNEL_SIZE - 1 ? in_val : cache[i+1];		// left shift of elements in cache, inserting new value from estimators
            out_val += cache[i] * (double)b_spline_kernel[i];							// performing actual convolution
         }
         if (col >= KERNEL_SIZE - 1) {
            estimators[row][col-1] = out_val;		// if position is valid, overwrites value in estimator which is already in cache
         }
      }
   }

   // vertical pass
   vconv:for (int col = PADDING; col < J_HISTO_ROWS+PADDING; ++col) {
      for (int row = 0; row < J_HISTO_ROWS+2*PADDING; ++row) {
         double in_val = estimators[row][col];
         double out_val = 0;
         for (int i = 0; i < KERNEL_SIZE; ++i) {
            cache[i] = i == KERNEL_SIZE - 1 ? in_val : cache[i+1];
            out_val += cache[i] * (double)b_spline_kernel[i];
         }
         if (row >= KERNEL_SIZE - 1) {
            estimators[row-1][col] = out_val;
         }
      }
   }

   for (int i = 0; i < 100; ++i)
      printf("%f ", estimators[1][i+1]);

   normalization:for(int i = PADDING; i < J_HISTO_ROWS + PADDING; ++i) {
      for(int j = PADDING; j < J_HISTO_COLS + PADDING; ++j) {
         estimators[i][j] = estimators[i][j] / (((double)DIMENSION)*((double)DIMENSION));
      }
   }

   printf("\n");


   for (int i = 0; i < 100; ++i)
      printf("%f ", estimators[1][i+1]);

   printf("\n");
   

   partial_k:for(int i = 0; i < J_HISTO_ROWS; ++i) {
      for (int k = 0; k < J_HISTO_COLS; ++k) {
         partial_k_estimators[k] += estimators[i+1][k+1];
      }
   }


   partial_i:for (int k = 0; k < J_HISTO_ROWS; ++k) {
      for(int i = 0; i < J_HISTO_COLS; ++i) {
         partial_i_estimators[i] += estimators[i+1][k+1];
      }
   }


   partial_estimator_nmi = 0;
   partial_estimator_log:for(int i = 0; i < J_HISTO_ROWS; ++i) {
      double log0 = 0, log1 = 0;
      if (partial_i_estimators[i] > 0)
         log0 = partial_i_estimators[i] * std::log2(partial_i_estimators[i]);
      if(partial_k_estimators[i] > 0)
         log1 = partial_k_estimators[i] * std::log2(partial_k_estimators[i]);
      partial_estimator_nmi += log0 + log1;
   }


   partial_estimator_nmi *= -1;

   full_estimator_nmi = 0;
   sum_calculation:for(int i = 0; i < J_HISTO_ROWS; ++i){
      for(int k = 0; k < J_HISTO_COLS; ++k) {
         double prob_ik = estimators[i+1][k+1];
         if(prob_ik > 0) {
            double log_val = std::log2(prob_ik);
            double prod = prob_ik * log_val;
            full_estimator_nmi += prod;
         }
      }
   }

   nmi_sw = -(partial_estimator_nmi + full_estimator_nmi);
   printf("First Software NMI %lf\n",nmi_sw);

   mutual_information<256>((unsigned char *)flt, (unsigned char *)ref, DIMENSION*DIMENSION, &nmi_sw, NULL);
   printf("Second Software NMI %lf\n",nmi_sw);


#ifndef CACHING
   parzen_master((INPUT_DATA_TYPE*)flt, (INPUT_DATA_TYPE*)ref, &nmi_hw_0);

   printf("First Hardware NMI %f\n", nmi_hw_0);

   parzen_master((INPUT_DATA_TYPE*)flt, (INPUT_DATA_TYPE*)ref, &nmi_hw_1);
   printf("Second Hardware NMI %f\n", nmi_hw_1);
#else
   parzen_master((INPUT_DATA_TYPE*)flt, &nmi_hw_0, 1, &status);

   printf("First Hardware NMI %f\n", nmi_hw_0);
   printf("Status %d\n", status);

   parzen_master((INPUT_DATA_TYPE*)flt, &nmi_hw_1, 1, &status);
   printf("Second Hardware NMI %f\n", nmi_hw_1);
   printf("Status %d\n", status);

   parzen_master((INPUT_DATA_TYPE*)flt, &nmi_hw_2, 2, &status);
   printf("Status %d\n", status);
#endif

   if(((data_t)nmi_sw - nmi_hw_0 > 0.01) || ((data_t)nmi_sw - nmi_hw_1 > 0.01)){
       return 1;
   }

   return 0;
}
