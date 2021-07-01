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
#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "mutual_information_derived.hpp"
#include "hls_stream.h"
#include <iostream>
#include <fstream>

template<typename Talpha, typename Tbeta, typename Tout, unsigned int rows, unsigned int cols>
#ifndef ALPHA_ONLY
	#ifndef DERIV_MATRIX
	void compute_gradient(hls::stream<Talpha> &alpha_matrix, hls::stream<Tbeta> &beta_matrix, Tout gradient_matrix[rows][cols]) {
	#else
	void compute_gradient(hls::stream<Talpha> &alpha_matrix, hls::stream<Tbeta> &beta_matrix, Tout *gradient_matrix) {
	#endif
#else
	#ifndef DERIV_MATRIX
	void compute_gradient(hls::stream<Talpha> &alpha_matrix, Tout gradient_matrix[rows][cols]) {
	#else
	void compute_gradient(hls::stream<Talpha> &alpha_matrix, Tout *gradient_matrix) {
	#endif
#endif
	//std::ofstream file;
	//file.open("hw.txt");

	const Tout alpha_factor = 1./12.;
	const Tout beta_factor = 1./(2.*512.*512.*36.);

	for (int j = 0; j < rows; ++j) {
		for (int k = 0; k < cols; ++k) {
#pragma HLS PIPELINE
			Tout curr_alpha = (Tout) alpha_matrix.read();
			curr_alpha *= alpha_factor;
			#ifndef ALPHA_ONLY
				Tout curr_beta = beta_matrix.read();
				curr_beta *= beta_factor;
				#ifndef DERIV_MATRIX
				gradient_matrix[(int)j][(int)k] = curr_beta - curr_alpha;
				#else
				gradient_matrix[((int)j)*cols+(int)k] = curr_beta - curr_alpha;
				#endif
			#else
				#ifndef DERIV_MATRIX
				gradient_matrix[(int)j][(int)k] = - curr_alpha;
				#else
				gradient_matrix[((int)j)*cols+(int)k] = - curr_alpha;
				#endif
			#endif
			//if (j > 0 && k > 0 && j < rows-1 && k < cols-1) file << curr_beta - curr_alpha << '\t' << curr_beta << '\t' << curr_alpha << std::endl;
		}
	}

	//file.close();
}

template<typename Timage, typename Tgrad, unsigned int rows, unsigned int cols, unsigned int packed_bitwidth, unsigned int pixel_bitwidth, unsigned int size>
void populate_gradient_matrix(Tgrad gradient_matrix[rows][cols], Timage ref_img[size], Timage mov_img[size], Tgrad *result) {
	const unsigned int ratio = packed_bitwidth/pixel_bitwidth;
	for (int i = 0; i < size; ++i) {
		#pragma HLS PIPELINE
		for (int j = 0; j < ratio; ++j) {
			ap_uint<pixel_bitwidth> ref_id = ref_img[i].range((j+1)*pixel_bitwidth - 1, j*pixel_bitwidth);
			ap_uint<pixel_bitwidth> mov_id = mov_img[i].range((j+1)*pixel_bitwidth - 1, j*pixel_bitwidth);
			Tgrad curr_grad = gradient_matrix[ref_id+1][mov_id+1];
			result[i*ratio+j] = curr_grad;
		}
	}
}

#endif // GRADIENT_HPP
