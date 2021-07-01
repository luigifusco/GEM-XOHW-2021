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
#include "mutual_information_derived.hpp"
#include "utils.hpp"
#include "hls_math.h"
#include "histogram.h"
#include "entropy.h"
#include "convolution.h"
#include "gradient.h"

#include "ap_int.h"
#include "ap_fixed.h"


#ifndef DERIV_MATRIX
void compute(INPUT_DATA_TYPE* input_img, INPUT_DATA_TYPE* input_ref, data_t gradient_matrix[J_HISTO_ROWS][J_HISTO_COLS]){
#else
void compute(INPUT_DATA_TYPE* input_img, INPUT_DATA_TYPE* input_ref, data_t *gradient_matrix){
#endif

	const COMPUTATION_TYPE b_spline_kernel[KERNEL_SIZE] = { 1, 4, 1 };
	const COMPUTATION_TYPE omega_deriv[KERNEL_SIZE] = { -1, 0, 1 };
	const COMPUTATION_TYPE omega_deriv_k[KERNEL_SIZE] = { -1, 0, 1 };

	//COMPUTATION_TYPE debug[J_HISTO_ROWS][J_HISTO_COLS];


#ifndef CACHING
	#pragma HLS INLINE
#endif

#pragma HLS DATAFLOW

	static	hls::stream<INPUT_DATA_TYPE> ref_stream("ref_stream");
	#pragma HLS STREAM variable=ref_stream depth=2 dim=1
	static	hls::stream<INPUT_DATA_TYPE> flt_stream("flt_stream");
	#pragma HLS STREAM variable=flt_stream depth=2 dim=1

	// Step 1: read data from DDR and split them
#ifndef CACHING
	axi2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(ref_stream, input_ref);
	axi2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(flt_stream, input_img);
#else
	bram2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(ref_stream, input_ref);
	bram2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(flt_stream, input_img);
#endif

	static  hls::stream<UNPACK_DATA_TYPE> ref_pe_stream[HIST_PE];
	#pragma HLS STREAM variable=ref_pe_stream depth=2 dim=1
	static  hls::stream<UNPACK_DATA_TYPE> flt_pe_stream[HIST_PE];
	#pragma HLS STREAM variable=flt_pe_stream depth=2 dim=1

	split_stream<INPUT_DATA_TYPE, UNPACK_DATA_TYPE, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA, HIST_PE>(ref_stream, ref_pe_stream);
	split_stream<INPUT_DATA_TYPE, UNPACK_DATA_TYPE, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA, HIST_PE>(flt_stream, flt_pe_stream);
	// End Step 1


	// Step 2: Compute two histograms in parallel
	// Note that the histograms contain the padded version of the matrix
	static	hls::stream<PACKED_HIST_PE_DATA_TYPE> j_h_pe_stream[HIST_PE];
	#pragma HLS STREAM variable=j_h_pe_stream depth=2 dim=1

	WRAPPER_HIST(HIST_PE)<UNPACK_DATA_TYPE, NUM_INPUT_DATA, HIST_PE_TYPE, PACKED_HIST_PE_DATA_TYPE, MIN_HIST_PE_BITS>(ref_pe_stream, flt_pe_stream, j_h_pe_stream);

	static	hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream("joint_j_h_stream"); // max 512*512 (18 bits)
	#pragma HLS STREAM variable=joint_j_h_stream depth=2 dim=1

	sum_joint_histogram<PACKED_HIST_PE_DATA_TYPE, J_HISTO_ROWS*J_HISTO_COLS/ENTROPY_PE, PACKED_HIST_DATA_TYPE, HIST_PE, HIST_PE_TYPE, MIN_HIST_PE_BITS, HIST_TYPE, MIN_HIST_BITS>(j_h_pe_stream, joint_j_h_stream);
	// End Step 2



	// Step 3: Compute histograms per row and column
	static	hls::stream<uint_small> h_conv_stream("h_conv_stream"); // max 512*512*6 (21 bits)
	#pragma HLS STREAM variable=h_conv_stream depth=2 dim=1
	static	hls::stream<uint_small> v_conv_stream("v_conv_stream"); // max 512*512*36 (24 bits)
	#pragma HLS STREAM variable=v_conv_stream depth=2 dim=1

	horizontal_convolution<PACKED_HIST_DATA_TYPE, uint_small, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE, KERNEL_SIZE>(joint_j_h_stream, h_conv_stream, b_spline_kernel);
	vertical_convolution<uint_small, uint_small, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE, KERNEL_SIZE>(h_conv_stream, v_conv_stream, b_spline_kernel);

	static	hls::stream<uint_small> joint_j_h_stream_0("joint_j_h_stream_0");
	#pragma HLS STREAM variable=joint_j_h_stream_0 depth=2 dim=1
	static	hls::stream<uint_small> joint_j_h_stream_1("joint_j_h_stream_1");
	#pragma HLS STREAM variable=joint_j_h_stream_1 depth=2 dim=1
	static	hls::stream<uint_small> joint_j_h_stream_2("joint_j_h_stream_2");
	#pragma HLS STREAM variable=joint_j_h_stream_2 depth=big_q_depth dim=1

	tri_stream<uint_small, J_HISTO_ROWS*J_HISTO_COLS/ENTROPY_PE>(v_conv_stream, joint_j_h_stream_0, joint_j_h_stream_1, joint_j_h_stream_2);


	static	hls::stream<uint_small> row_hist_stream("row_hist_stream"); // prob_j, max 512*512*36 (24 bits)
	#pragma HLS STREAM variable=row_hist_stream depth=dim_row dim=1
	static	hls::stream<uint_small> col_hist_stream("col_hist_stream"); // prob_k, max 512*512*36 (24 bits)
	#pragma HLS STREAM variable=col_hist_stream depth=2 dim=1

	hist_row_simple<uint_small, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE>(joint_j_h_stream_0, row_hist_stream);
	hist_col<uint_small, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE>(joint_j_h_stream_1, col_hist_stream);
	// End Step 3
 

#ifndef ALPHA_ONLY
	// Step 4: Compute utility matrices (pjk over pk, logs matrix)
	/*
		pjk_over_pk: min = 1/(512*512*36), avg = 1/(512), max = 1
		solution is to multiply all results by (512*512*36), this way min = 1, avg = 512*36, max = (512*512*36)
		does it make sense?
	*/
	static	hls::stream<uint_small> pjk_over_pk("pjk_over_pk"); // "max" 512*512*36 (24 bits)
	#pragma HLS STREAM variable=pjk_over_pk depth=2 dim=1
#endif
	/*
		if logs argument is normalized: min = (1/(512*512))/(0.5*0.5) [-16 after log], avg = 1 [0 after log], max = (1/(512^2))/(1/(512^4)) [18 after log]
		if logs argument is not normalized: min = [-40], avg = [-23], max = [-5]
		issue on number of fractional bits (using integers is NOT ENOUGH)
	*/
	static	hls::stream<fixed_small> logs_matrix("logs_matrix"); // "max" NOTMUCH
	#pragma HLS STREAM variable=logs_matrix depth=2 dim=1

#ifndef ALPHA_ONLY
	compute_pjkovrpk_logsmatrix<uint_small, uint_small, fixed_small, J_HISTO_ROWS, J_HISTO_COLS>(joint_j_h_stream_2, row_hist_stream, col_hist_stream, pjk_over_pk, logs_matrix);
#else
	compute_logsmatrix<uint_small, fixed_small, J_HISTO_ROWS, J_HISTO_COLS>(joint_j_h_stream_2, row_hist_stream, col_hist_stream, logs_matrix);
#endif
	// End Step 4


	// Step 5: Compute alpha and beta streams
	static	hls::stream<fixed_big> partial_alpha("partial_alpha");
	#pragma HLS STREAM variable=partial_alpha depth=2 dim=1
	static	hls::stream<fixed_big> alpha_matrix("alpha_matrix"); // "max" +-NOTMUCH*6 (xx bits signed)

#ifndef ALPHA_ONLY
	#pragma HLS STREAM variable=alpha_matrix depth=20 dim=1
	static	hls::stream<my_int> beta_matrix("beta_matrix"); // "max" +-512*512*36 (25 bits) 
	#pragma HLS STREAM variable=beta_matrix depth=2 dim=1
#else
	#pragma HLS STREAM variable=alpha_matrix depth=2 dim=1
#endif

	horizontal_convolution<fixed_small, fixed_big, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS, KERNEL_SIZE>(logs_matrix, partial_alpha, b_spline_kernel);
	vertical_convolution<fixed_big, fixed_big, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS, KERNEL_SIZE>(partial_alpha, alpha_matrix, omega_deriv);
	// alpha matrix needs a scaling of 1/12 (the additive part is canceled as the kernel has sum zero, acting as a high pass filter)

#ifndef ALPHA_ONLY
	vertical_convolution<uint_small, my_int, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS, KERNEL_SIZE>(pjk_over_pk, beta_matrix, omega_deriv_k);
	// beta_matrix needs a scaling of 1/2
#endif
	// End Step 5

	// Step 6: compute final gradient stream matrix
#ifndef ALPHA_ONLY
	compute_gradient<fixed_big, my_int, data_t, J_HISTO_ROWS, J_HISTO_COLS>(alpha_matrix, beta_matrix, gradient_matrix);
#else
	compute_gradient<fixed_big, my_int, data_t, J_HISTO_ROWS, J_HISTO_COLS>(alpha_matrix, gradient_matrix);
#endif
	// End Step 6
}


#ifdef KERNEL_NAME
extern "C"{
	void KERNEL_NAME
#else
	void mutual_information_derived_master
#endif //KERNEL_NAME
#ifndef DERIV_MATRIX
	(INPUT_DATA_TYPE * input_mov, INPUT_DATA_TYPE * input_ref, INPUT_DATA_TYPE * second_mov, INPUT_DATA_TYPE * second_ref, data_t * result){
#pragma HLS INTERFACE m_axi port=second_mov depth=fifo_in_depth offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=second_ref depth=fifo_in_depth offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=second_mov bundle=control
#pragma HLS INTERFACE s_axilite port=second_ref bundle=control
#else
	(INPUT_DATA_TYPE * input_mov, INPUT_DATA_TYPE * input_ref, data_t * result){
#endif
#pragma HLS INTERFACE m_axi port=input_mov depth=fifo_in_depth offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=input_ref depth=fifo_in_depth offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=result depth=out_size offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=input_mov bundle=control
#pragma HLS INTERFACE s_axilite port=input_ref bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	#ifndef DERIV_MATRIX
	data_t gradient_matrix[J_HISTO_ROWS][J_HISTO_COLS];
	#endif

	#ifndef DERIV_MATRIX
	compute(input_mov, input_ref, gradient_matrix);
	populate_gradient_matrix<INPUT_DATA_TYPE, data_t, J_HISTO_ROWS, J_HISTO_COLS, INPUT_DATA_BITWIDTH, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA>(gradient_matrix, second_ref, second_mov, result);
	#else
	compute(input_mov, input_ref, result);
	#endif
}


#ifdef KERNEL_NAME

} // extern "C"

#endif //KERNEL_NAME