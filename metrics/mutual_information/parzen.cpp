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
#include "parzen.hpp"
#include "utils.hpp"
#include "hls_math.h"
#include "histogram.h"
#include "entropy.h"
#include "convolution.h"

#include <iostream>
#include <fstream>


void compute(INPUT_DATA_TYPE* input_img, INPUT_DATA_TYPE* input_ref, data_t *result){

#ifndef CACHING
	#pragma HLS INLINE
#endif

#pragma HLS DATAFLOW

	static	hls::stream<INPUT_DATA_TYPE> ref_stream("ref_stream");
	#pragma HLS STREAM variable=ref_stream depth=2 dim=1
	static	hls::stream<INPUT_DATA_TYPE> flt_stream("flt_stream");
	#pragma HLS STREAM variable=flt_stream depth=2 dim=1

	// Step 1: read data from DDR and split them
	axi2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(flt_stream, input_img);
#ifndef CACHING
	axi2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(ref_stream, input_ref);
#else
	bram2stream<INPUT_DATA_TYPE, NUM_INPUT_DATA>(ref_stream, input_ref);
#endif

	static  hls::stream<UNPACK_DATA_TYPE> ref_pe_stream[HIST_PE];
	#pragma HLS STREAM variable=ref_pe_stream depth=2 dim=1
	static  hls::stream<UNPACK_DATA_TYPE> flt_pe_stream[HIST_PE];
	#pragma HLS STREAM variable=flt_pe_stream depth=2 dim=1

	split_stream<INPUT_DATA_TYPE, UNPACK_DATA_TYPE, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA, HIST_PE>(ref_stream, ref_pe_stream);
	split_stream<INPUT_DATA_TYPE, UNPACK_DATA_TYPE, UNPACK_DATA_BITWIDTH, NUM_INPUT_DATA, HIST_PE>(flt_stream, flt_pe_stream);
	// End Step 1


	// Step 2: Compute two histograms in parallel
	static	hls::stream<PACKED_HIST_PE_DATA_TYPE> j_h_pe_stream[HIST_PE];
	#pragma HLS STREAM variable=j_h_pe_stream depth=2 dim=1

	WRAPPER_HIST(HIST_PE)<UNPACK_DATA_TYPE, NUM_INPUT_DATA, HIST_PE_TYPE, PACKED_HIST_PE_DATA_TYPE, MIN_HIST_PE_BITS, PADDING>(ref_pe_stream, flt_pe_stream, j_h_pe_stream);

	static	hls::stream<PACKED_HIST_DATA_TYPE> joint_j_h_stream("joint_j_h_stream");
	#pragma HLS STREAM variable=joint_j_h_stream depth=2 dim=1

	sum_joint_histogram<PACKED_HIST_PE_DATA_TYPE, J_HISTO_ROWS*J_HISTO_COLS/ENTROPY_PE, PACKED_HIST_DATA_TYPE, HIST_PE, HIST_PE_TYPE, MIN_HIST_PE_BITS, HIST_TYPE, MIN_HIST_BITS>(j_h_pe_stream, joint_j_h_stream);
	// End Step 2

	static	hls::stream<COMPUTATION_TYPE> h_conv_stream("h_conv_stream");
	#pragma HLS STREAM variable=h_conv_stream depth=2 dim=1
	static	hls::stream<COMPUTATION_TYPE> v_conv_stream("v_conv_stream");
	#pragma HLS STREAM variable=v_conv_stream depth=2 dim=1

	horizontal_convolution<PACKED_HIST_DATA_TYPE, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE, KERNEL_SIZE>(joint_j_h_stream, h_conv_stream);
	vertical_convolution<COMPUTATION_TYPE, COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS/ENTROPY_PE, KERNEL_SIZE>(h_conv_stream, v_conv_stream);


	// Step 3: Compute histograms per row and column
	static	hls::stream<COMPUTATION_TYPE> joint_j_h_stream_0("joint_j_h_stream_0");
	#pragma HLS STREAM variable=joint_j_h_stream_0 depth=2 dim=1
	static	hls::stream<COMPUTATION_TYPE> joint_j_h_stream_1("joint_j_h_stream_1");
	#pragma HLS STREAM variable=joint_j_h_stream_1 depth=2 dim=1
	static	hls::stream<COMPUTATION_TYPE> joint_j_h_stream_2("joint_j_h_stream_2");
	#pragma HLS STREAM variable=joint_j_h_stream_2 depth=2 dim=1

	tri_stream<COMPUTATION_TYPE, J_HISTO_ROWS*J_HISTO_COLS>(v_conv_stream, joint_j_h_stream_0, joint_j_h_stream_1, joint_j_h_stream_2);

	static	hls::stream<COMPUTATION_TYPE> row_hist_stream("row_hist_stream");
	#pragma HLS STREAM variable=row_hist_stream depth=dim_row dim=1
	static	hls::stream<COMPUTATION_TYPE> col_hist_stream("col_hist_stream");
	#pragma HLS STREAM variable=col_hist_stream depth=2 dim=1

	hist_row_simple<COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS>(joint_j_h_stream_0, row_hist_stream);
	hist_col<COMPUTATION_TYPE, J_HISTO_ROWS, J_HISTO_COLS>(joint_j_h_stream_1, col_hist_stream);


	static	hls::stream<COMPUTATION_TYPE> full_entropy_stream("full_entropy_stream");
	#pragma HLS STREAM variable=full_entropy_stream depth=2 dim=1
	static	hls::stream<COMPUTATION_TYPE> row_entropy_stream("row_entropy_stream");
	#pragma HLS STREAM variable=row_entropy_stream depth=2 dim=1
	static	hls::stream<COMPUTATION_TYPE> col_entropy_stream("col_entropy_stream");
	#pragma HLS STREAM variable=col_entropy_stream depth=2 dim=1

	compute_entropy<COMPUTATION_TYPE, COMPUTATION_TYPE, J_HISTO_ROWS, 0>(row_hist_stream, row_entropy_stream);
	compute_entropy<COMPUTATION_TYPE, COMPUTATION_TYPE, J_HISTO_COLS, 1>(col_hist_stream, col_entropy_stream);
	compute_entropy<COMPUTATION_TYPE, COMPUTATION_TYPE, J_HISTO_ROWS*J_HISTO_COLS, 2>(joint_j_h_stream_2, full_entropy_stream);
	// End Step 3

	static	hls::stream<data_t> mutual_information_stream("mutual_information_stream");
	#pragma HLS STREAM variable=mutual_information_stream depth=2 dim=1
	
	compute_mutual_information<COMPUTATION_TYPE, data_t>(row_entropy_stream, col_entropy_stream, full_entropy_stream, mutual_information_stream);

	stream2axi<data_t>(result, mutual_information_stream);
}


#ifdef KERNEL_NAME
extern "C"{
	void KERNEL_NAME
#else
	void parzen_master
#endif //KERNEL_NAME
	(INPUT_DATA_TYPE* input_img, INPUT_DATA_TYPE* input_ref, data_t *result){
	#pragma HLS INTERFACE m_axi port=input_img depth=fifo_in_depth offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=input_ref depth=fifo_in_depth offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=result depth=fifo_out_depth offset=slave bundle=gmem2

	#pragma HLS INTERFACE s_axilite port=input_img bundle=control
	#pragma HLS INTERFACE s_axilite port=input_ref bundle=control
	#pragma HLS INTERFACE s_axilite port=result register bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	compute(input_img, input_ref, result);

}


#ifdef KERNEL_NAME

} // extern "C"

#endif //KERNEL_NAME