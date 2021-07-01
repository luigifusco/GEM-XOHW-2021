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


extern "C" {
    void rotate_shift_transform_derivatives(int shape_y, int shape_x, double *gradient_x, double *gradient_y, double theta, double alpha, double (*grads)[3]);
}


void rotate_shift_transform_derivatives(int shape_y, int shape_x, double *gradient_x, double *gradient_y, double theta, double alpha, double (*grads)[3]) {
    for (int y = 0; y < shape_y; ++y) {
        for (int x = 0; x < shape_x; ++x) {
            int index = y*shape_x+x;
            double i = gradient_x[index];
            double j = gradient_y[index];
            grads[index][0] = alpha*(i*(- x*sin(theta) - y*cos(theta)) + j*(x*cos(theta) - y*sin(theta)));
            grads[index][1] = j;
            grads[index][2] = i;
        }
    }
}