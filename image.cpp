// SPDX-License-Identifier: MPL-2.0
/**
 * @file image.cpp
 * @brief Elementary image class
 * @author Robin Gay
 *         Pascal Monasse <pascal.monasse@enpc.fr>
 * @date 2021-2023, 2026
 */

#include "image.h"
#include <cmath>
#include <cassert>

/// Return Gaussian kernel of size 2*radius+1. \a radius is ceil(3*sigma).
float* gaussKernel(float sigma, int& radius) {
    radius = (int)std::ceil(3*sigma);
    int sz = 2*radius+1;
    float sum=0, *ker = new float[sz];
    for(int i=0; i<=radius; i++) {
        ker[sz-i-1] = ker[i] = std::exp(-(i-radius)*(i-radius)/(2*sigma*sigma));
    }
    for(int i=0; i<sz; i++)
        sum += ker[i];
    for(int i=0; i<sz; i++)
        ker[i] /= sum;
    return ker;
}

/// x-convolution, the result is transposed.
void convx(const Image<float>& im, const float* ker, int radius,
           Image<float>& out) {
    assert(radius<=2*im.w);
    out.reset(im.h, im.w);
    for(int y=0; y<im.h; y++) {
        const float* in=im.data+y*im.w;
        for(int x=0; x<im.w; x++) {
            float c=0;
            const float* k = ker;
            for(int i=-radius; i<=+radius; i++) {
                int j = x+i;
                if(j<0) j = -j-1;
                if(j>=im.w) j = 2*im.w-j-1;
                c += in[j] * *k++;
            }
            out(y,x) = c;
        }
    }
}

/// Generate Gaussian pyramid. \a n is the number of scales. The first scale
/// is always the initial image.
void blurGaussian(Image<float>& in, float sigma) {
    if(sigma==0) return;
    int radius;
    float* ker = gaussKernel(sigma, radius);
    Image<float> xconv;
    convx(in, ker, radius, xconv);
    convx(xconv, ker, radius, in);
    delete [] ker;
}
