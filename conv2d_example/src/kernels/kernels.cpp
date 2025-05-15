#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "kernels.h"

// Tile dimensions with 1-pixel overlap for 3x3 convolution
constexpr int IN_TILE_WIDTH = 130;  // 128 output + 2 overlap
constexpr int IN_TILE_HEIGHT = 130;
constexpr int OUT_TILE_WIDTH = 128;
constexpr int OUT_TILE_HEIGHT = 128;
constexpr int VEC = 64; // 64 int8 per 512b vector (max for AIE1)

// Gaussian blur coefficients (int8 scaled by 1/16)
const int8 coeffs[9] = {1, 2, 1, 
                        2, 4, 2, 
                        1, 2, 1}; 

void conv3x3(input_buffer<int8>& in, output_buffer<int8>& out) {
    auto in_iter = aie::begin_vector<VEC>(in);
    auto out_iter = aie::begin_vector<VEC>(out);
    
    aie::vector<int8, VEC> window[3];
    aie::accum<acc48, VEC> acc;

    // Load initial rows
    window[0] = *in_iter++;
    window[1] = *in_iter++;
    window[2] = *in_iter++;

    for(int row = 0; row < OUT_TILE_HEIGHT; row++) {
        for(int col = 0; col < OUT_TILE_WIDTH/VEC; col++) {
            // Compute 3x3 convolution using vector intrinsics
            acc = aie::sliding_mul_xy_3x3(
                window[0], window[1], window[2],
                coeffs[0], coeffs[1], coeffs[2],
                coeffs[3], coeffs[4], coeffs[5],
                coeffs[6], coeffs[7], coeffs[8]
            );
            
            // Normalize and clamp to int8 (divide by 16)
            *out_iter++ = acc.template to_vector<int8>(4); // Shift right 4 bits

            // Slide window right by 1 pixel (64 elements = 64 pixels)
            window[0] = window[0].shift_right(1);
            window[1] = window[1].shift_right(1);
            window[2] = window[2].shift_right(1);
            
            // Insert new elements from next column
            if(col < OUT_TILE_WIDTH/VEC - 1) {
                window[0].insert(0, window[0].extract<1>());
                window[1].insert(0, window[1].extract<1>());
                window[2].insert(0, window[2].extract<1>());
            }
        }
        
        // Load new row
        if(row < OUT_TILE_HEIGHT - 1) {
            window[0] = window[1];
            window[1] = window[2];
            window[2] = *in_iter++;
        }
    }
}

