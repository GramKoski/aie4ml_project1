#include <adf.h>
#include "kernels.h"

using namespace adf;

class ConvGraph : public graph {
public:
    static constexpr int NUM_TILES = 16;

    kernel conv_kernels[NUM_TILES];
    input_plio in_plio[NUM_TILES];
    output_plio out_plio[NUM_TILES];

    ConvGraph() {
        for (int t = 0; t < NUM_TILES; ++t) {
            // PLIOs for simulation file I/O
            in_plio[t] = input_plio::create(
                "in" + std::to_string(t), plio_128_bits,
                "data/input_tile" + std::to_string(t) + ".txt");
            out_plio[t] = output_plio::create(
                "out" + std::to_string(t), plio_128_bits,
                "data/output_tile" + std::to_string(t) + ".txt");

            // Kernel
            conv_kernels[t] = kernel::create(conv3x3);
            source(conv_kernels[t]) = "kernels/kernels.cpp";
            runtime<ratio>(conv_kernels[t]) = 0.9;

            // Connect input/output buffers
            connect<buffer>(in_plio[t].out[0], conv_kernels[t].in[0]);
            connect<buffer>(conv_kernels[t].out[0], out_plio[t].in[0]);
        }
    }
};


