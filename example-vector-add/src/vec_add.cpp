#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
using namespace adf;

void vec_add(
    input_buffer<int32>& __restrict data1,
    input_buffer<int32>& __restrict data2,
    output_buffer<int32>& __restrict out
) {
    // The SIMD instructions can process 16 int32 per cycle (512b registers)
    auto inIter1 = aie::begin_vector<16>(data1);
    auto inIter2 = aie::begin_vector<16>(data2);
    auto outIter = aie::begin_vector<16>(out);

    for (unsigned i = 0; i < data1.size() / 16; i++)
    {
        aie::vector<int32, 16> vec1 = *inIter1;
        aie::vector<int32, 16> vec2 = *inIter2;
        aie::vector<int32, 16> res = aie::add(vec1, vec2);
        *outIter = res;
		
		//Increment indices
		inIter1++;
		inIter2++;
		outIter++;
    }
}

