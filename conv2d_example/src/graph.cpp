#include <adf.h>
#include "kernels.h"
#include "graph.h"

using namespace adf;

ConvGraph c_graph;

int main(void) {
  c_graph.init();
  c_graph.run(1);
  c_graph.end();
  return 0;
}
