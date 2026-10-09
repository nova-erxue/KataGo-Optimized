#ifndef NEURALNET_CUDAARCHITECTURE_H_
#define NEURALNET_CUDAARCHITECTURE_H_

namespace CudaArchitecture {

// Hardware capability defines support; product branding is intentionally ignored.
inline bool isSM120(int major, int minor) {
  return major == 12 && minor == 0;
}

}
#endif
