#include "src/fastertransformer/models/whisper/WhisperEncoderLayerWeight.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/DenseWeight.h"
using namespace std;

namespace fastertransformer{
template <typename T>
struct WhisperEncoderWeight {
    DenseWeight<T> conv1; 
    DenseWeight<T> conv2;
    vector<WhisperEncoderLayerWeight<T>> layers;
    LayerNormWeight<T> layernorm;
};}