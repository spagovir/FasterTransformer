#include <vector>
#include "src/fastertransformer/th_op/th_utils.h"
#include <iostream>
namespace th = torch;
namespace torch_ext{
template <typename T> 
struct VectorReader {
    uint32_t idx;
    std::vector<th::Tensor> *vec_;
    VectorReader(std::vector<th::Tensor> *vec): idx(0), vec_(vec) {};
    T* read() { th::Tensor t = vec_->at(idx++); return get_ptr<T>(t); }
};
}