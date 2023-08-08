#include "src/fastertransformer/th_op/whisper/FTWhisperConfig.h"
#include <cstdint>
namespace th = torch;

static th::jit::class_<torch_ext::FTWhisperConfig> ftWhisperConfigTh 
=   th::jit::class_<torch_ext::FTWhisperConfig>("FasterTransformer", "FTWhisperConfig")
    .def(torch::jit::init<
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t
    >())
;