add_library(th_whisper STATIC FTWhisperConfig.cc FTWhisperEncoder.cc FTWhisperDecoder.cc)
set_property(TARGET th_whisper PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(th_whisper PRIVATE "${TORCH_LIBRARIES}" WhisperEncoder WhisperDecoder th_utils)
