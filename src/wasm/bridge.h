#ifndef SPLEETER_WASM_BRIDGE_H_
#define SPLEETER_WASM_BRIDGE_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int sp_init(uint32_t sample_rate, uint32_t channels, uint32_t chunk_size, uint32_t hop_size);
unsigned char* sp_alloc(uint32_t byte_length);
int sp_process_chunk(float* interleaved, uint32_t frames);
int sp_finalize();
const char* sp_version();

#ifdef __cplusplus
}
#endif

#endif  // SPLEETER_WASM_BRIDGE_H_
