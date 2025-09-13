#include "wasm/bridge.h"

#include <atomic>
#include <cstring>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define KEEPALIVE EMSCRIPTEN_KEEPALIVE
#else
#define KEEPALIVE
#endif

namespace {
struct State {
  uint32_t sr = 44100;
  uint32_t ch = 2;
  uint32_t chunk = 0;
  uint32_t hop = 0;
  std::atomic<uint64_t> processed_frames{0};
};

State& state() {
  static State s;
  return s;
}
}

extern "C" {

KEEPALIVE int sp_init(uint32_t sample_rate, uint32_t channels, uint32_t chunk_size, uint32_t hop_size) {
  state().sr = sample_rate;
  state().ch = channels;
  state().chunk = chunk_size;
  state().hop = hop_size;
  state().processed_frames.store(0);
  return 0;
}

KEEPALIVE unsigned char* sp_alloc(uint32_t byte_length) {
  // Emscripten ensures malloc-export on demand; we return a raw pointer for JS to fill.
  return reinterpret_cast<unsigned char*>(malloc(byte_length));
}

KEEPALIVE int sp_process_chunk(float* interleaved, uint32_t frames) {
  // No ML here; just account for progress for now.
  (void)interleaved;
  state().processed_frames.fetch_add(frames);
  return 0;
}

KEEPALIVE int sp_finalize() {
  state().processed_frames.store(0);
  return 0;
}

KEEPALIVE const char* sp_version() {
  static std::string v = "spleeter-wasm-0.1.0";
  return v.c_str();
}

}
