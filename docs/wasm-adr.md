# ADR: WebAssembly + TensorFlow.js Architecture for spleeter++

- Status: proposed
- Date: 2025-09-13

## Context

The native project uses TensorFlow C API for inference and file-based I/O. For a browser implementation we must:

- Run core logic in WebAssembly (WASM) compiled with Emscripten.
- Replace TensorFlow C API with TensorFlow.js (WebGL preferred, CPU fallback).
- Handle audio via Web APIs (decode, resample to 44.1kHz, chunk, reconstruct, encode).

## Decision

Split responsibilities:

- C++/WASM: lightweight audio buffer handling utilities needed from existing code (where applicable), stateful processing window bookkeeping, normalization helpers, simple progress counters. Expose a minimal C ABI for init/process/finalize; avoid linking TF in WASM.
- JavaScript: decoding, 44.1kHz resampling, chunk planning, TensorFlow.js inference, overlap-add reconstruction, WAV encoding, UI, workers, telemetry.

ABI (initial):

- `int sp_init(uint32_t sample_rate, uint32_t channels, uint32_t chunk_size, uint32_t hop_size)`
- `uint8_t* sp_alloc(uint32_t byte_length)` returns pointer into WASM heap for zero-copy transfer.
- `int sp_process_chunk(float* interleaved, uint32_t frames)` updates internal progress; no ML inside.
- `int sp_finalize()` resets internal state.
- `const char* sp_version()` returns semantic version string.

Error handling: integer codes (0 OK, non-zero error). Progress is polled via a shared counter or emitted via JS wrapper.

Memory: `ALLOW_MEMORY_GROWTH=1`, start at 512MB. Prefer transferring ArrayBuffers to worker and writing directly into WASM heap via `HEAPF32` views.

TFJS: Default to `webgl` backend, fallback to `cpu`. Disable TFJS wasm backend.

## Consequences

- No TensorFlow linkage inside WASM; easier, smaller module, fewer syscalls.
- JS hosts inference; easier GPU utilization but requires careful memory and threading (Worker).
- Chunking and overlap logic must be tested for parity to native.

## Alternatives considered

- Port TF C API to WASM: impractical due to heavy deps and lack of GPU.
- Use TFJS WASM backend: duplicates our WASM and loses WebGL acceleration.

## Work Items

1. Toolchain + CMake target for WASM
2. C ABI bridge (`src/wasm/`)
3. JS wrapper + worker + audio pipeline
4. TFJS model conversion + loader
5. Demo UI + tests + CI
