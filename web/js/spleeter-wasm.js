// Minimal loader for the Emscripten ES6 module output (spleeter.js)
// Usage:
//   const api = await createSpleeterWasm({ moduleUrl: '/build-wasm/src/wasm/spleeter.js' });
//   console.log(await api.version());

export async function createSpleeterWasm({ moduleUrl } = {}) {
  if (!moduleUrl) throw new Error("moduleUrl is required");
  const factory = (await import(moduleUrl)).default;
  const Module = await factory();

  const sp_version = Module.cwrap("sp_version", "string", []);
  const sp_init = Module.cwrap("sp_init", "number", [
    "number",
    "number",
    "number",
    "number",
  ]);
  const sp_alloc = Module.cwrap("sp_alloc", "number", ["number"]);
  const sp_process_chunk = Module.cwrap("sp_process_chunk", "number", [
    "number",
    "number",
  ]);
  const sp_finalize = Module.cwrap("sp_finalize", "number", []);

  return {
    module: Module,
    version: async () => sp_version(),
    init: async (opts) => {
      const {
        sampleRate = 44100,
        channels = 2,
        chunkSize = 0,
        hopSize = 0,
      } = opts || {};
      return sp_init(sampleRate, channels, chunkSize, hopSize);
    },
    alloc: (byteLength) => sp_alloc(byteLength >>> 0),
    processChunk: (ptr, frames) => sp_process_chunk(ptr >>> 0, frames >>> 0),
    finalize: () => sp_finalize(),
    heaps: {
      u8: () => Module.HEAPU8,
      f32: () => Module.HEAPF32,
    },
  };
}
