// TensorFlow.js backend configuration utilities
// Loads TF.js from CDN and sets backend to WebGL (fallback CPU)

export async function ensureTF() {
  if (globalThis.tf) return globalThis.tf;
  await import(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js"
  );
  if (!globalThis.tf) throw new Error("Failed to load TensorFlow.js");
  return globalThis.tf;
}

async function ensureWasmBackend() {
  const tf = await ensureTF();
  // Load tfjs-backend-wasm and set paths to CDN distribution
  const wasmMod = await import(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.18.0/dist/tf-backend-wasm.min.js"
  );
  try {
    // Ensure the wasm binaries are loaded from our same-origin path.
    // Prefer the stable API on tf.wasm; fall back to module export if present.
    if (tf && tf.wasm && typeof tf.wasm.setWasmPaths === "function") {
      tf.wasm.setWasmPaths("/tfjs-wasm/");
    } else if (wasmMod && typeof wasmMod.setWasmPaths === "function") {
      wasmMod.setWasmPaths("/tfjs-wasm/");
    }
  } catch (_) {}
  return tf;
}

async function ensureWebGPUBackend() {
  const tf = await ensureTF();
  await import(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.18.0/dist/tf-backend-webgpu.min.js"
  );
  return tf;
}

export async function configureBackend(preference = "auto") {
  const tf = await ensureTF();
  const pref = String(preference || "auto");
  const attempts = [];
  const trySet = async (name) => {
    try {
      if (name === "wasm") await ensureWasmBackend();
      await tf.setBackend(name);
      await tf.ready();
      const ok = tf.getBackend() === name;
      attempts.push({ name, ok });
      return ok;
    } catch (_) {
      attempts.push({ name, ok: false });
      return false;
    }
  };
  let backend = tf.getBackend() || "cpu";
  if (pref === "cpu") {
    await trySet("cpu");
    backend = tf.getBackend();
  } else if (pref === "webgl" || pref === "gpu") {
    if (!(await trySet("webgl"))) {
      await trySet("cpu");
    }
    backend = tf.getBackend();
  } else if (pref === "wasm") {
    if (!(await trySet("wasm"))) {
      await trySet("cpu");
    }
    backend = tf.getBackend();
  } else if (pref === "webgpu") {
    await ensureWebGPUBackend();
    if (!(await trySet("webgpu"))) {
      await trySet("cpu");
    }
    backend = tf.getBackend();
  } else {
    // auto: prefer webgpu, then webgl, then wasm, then cpu
    const triedGpu = await (async () => {
      try {
        await ensureWebGPUBackend();
        return await trySet("webgpu");
      } catch (_) {
        return false;
      }
    })();
    if (!triedGpu) {
      if (!(await trySet("webgl"))) {
        if (!(await trySet("wasm"))) {
          await trySet("cpu");
        }
      }
    }
    backend = tf.getBackend();
  }
  return { backend, gpu: backend === "webgl", attempts };
}

export async function checkGPUSupport() {
  const tf = await ensureTF();
  try {
    await tf.setBackend("webgl");
    await tf.ready();
    const ok = tf.getBackend() === "webgl";
    await tf.setBackend("cpu");
    await tf.ready();
    return ok;
  } catch (_) {
    return false;
  }
}

export async function currentBackend() {
  const tf = await ensureTF();
  return tf.getBackend();
}

// Apply optional WebGL tuning flags. Unsupported flags are ignored.
// opts: { webglVersion?: 1|2, pack?: boolean }
export async function applyWebGLTuning(opts = {}) {
  const tf = await ensureTF();
  const safeSet = (key, val) => {
    try {
      // Some flags require setting before backend init; we set before switching to webgl below if needed
      tf.env().set(key, val);
      return true;
    } catch (_) {
      return false;
    }
  };
  const { webglVersion, pack } = opts;
  if (typeof pack === "boolean") safeSet("WEBGL_PACK", pack);
  if (webglVersion === 1 || webglVersion === 2)
    safeSet("WEBGL_VERSION", webglVersion);
  if (opts && typeof opts.forceF16 === "boolean")
    safeSet("WEBGL_FORCE_F16_TEXTURES", !!opts.forceF16);
  return true;
}
