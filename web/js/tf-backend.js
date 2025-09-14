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

export async function configureBackend(preference = "auto") {
  const tf = await ensureTF();
  const wantWebGL = preference === "gpu" || preference === "auto";
  let backend = "cpu";
  try {
    if (wantWebGL) {
      await tf.setBackend("webgl");
      await tf.ready();
      backend = tf.getBackend();
      if (backend !== "webgl" && preference === "gpu") {
        // hard gpu preference, throw if not webgl
        throw new Error("WebGL backend unavailable");
      }
    }
    if (backend !== "webgl") {
      await tf.setBackend("cpu");
      await tf.ready();
      backend = "cpu";
    }
  } catch (e) {
    await tf.setBackend("cpu");
    await tf.ready();
    backend = "cpu";
  }
  return { backend, gpu: backend === "webgl" };
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
// opts: { precision: 'low'|'high', webglVersion?: 1|2, pack?: boolean }
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
  const { precision, webglVersion, pack } = opts;
  if (typeof pack === "boolean") safeSet("WEBGL_PACK", pack);
  if (webglVersion === 1 || webglVersion === 2)
    safeSet("WEBGL_VERSION", webglVersion);
  if (precision === "low") {
    // Prefer half-float textures to reduce bandwidth; ignore if not supported
    safeSet("WEBGL_FORCE_F16_TEXTURES", true);
  } else if (precision === "high") {
    safeSet("WEBGL_FORCE_F16_TEXTURES", false);
  }
  return true;
}
