// Lightweight radix-2 FFT optimized for N=4096 and STFT helpers
// ESM module: export init, hann, stftStereo

const N = 4096;
const LOG2N = 12; // 2^12 = 4096
let bitrev = null;
let cosTable = null;
let sinTable = null;
let hannWindow = null;

function initTables() {
  if (bitrev && cosTable && sinTable) return;
  // Bit-reversal indices
  bitrev = new Uint32Array(N);
  let j = 0;
  for (let i = 0; i < N; i++) {
    bitrev[i] = j;
    let bit = N >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;
  }
  // Twiddles W_N^k = cos(2pi k/N) - i sin(...)
  cosTable = new Float32Array(N >> 1);
  sinTable = new Float32Array(N >> 1);
  for (let k = 0; k < N >> 1; k++) {
    const ang = (-2 * Math.PI * k) / N;
    cosTable[k] = Math.cos(ang);
    sinTable[k] = Math.sin(ang);
  }
}

export function hann() {
  if (hannWindow) return hannWindow;
  hannWindow = new Float32Array(N);
  // Use periodic Hann to match TensorFlow's hann_window(periodic=True)
  for (let n = 0; n < N; n++) {
    hannWindow[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / N));
  }
  return hannWindow;
}

function fftInPlace(real, imag) {
  // bit-reverse copy
  for (let i = 0; i < N; i++) {
    const j = bitrev[i];
    if (j > i) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      const ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
  }
  // Cooley–Tukey iterative
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const step = N / len;
    for (let i = 0; i < N; i += len) {
      let k = 0;
      for (let j = 0; j < half; j++, k += step) {
        const wr = cosTable[k];
        const wi = sinTable[k];
        const i0 = i + j;
        const i1 = i0 + half;
        const xr = real[i1];
        const xi = imag[i1];
        const tr = wr * xr - wi * xi;
        const ti = wr * xi + wi * xr;
        real[i1] = real[i0] - tr;
        imag[i1] = imag[i0] - ti;
        real[i0] = real[i0] + tr;
        imag[i0] = imag[i0] + ti;
      }
    }
  }
}

// STFT for up to 2 channels. Returns arrays per channel of shape [frames][bins]
export function stftStereo(channels, hop = 1024, maxFrames) {
  initTables();
  const w = hann();
  const C = Math.min(2, channels.length);
  const T = channels[0].length;
  const framesTotal = Math.max(0, 1 + Math.floor((T - N) / hop));
  const frames =
    maxFrames != null
      ? Math.max(0, Math.min(framesTotal, maxFrames))
      : framesTotal;
  const bins = (N >> 1) + 1; // 2049
  const out = new Array(C);
  for (let c = 0; c < C; c++) {
    const real = new Float32Array(frames * bins);
    const imag = new Float32Array(frames * bins);
    const src = channels[c];
    const bufR = new Float32Array(N);
    const bufI = new Float32Array(N);
    let idx = 0;
    for (let f = 0; f < frames; f++) {
      const start = f * hop;
      for (let n = 0; n < N; n++) {
        bufR[n] = src[start + n] * w[n];
        bufI[n] = 0;
      }
      fftInPlace(bufR, bufI);
      // take bins 0..N/2 inclusive
      for (let k = 0; k < bins; k++, idx++) {
        real[idx] = bufR[k];
        imag[idx] = bufI[k];
      }
    }
    out[c] = { real, imag, frames, bins };
  }
  return { channels: out, frames, bins };
}
