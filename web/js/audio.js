// Audio decoding and high-quality resampling to 44.1kHz

export async function decodeArrayBufferToPCM(ab) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuf = await ctx.decodeAudioData(ab.slice(0));
  const sr = audioBuf.sampleRate;
  const ch = audioBuf.numberOfChannels;
  const frames = audioBuf.length;
  const channels = [];
  for (let i = 0; i < ch; i++) channels.push(audioBuf.getChannelData(i));
  ctx.close();
  return { sampleRate: sr, channels: channels, frames };
}

export async function resampleTo44100({ sampleRate, channels }) {
  if (sampleRate === 44100) return { sampleRate, channels };
  const length = Math.ceil(channels[0].length * (44100 / sampleRate));
  const off = new OfflineAudioContext(channels.length, length, 44100);
  const buffer = off.createBuffer(
    channels.length,
    channels[0].length,
    sampleRate
  );
  for (let i = 0; i < channels.length; i++)
    buffer.copyToChannel(channels[i], i, 0);
  const src = off.createBufferSource();
  src.buffer = buffer;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  const outCh = [];
  for (let i = 0; i < out.numberOfChannels; i++)
    outCh.push(out.getChannelData(i));
  return { sampleRate: 44100, channels: outCh };
}

export function interleaveStereoFloat32(left, right) {
  const n = left.length;
  const out = new Float32Array(n * 2);
  let o = 0;
  for (let i = 0; i < n; i++) {
    out[o++] = left[i];
    out[o++] = right[i];
  }
  return out;
}

export function toMonoFloat32(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < channels.length; c++) s += channels[c][i];
    out[i] = s / channels.length;
  }
  return out;
}
