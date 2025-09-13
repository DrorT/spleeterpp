// Chunk planner with optional overlap

export function planChunks(totalFrames, chunkSize, hopSize) {
  if (chunkSize <= 0) return [{ start: 0, end: totalFrames }];
  const hop = hopSize > 0 ? hopSize : chunkSize;
  const chunks = [];
  for (let start = 0; start < totalFrames; start += hop) {
    const end = Math.min(start + chunkSize, totalFrames);
    chunks.push({ start, end });
    if (end === totalFrames) break;
  }
  return chunks;
}

export function overlapAddStitchMono(outputs, totalFrames, chunkSize, hopSize) {
  const out = new Float32Array(totalFrames);
  const acc = new Float32Array(totalFrames);
  const fade = Math.max(0, chunkSize - hopSize);
  const half = Math.floor(fade / 2);
  let writePos = 0;
  for (let ci = 0; ci < outputs.length; ci++) {
    const chunk = outputs[ci];
    for (let i = 0; i < chunk.length; i++) {
      const pos = writePos + i;
      if (pos >= totalFrames) break;
      let w = 1.0;
      if (fade > 0) {
        if (i < half) w = i / half;
        else if (i > chunk.length - 1 - half) w = (chunk.length - 1 - i) / half;
        if (w < 0) w = 0; if (w > 1) w = 1;
      }
      out[pos] += chunk[i] * w;
      acc[pos] += w;
    }
    writePos += hopSize;
  }
  for (let i = 0; i < totalFrames; i++) {
    const a = acc[i];
    if (a > 1e-12) out[i] /= a;
  }
  return out;
}
