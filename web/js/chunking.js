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

// Enhanced chunk planner with pre-padding strategy
export function planChunksWithPadding(totalFrames, chunkSize, hopSize) {
  if (chunkSize <= 0) return [{ start: 0, end: totalFrames }];
  const hop = hopSize > 0 ? hopSize : chunkSize;
  const fadeLength = Math.max(0, chunkSize - hopSize);
  
  // If no overlap/fade, use original planChunks
  if (fadeLength === 0) {
    return planChunks(totalFrames, chunkSize, hopSize);
  }
  
  const chunks = [];
  
  // Add pre-padding chunk (silent region before the first real chunk)
  chunks.push({ 
    start: -fadeLength, 
    end: 0, 
    isPadding: true, 
    paddingType: 'pre' 
  });
  
  // Add regular chunks
  for (let start = 0; start < totalFrames; start += hop) {
    const end = Math.min(start + chunkSize, totalFrames);
    chunks.push({ start, end, isPadding: false });
    if (end === totalFrames) break;
  }
  
  // Add post-padding chunk (silent region after the last real chunk)
  chunks.push({ 
    start: totalFrames, 
    end: totalFrames + fadeLength, 
    isPadding: true, 
    paddingType: 'post' 
  });
  
  return chunks;
}

export function overlapAddStitchMono(outputs, totalFrames, chunkSize, hopSize) {
  const out = new Float32Array(totalFrames);
  const acc = new Float32Array(totalFrames);
  const fade = Math.max(0, chunkSize - hopSize);
  let writePos = 0;
  for (let ci = 0; ci < outputs.length; ci++) {
    const chunk = outputs[ci];
    for (let i = 0; i < chunk.length; i++) {
      const pos = writePos + i;
      if (pos >= totalFrames) break;
      // Triangular window across the overlap region: ramps up in the first fade samples,
      // stays 1 in the middle (if any), ramps down in the last fade samples.
      let w = 1.0;
      if (fade > 0) {
        if (i < fade) w = i / fade; // rising edge
        else if (i >= chunk.length - fade) w = (chunk.length - 1 - i) / fade; // falling edge
        if (w < 0) w = 0;
        if (w > 1) w = 1;
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

// Enhanced overlap-add stitch function that works with padded chunks
export function overlapAddStitchMonoWithPadding(chunks, outputs, originalTotalFrames, chunkSize, hopSize) {
  const fadeLength = Math.max(0, chunkSize - hopSize);
  
  // If no fade/overlap, use original function
  if (fadeLength === 0) {
    return overlapAddStitchMono(outputs, originalTotalFrames, chunkSize, hopSize);
  }
  
  // Extract only the real chunks (excluding padding chunks) and use standard overlap-add
  const realChunks = chunks.filter(c => !c.isPadding);
  const realOutputs = outputs.filter((_, i) => !chunks[i].isPadding);
  
  // Use the standard overlap-add function on the real chunks
  const result = overlapAddStitchMono(realOutputs, originalTotalFrames, chunkSize, hopSize);
  
  return result;
}
