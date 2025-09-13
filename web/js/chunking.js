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
