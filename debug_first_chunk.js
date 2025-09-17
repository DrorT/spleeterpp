// Debug script to examine the first chunk from the chunking process
import { planChunks, planChunksWithPadding } from './web/js/chunking.js';
import { writeFileSync } from 'fs';

// Test parameters
const sampleRate = 44100;
const windowSizeSeconds = 11.88;
const chunkSize = Math.floor(windowSizeSeconds * sampleRate);
const overlapFraction = 0.5; // Start with 50% overlap
const hopSize = Math.floor(chunkSize * (1 - overlapFraction));

// Generate test audio (30 seconds)
function generateTestAudio(lengthSeconds, sampleRate = 44100) {
  const samples = Math.floor(lengthSeconds * sampleRate);
  const audio = new Float32Array(samples);
  const frequency = 440; // A4 note
  
  for (let i = 0; i < samples; i++) {
    const t = i / sampleRate;
    audio[i] = Math.sin(2 * Math.PI * frequency * t) * 0.5;
  }
  
  return audio;
}

console.log('=== Debug First Chunk Analysis ===');
console.log(`Chunk size: ${chunkSize} samples (${(chunkSize/sampleRate).toFixed(2)}s)`);
console.log(`Hop size: ${hopSize} samples (${(hopSize/sampleRate).toFixed(2)}s)`);
console.log(`Overlap: ${overlapFraction * 100}%`);

// Generate test audio
const audioLengthSeconds = 30;
const totalFrames = Math.floor(audioLengthSeconds * sampleRate);
const testAudio = generateTestAudio(audioLengthSeconds, sampleRate);

console.log(`\nTest audio: ${audioLengthSeconds}s, ${totalFrames} samples`);

// Test original chunking
console.log('\n--- Original Chunking ---');
const originalChunks = planChunks(totalFrames, chunkSize, hopSize);
console.log(`Number of chunks: ${originalChunks.length}`);

const firstOriginalChunk = originalChunks[0];
console.log(`\nFirst original chunk:`);
console.log(`  Start: ${firstOriginalChunk.start}`);
console.log(`  End: ${firstOriginalChunk.end}`);
console.log(`  Length: ${firstOriginalChunk.end - firstOriginalChunk.start} samples (${((firstOriginalChunk.end - firstOriginalChunk.start)/sampleRate).toFixed(2)}s)`);

// Extract the first chunk audio
const firstOriginalAudio = testAudio.subarray(firstOriginalChunk.start, firstOriginalChunk.end);
console.log(`\nFirst original chunk audio:`);
console.log(`  First 10 samples: ${Array.from(firstOriginalAudio.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}`);
console.log(`  Last 10 samples: ${Array.from(firstOriginalAudio.slice(-10)).map(v => v.toFixed(4)).join(', ')}`);

// Calculate RMS of first chunk
let rms = 0;
for (let i = 0; i < firstOriginalAudio.length; i++) {
  rms += firstOriginalAudio[i] * firstOriginalAudio[i];
}
rms = Math.sqrt(rms / firstOriginalAudio.length);
console.log(`  RMS: ${rms.toFixed(6)}`);

// Test pre-padded chunking
console.log('\n--- Pre-Padded Chunking ---');
const paddedChunks = planChunksWithPadding(totalFrames, chunkSize, hopSize);
console.log(`Number of chunks: ${paddedChunks.length}`);

// Show all chunks
console.log('\nAll padded chunks:');
paddedChunks.forEach((chunk, index) => {
  const isPadding = chunk.isPadding ? ' (PADDING)' : '';
  console.log(`  Chunk ${index}: start=${chunk.start}, end=${chunk.end}, length=${chunk.end - chunk.start}${isPadding}`);
});

// Find the first real chunk (not padding)
const firstRealChunkIndex = paddedChunks.findIndex(c => !c.isPadding);
const firstRealChunk = paddedChunks[firstRealChunkIndex];
console.log(`\nFirst real chunk (index ${firstRealChunkIndex}):`);
console.log(`  Start: ${firstRealChunk.start}`);
console.log(`  End: ${firstRealChunk.end}`);
console.log(`  Length: ${firstRealChunk.end - firstRealChunk.start} samples (${((firstRealChunk.end - firstRealChunk.start)/sampleRate).toFixed(2)}s)`);

// Extract the first real chunk audio
const firstRealAudio = testAudio.subarray(firstRealChunk.start, firstRealChunk.end);
console.log(`\nFirst real chunk audio:`);
console.log(`  First 10 samples: ${Array.from(firstRealAudio.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}`);
console.log(`  Last 10 samples: ${Array.from(firstRealAudio.slice(-10)).map(v => v.toFixed(4)).join(', ')}`);

// Calculate RMS of first real chunk
let realRms = 0;
for (let i = 0; i < firstRealAudio.length; i++) {
  realRms += firstRealAudio[i] * firstRealAudio[i];
}
realRms = Math.sqrt(realRms / firstRealAudio.length);
console.log(`  RMS: ${realRms.toFixed(6)}`);

// Save the first real chunk as a WAV file for testing
function saveWavFile(audioData, sampleRate, filename) {
  // Simple WAV file creation
  const buffer = new ArrayBuffer(44 + audioData.length * 2);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + audioData.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, audioData.length * 2, true);
  
  // Audio data (convert Float32 to Int16)
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    view.setInt16(44 + i * 2, sample * 0x7FFF, true);
  }
  
  writeFileSync(filename, Buffer.from(buffer));
}

// Save the first real chunk as a WAV file
saveWavFile(firstRealAudio, sampleRate, 'first_real_chunk.wav');
console.log(`\nSaved first real chunk as 'first_real_chunk.wav'`);

// Also save the original first chunk for comparison
saveWavFile(firstOriginalAudio, sampleRate, 'first_original_chunk.wav');
console.log(`Saved original first chunk as 'first_original_chunk.wav'`);

console.log('\n=== Analysis Complete ===');
console.log('Please play the WAV files to compare the audio:');
console.log('1. first_original_chunk.wav - Should have normal audio');
console.log('2. first_real_chunk.wav - Should have the same audio as original');
