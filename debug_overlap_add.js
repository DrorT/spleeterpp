// Debug script to examine the overlap-add process in detail
import { planChunks, planChunksWithPadding, overlapAddStitchMono, overlapAddStitchMonoWithPadding } from './web/js/chunking.js';
import { writeFileSync } from 'fs';

// Test parameters
const sampleRate = 44100;
const windowSizeSeconds = 11.88;
const chunkSize = Math.floor(windowSizeSeconds * sampleRate);
const overlapFraction = 0.5; // 50% overlap
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

console.log('=== Debug Overlap-Add Process ===');
console.log(`Chunk size: ${chunkSize} samples (${(chunkSize/sampleRate).toFixed(2)}s)`);
console.log(`Hop size: ${hopSize} samples (${(hopSize/sampleRate).toFixed(2)}s)`);
console.log(`Overlap: ${overlapFraction * 100}%`);

// Generate test audio
const audioLengthSeconds = 30;
const totalFrames = Math.floor(audioLengthSeconds * sampleRate);
const testAudio = generateTestAudio(audioLengthSeconds, sampleRate);

console.log(`\nTest audio: ${audioLengthSeconds}s, ${totalFrames} samples`);

// Test original approach
console.log('\n--- Original Approach ---');
const originalChunks = planChunks(totalFrames, chunkSize, hopSize);
const originalOutputs = originalChunks.map(chunk => {
  return testAudio.subarray(chunk.start, chunk.end);
});

console.log(`Original chunks: ${originalChunks.length}`);
originalChunks.forEach((chunk, index) => {
  console.log(`  Chunk ${index}: start=${chunk.start}, end=${chunk.end}, length=${chunk.end - chunk.start}`);
});

const originalResult = overlapAddStitchMono(originalOutputs, totalFrames, chunkSize, hopSize);

// Test pre-padded approach
console.log('\n--- Pre-Padded Approach ---');
const paddedChunks = planChunksWithPadding(totalFrames, chunkSize, hopSize);
const paddedOutputs = paddedChunks.map(chunk => {
  if (chunk.isPadding) {
    // Return zeros for padding chunks
    const paddingLength = chunk.end - chunk.start;
    return new Float32Array(paddingLength);
  } else {
    return testAudio.subarray(chunk.start, chunk.end);
  }
});

console.log(`Padded chunks: ${paddedChunks.length}`);
paddedChunks.forEach((chunk, index) => {
  const isPadding = chunk.isPadding ? ' (PADDING)' : '';
  console.log(`  Chunk ${index}: start=${chunk.start}, end=${chunk.end}, length=${chunk.end - chunk.start}${isPadding}`);
});

const paddedResult = overlapAddStitchMonoWithPadding(paddedChunks, paddedOutputs, totalFrames, chunkSize, hopSize);

// Compare results
console.log('\n--- Results Comparison ---');

// Check first few seconds of both results
const checkSeconds = 6; // Check first 6 seconds as user mentioned 6 seconds of silence
const checkSamples = Math.floor(checkSeconds * sampleRate);

console.log(`\nFirst ${checkSeconds} seconds comparison:`);
console.log('Sample | Original | Padded | Difference');
console.log('-------|----------|--------|------------');

let silentSamples = 0;
for (let i = 0; i < Math.min(checkSamples, totalFrames); i++) {
  const orig = originalResult[i];
  const pad = paddedResult[i];
  const diff = Math.abs(orig - pad);
  
  if (i < 20) { // Show first 20 samples
    console.log(`${i.toString().padStart(5)} | ${orig.toFixed(6)} | ${pad.toFixed(6)} | ${diff.toFixed(6)}`);
  }
  
  // Count silent samples in padded result
  if (Math.abs(pad) < 0.001) {
    silentSamples++;
  }
}

console.log(`\nSilent samples in first ${checkSeconds} seconds of padded result: ${silentSamples}/${checkSamples} (${(silentSamples/checkSamples*100).toFixed(1)}%)`);

// Check RMS values
function calculateRMS(audio, start, length) {
  let sum = 0;
  const end = Math.min(start + length, audio.length);
  for (let i = start; i < end; i++) {
    sum += audio[i] * audio[i];
  }
  return Math.sqrt(sum / (end - start));
}

const origRMS = calculateRMS(originalResult, 0, checkSamples);
const padRMS = calculateRMS(paddedResult, 0, checkSamples);

console.log(`\nRMS comparison for first ${checkSeconds} seconds:`);
console.log(`Original: ${origRMS.toFixed(6)}`);
console.log(`Padded: ${padRMS.toFixed(6)}`);

// Save both results as WAV files for comparison
function saveWavFile(audioData, sampleRate, filename) {
  const buffer = new ArrayBuffer(44 + audioData.length * 2);
  const view = new DataView(buffer);
  
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
  
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    view.setInt16(44 + i * 2, sample * 0x7FFF, true);
  }
  
  writeFileSync(filename, Buffer.from(buffer));
}

saveWavFile(originalResult, sampleRate, 'original_result.wav');
saveWavFile(paddedResult, sampleRate, 'padded_result.wav');

console.log(`\nSaved results as WAV files:`);
console.log(`- original_result.wav: Original overlap-add result`);
console.log(`- padded_result.wav: Pre-padded overlap-add result`);

// Also save just the first 10 seconds for easier comparison
const first10Seconds = Math.floor(10 * sampleRate);
saveWavFile(originalResult.slice(0, first10Seconds), sampleRate, 'original_first_10s.wav');
saveWavFile(paddedResult.slice(0, first10Seconds), sampleRate, 'padded_first_10s.wav');

console.log(`\nSaved first 10 seconds as:`);
console.log(`- original_first_10s.wav`);
console.log(`- padded_first_10s.wav`);

console.log('\n=== Analysis Complete ===');
console.log('Please play the WAV files to hear the difference.');
console.log('The padded result should have silence at the beginning if there\'s an issue.');
