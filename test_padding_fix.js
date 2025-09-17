// Test script to verify the pre-padding fix for missing audio issue
import { planChunks, planChunksWithPadding, overlapAddStitchMono, overlapAddStitchMonoWithPadding } from './web/js/chunking.js';

// Test parameters based on user's reported issue
const sampleRate = 44100;
const windowSizeSeconds = 11.88;
const chunkSize = Math.floor(windowSizeSeconds * sampleRate); // ~11.88 seconds

// Test cases: 50% overlap and 25% overlap
const testCases = [
  { 
    name: '50% overlap (user issue: missing 6 seconds)', 
    overlapFraction: 0.5,
    expectedMissingSeconds: 6
  },
  { 
    name: '25% overlap (user issue: missing 9 seconds)', 
    overlapFraction: 0.25,
    expectedMissingSeconds: 9
  }
];

// Simulate some audio data (simple sine wave for testing)
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

// Test the chunking and stitching
function runTest(testCase) {
  console.log(`\n=== Testing: ${testCase.name} ===`);
  
  const audioLengthSeconds = 30; // 30 seconds of test audio
  const totalFrames = Math.floor(audioLengthSeconds * sampleRate);
  const hopSize = Math.floor(chunkSize * (1 - testCase.overlapFraction));
  
  console.log(`Chunk size: ${chunkSize} samples (${(chunkSize/sampleRate).toFixed(2)}s)`);
  console.log(`Hop size: ${hopSize} samples (${(hopSize/sampleRate).toFixed(2)}s)`);
  console.log(`Overlap: ${testCase.overlapFraction * 100}%`);
  console.log(`Fade length: ${chunkSize - hopSize} samples (${((chunkSize - hopSize)/sampleRate).toFixed(2)}s)`);
  
  // Generate test audio
  const testAudio = generateTestAudio(audioLengthSeconds, sampleRate);
  
  // Test original approach (should have missing audio at beginning/end)
  console.log('\n--- Original Approach ---');
  const originalChunks = planChunks(totalFrames, chunkSize, hopSize);
  const originalOutputs = originalChunks.map(chunk => {
    return testAudio.subarray(chunk.start, chunk.end);
  });
  const originalResult = overlapAddStitchMono(originalOutputs, totalFrames, chunkSize, hopSize);
  
  // Test new pre-padded approach (should preserve beginning/end audio)
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
  
  // Debug: Show chunk information
  console.log('Chunk details:');
  paddedChunks.forEach((chunk, index) => {
    const isPadding = chunk.isPadding ? ' (PADDING)' : '';
    console.log(`  Chunk ${index}: start=${chunk.start}, end=${chunk.end}, length=${chunk.end - chunk.start}${isPadding}`);
  });
  
  const paddedResult = overlapAddStitchMonoWithPadding(paddedChunks, paddedOutputs, totalFrames, chunkSize, hopSize);
  
  // Analyze results
  console.log('\n--- Analysis ---');
  console.log(`Original chunks: ${originalChunks.length}`);
  console.log(`Padded chunks: ${paddedChunks.length} (includes ${paddedChunks.filter(c => c.isPadding).length} padding chunks)`);
  
  // Check beginning of audio (first few seconds)
  const beginningCheckSamples = Math.floor(2 * sampleRate); // Check first 2 seconds
  let originalBeginningRMS = 0;
  let paddedBeginningRMS = 0;
  
  for (let i = 0; i < beginningCheckSamples; i++) {
    originalBeginningRMS += originalResult[i] * originalResult[i];
    paddedBeginningRMS += paddedResult[i] * paddedResult[i];
  }
  originalBeginningRMS = Math.sqrt(originalBeginningRMS / beginningCheckSamples);
  paddedBeginningRMS = Math.sqrt(paddedBeginningRMS / beginningCheckSamples);
  
  console.log(`Beginning audio RMS - Original: ${originalBeginningRMS.toFixed(6)}, Padded: ${paddedBeginningRMS.toFixed(6)}`);
  
  // Check end of audio (last few seconds)
  const endCheckSamples = Math.floor(2 * sampleRate); // Check last 2 seconds
  let originalEndRMS = 0;
  let paddedEndRMS = 0;
  
  for (let i = totalFrames - endCheckSamples; i < totalFrames; i++) {
    originalEndRMS += originalResult[i] * originalResult[i];
    paddedEndRMS += paddedResult[i] * paddedResult[i];
  }
  originalEndRMS = Math.sqrt(originalEndRMS / endCheckSamples);
  paddedEndRMS = Math.sqrt(paddedEndRMS / endCheckSamples);
  
  console.log(`End audio RMS - Original: ${originalEndRMS.toFixed(6)}, Padded: ${paddedEndRMS.toFixed(6)}`);
  
  // Check if the fix worked
  const expectedRMS = 0.3535; // RMS of sine wave with amplitude 0.5
  const beginningImprovement = paddedBeginningRMS / originalBeginningRMS;
  const endImprovement = paddedEndRMS / originalEndRMS;
  
  console.log(`\nBeginning improvement factor: ${beginningImprovement.toFixed(2)}x`);
  console.log(`End improvement factor: ${endImprovement.toFixed(2)}x`);
  
  const success = (paddedBeginningRMS > expectedRMS * 0.8) && (paddedEndRMS > expectedRMS * 0.8);
  console.log(`\n✅ Test ${success ? 'PASSED' : 'FAILED'}`);
  
  return {
    success,
    beginningImprovement,
    endImprovement,
    originalBeginningRMS,
    paddedBeginningRMS,
    originalEndRMS,
    paddedEndRMS
  };
}

// Run all tests
console.log('Testing Pre-Padding Fix for Missing Audio Issue');
console.log('==============================================');

const results = testCases.map(runTest);

// Summary
console.log('\n=== SUMMARY ===');
results.forEach((result, index) => {
  const testCase = testCases[index];
  console.log(`${testCase.name}: ${result.success ? '✅ FIXED' : '❌ STILL BROKEN'}`);
  console.log(`  Beginning: ${result.originalBeginningRMS.toFixed(4)} → ${result.paddedBeginningRMS.toFixed(4)} (${result.beginningImprovement.toFixed(1)}x improvement)`);
  console.log(`  End: ${result.originalEndRMS.toFixed(4)} → ${result.paddedEndRMS.toFixed(4)} (${result.endImprovement.toFixed(1)}x improvement)`);
});

const allPassed = results.every(r => r.success);
console.log(`\nOverall: ${allPassed ? '✅ ALL TESTS PASSED - Fix appears to work!' : '❌ Some tests failed - Fix needs more work'}`);
