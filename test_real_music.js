// Test script to verify chunking and reassembly with real music
import { planChunks, planChunksWithPadding, overlapAddStitchMono, overlapAddStitchMonoWithPadding } from './web/js/chunking.js';
import { readFileSync, writeFileSync } from 'fs';
import { decode } from 'wav-decoder';

// Test parameters based on user's settings
const sampleRate = 44100;
const windowSizeSeconds = 11.88;
const chunkSize = Math.floor(windowSizeSeconds * sampleRate);
const overlapFraction = 0.5; // Start with 50% overlap
const hopSize = Math.floor(chunkSize * (1 - overlapFraction));

console.log('=== Real Music Chunking/Reassembly Test ===');
console.log(`Chunk size: ${chunkSize} samples (${(chunkSize/sampleRate).toFixed(2)}s)`);
console.log(`Hop size: ${hopSize} samples (${(hopSize/sampleRate).toFixed(2)}s)`);
console.log(`Overlap: ${overlapFraction * 100}%`);

// Function to load a WAV file
async function loadWavFile(filename) {
  try {
    const buffer = readFileSync(filename);
    const audioData = await decode(buffer);
    
    if (audioData.channelData.length === 0) {
      throw new Error('No audio data found');
    }
    
    // Convert to mono by averaging channels if needed
    let monoData;
    if (audioData.channelData.length === 1) {
      monoData = audioData.channelData[0];
    } else {
      // Average all channels to create mono
      const numChannels = audioData.channelData.length;
      const length = audioData.channelData[0].length;
      monoData = new Float32Array(length);
      
      for (let i = 0; i < length; i++) {
        let sum = 0;
        for (let ch = 0; ch < numChannels; ch++) {
          sum += audioData.channelData[ch][i];
        }
        monoData[i] = sum / numChannels;
      }
    }
    
    console.log(`Loaded ${filename}:`);
    console.log(`  Sample rate: ${audioData.sampleRate} Hz`);
    console.log(`  Channels: ${audioData.channelData.length}`);
    console.log(`  Duration: ${(monoData.length / audioData.sampleRate).toFixed(2)}s`);
    console.log(`  Samples: ${monoData.length}`);
    
    return {
      data: monoData,
      sampleRate: audioData.sampleRate,
      channels: audioData.channelData.length
    };
  } catch (error) {
    console.error(`Error loading ${filename}:`, error.message);
    return null;
  }
}

// Function to save as WAV file
function saveWavFile(audioData, sampleRate, filename) {
  const buffer = new ArrayBuffer(44 + audioData.length * 2);
  const view = new DataView(buffer);
  
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  // WAV header
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + audioData.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // Mono
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
  console.log(`Saved ${filename}`);
}

// Function to calculate RMS
function calculateRMS(audio, start = 0, length = null) {
  const end = length !== null ? Math.min(start + length, audio.length) : audio.length;
  let sum = 0;
  for (let i = start; i < end; i++) {
    sum += audio[i] * audio[i];
  }
  return Math.sqrt(sum / (end - start));
}

// Function to calculate SNR (Signal-to-Noise Ratio)
function calculateSNR(original, processed) {
  if (original.length !== processed.length) {
    throw new Error('Audio lengths must match for SNR calculation');
  }
  
  let signalPower = 0;
  let noisePower = 0;
  
  for (let i = 0; i < original.length; i++) {
    signalPower += original[i] * original[i];
    const difference = original[i] - processed[i];
    noisePower += difference * difference;
  }
  
  const signalAvg = signalPower / original.length;
  const noiseAvg = noisePower / original.length;
  
  if (noiseAvg === 0) return Infinity;
  
  const snr = 10 * Math.log10(signalAvg / noiseAvg);
  return snr;
}

// Function to compare two audio signals
function compareAudio(original, processed, sampleRate) {
  console.log('\n--- Audio Comparison ---');
  
  // Overall comparison
  const originalRMS = calculateRMS(original);
  const processedRMS = calculateRMS(processed);
  const snr = calculateSNR(original, processed);
  
  console.log(`Original RMS: ${originalRMS.toFixed(6)}`);
  console.log(`Processed RMS: ${processedRMS.toFixed(6)}`);
  console.log(`SNR: ${snr.toFixed(2)} dB`);
  
  // Check beginning (first 3 seconds)
  const beginningSamples = Math.floor(3 * sampleRate);
  const originalBeginningRMS = calculateRMS(original, 0, beginningSamples);
  const processedBeginningRMS = calculateRMS(processed, 0, beginningSamples);
  
  console.log(`\nBeginning (first 3s):`);
  console.log(`  Original RMS: ${originalBeginningRMS.toFixed(6)}`);
  console.log(`  Processed RMS: ${processedBeginningRMS.toFixed(6)}`);
  console.log(`  Ratio: ${(processedBeginningRMS / originalBeginningRMS).toFixed(3)}`);
  
  // Check end (last 3 seconds)
  const endSamples = Math.floor(3 * sampleRate);
  const originalEndRMS = calculateRMS(original, original.length - endSamples, endSamples);
  const processedEndRMS = calculateRMS(processed, processed.length - endSamples, endSamples);
  
  console.log(`\nEnd (last 3s):`);
  console.log(`  Original RMS: ${originalEndRMS.toFixed(6)}`);
  console.log(`  Processed RMS: ${processedEndRMS.toFixed(6)}`);
  console.log(`  Ratio: ${(processedEndRMS / originalEndRMS).toFixed(3)}`);
  
  // Check for silence at beginning
  const silenceThreshold = 0.001;
  let silentSamples = 0;
  for (let i = 0; i < Math.min(1000, processed.length); i++) {
    if (Math.abs(processed[i]) < silenceThreshold) {
      silentSamples++;
    }
  }
  const silencePercentage = (silentSamples / Math.min(1000, processed.length)) * 100;
  console.log(`\nSilent samples at beginning: ${silentSamples}/1000 (${silencePercentage.toFixed(1)}%)`);
  
  return {
    originalRMS,
    processedRMS,
    snr,
    beginningRatio: processedBeginningRMS / originalBeginningRMS,
    endRatio: processedEndRMS / originalEndRMS,
    silencePercentage
  };
}

// Main test function
async function runTest() {
  // Note: You'll need to provide a real music file here
  const inputFilename = 'test_music.wav'; // Change this to your music file
  
  console.log(`\nLoading music file: ${inputFilename}`);
  const audio = await loadWavFile(inputFilename);
  
  if (!audio) {
    console.log('\nPlease provide a real music file named "test_music.wav" in the current directory.');
    console.log('You can create one by converting any MP3 or other audio format to WAV.');
    return;
  }
  
  const totalFrames = audio.data.length;
  console.log(`\nTotal frames: ${totalFrames}`);
  
  // Test 1: Original approach (planChunks + overlapAddStitchMono)
  console.log('\n=== Test 1: Original Approach ===');
  const originalChunks = planChunks(totalFrames, chunkSize, hopSize);
  console.log(`Created ${originalChunks.length} chunks`);
  
  // Extract chunks
  const originalOutputs = originalChunks.map(chunk => {
    return audio.data.subarray(chunk.start, chunk.end);
  });
  
  // Reassemble
  const originalReassembled = overlapAddStitchMono(originalOutputs, totalFrames, chunkSize, hopSize);
  
  // Test 2: Pre-padded approach (planChunksWithPadding + overlapAddStitchMonoWithPadding)
  console.log('\n=== Test 2: Pre-Padded Approach ===');
  const paddedChunks = planChunksWithPadding(totalFrames, chunkSize, hopSize);
  console.log(`Created ${paddedChunks.length} chunks (including padding)`);
  
  // Extract chunks (with padding)
  const paddedOutputs = paddedChunks.map(chunk => {
    if (chunk.isPadding) {
      // Return zeros for padding chunks
      const paddingLength = chunk.end - chunk.start;
      return new Float32Array(paddingLength);
    } else {
      return audio.data.subarray(chunk.start, chunk.end);
    }
  });
  
  // Reassemble
  const paddedReassembled = overlapAddStitchMonoWithPadding(paddedChunks, paddedOutputs, totalFrames, chunkSize, hopSize);
  
  // Compare results
  console.log('\n=== Comparison Results ===');
  
  console.log('\nOriginal vs Original Reassembled:');
  const originalComparison = compareAudio(audio.data, originalReassembled, audio.sampleRate);
  
  console.log('\nOriginal vs Padded Reassembled:');
  const paddedComparison = compareAudio(audio.data, paddedReassembled, audio.sampleRate);
  
  // Save all files for manual inspection
  saveWavFile(audio.data, audio.sampleRate, '01_original.wav');
  saveWavFile(originalReassembled, audio.sampleRate, '02_original_reassembled.wav');
  saveWavFile(paddedReassembled, audio.sampleRate, '03_padded_reassembled.wav');
  
  // Also save difference files
  const originalDiff = new Float32Array(audio.data.length);
  const paddedDiff = new Float32Array(audio.data.length);
  
  for (let i = 0; i < audio.data.length; i++) {
    originalDiff[i] = audio.data[i] - originalReassembled[i];
    paddedDiff[i] = audio.data[i] - paddedReassembled[i];
  }
  
  saveWavFile(originalDiff, audio.sampleRate, '04_original_difference.wav');
  saveWavFile(paddedDiff, audio.sampleRate, '05_padded_difference.wav');
  
  // Summary
  console.log('\n=== Test Summary ===');
  console.log('Files saved for manual inspection:');
  console.log('  01_original.wav - Original audio');
  console.log('  02_original_reassembled.wav - Original method reassembled');
  console.log('  03_padded_reassembled.wav - Pre-padded method reassembled');
  console.log('  04_original_difference.wav - Difference (original vs original reassembled)');
  console.log('  05_padded_difference.wav - Difference (original vs padded reassembled)');
  
  console.log('\nQuality metrics:');
  console.log(`Original method SNR: ${originalComparison.snr.toFixed(2)} dB`);
  console.log(`Pre-padded method SNR: ${paddedComparison.snr.toFixed(2)} dB`);
  console.log(`Original method beginning preservation: ${(originalComparison.beginningRatio * 100).toFixed(1)}%`);
  console.log(`Pre-padded method beginning preservation: ${(paddedComparison.beginningRatio * 100).toFixed(1)}%`);
  console.log(`Original method end preservation: ${(originalComparison.endRatio * 100).toFixed(1)}%`);
  console.log(`Pre-padded method end preservation: ${(paddedComparison.endRatio * 100).toFixed(1)}%`);
  
  // Determine if the fix works
  const snrThreshold = 60; // Good quality SNR
  const preservationThreshold = 0.95; // 95% preservation
  
  const originalWorks = originalComparison.snr >= snrThreshold && 
                      originalComparison.beginningRatio >= preservationThreshold &&
                      originalComparison.endRatio >= preservationThreshold;
  
  const paddedWorks = paddedComparison.snr >= snrThreshold && 
                    paddedComparison.beginningRatio >= preservationThreshold &&
                    paddedComparison.endRatio >= preservationThreshold;
  
  console.log('\n=== Conclusion ===');
  console.log(`Original method: ${originalWorks ? '✅ WORKS' : '❌ HAS ISSUES'}`);
  console.log(`Pre-padded method: ${paddedWorks ? '✅ WORKS' : '❌ HAS ISSUES'}`);
  
  if (paddedWorks && !originalWorks) {
    console.log('✅ Pre-padded method fixes the issues!');
  } else if (paddedWorks && originalWorks) {
    console.log('ℹ️  Both methods work, pre-padded method is equivalent.');
  } else if (!paddedWorks && originalWorks) {
    console.log('❌ Pre-padded method introduces issues!');
  } else {
    console.log('❌ Both methods have issues.');
  }
}

// Run the test
runTest().catch(console.error);
