const fs = require('fs');
const { planChunks, overlapAddStitchMono } = require('./web/js/chunking.js');

// Audio generation parameters
const SAMPLE_RATE = 44100;
const DURATION = 30; // seconds
const TOTAL_FRAMES = SAMPLE_RATE * DURATION;

// Generate diverse test audio with sine waves and beeps
function generateTestAudio() {
    console.log('Generating diverse 30-second test audio...');
    
    const audio = new Float32Array(TOTAL_FRAMES);
    
    // 1. Base sine wave that changes frequency over time
    for (let i = 0; i < TOTAL_FRAMES; i++) {
        const t = i / SAMPLE_RATE;
        // Frequency sweeps from 200Hz to 800Hz over 30 seconds
        const freq = 200 + (600 * t / DURATION);
        audio[i] = Math.sin(2 * Math.PI * freq * t) * 0.3;
    }
    
    // 2. Add secondary sine wave with different pattern
    for (let i = 0; i < TOTAL_FRAMES; i++) {
        const t = i / SAMPLE_RATE;
        // Frequency modulated sine wave
        const carrierFreq = 440;
        const modFreq = 5;
        const modIndex = 50;
        const freq = carrierFreq + modIndex * Math.sin(2 * Math.PI * modFreq * t);
        audio[i] += Math.sin(2 * Math.PI * freq * t) * 0.2;
    }
    
    // 3. Add specific beeps/ticks at known times for easy identification
    const beepTimes = [2, 5, 8, 12, 15, 18, 22, 25, 28]; // seconds
    const beepFreq = 1000; // 1kHz beeps
    const beepDuration = 0.1; // 100ms beeps
    
    beepTimes.forEach(time => {
        const startSample = Math.floor(time * SAMPLE_RATE);
        const endSample = Math.floor((time + beepDuration) * SAMPLE_RATE);
        
        for (let i = startSample; i < Math.min(endSample, TOTAL_FRAMES); i++) {
            const t = (i - startSample) / SAMPLE_RATE;
            audio[i] += Math.sin(2 * Math.PI * beepFreq * t) * 0.5;
        }
    });
    
    // 4. Add some noise bursts
    const noiseTimes = [3, 7, 11, 16, 20, 24, 27]; // seconds
    const noiseDuration = 0.05; // 50ms noise bursts
    
    noiseTimes.forEach(time => {
        const startSample = Math.floor(time * SAMPLE_RATE);
        const endSample = Math.floor((time + noiseDuration) * SAMPLE_RATE);
        
        for (let i = startSample; i < Math.min(endSample, TOTAL_FRAMES); i++) {
            audio[i] += (Math.random() - 0.5) * 0.3;
        }
    });
    
    // 5. Add low frequency rumble
    for (let i = 0; i < TOTAL_FRAMES; i++) {
        const t = i / SAMPLE_RATE;
        audio[i] += Math.sin(2 * Math.PI * 60 * t) * 0.1; // 60Hz rumble
    }
    
    // Normalize to prevent clipping
    let maxSample = 0;
    for (let i = 0; i < TOTAL_FRAMES; i++) {
        maxSample = Math.max(maxSample, Math.abs(audio[i]));
    }
    if (maxSample > 1) {
        for (let i = 0; i < TOTAL_FRAMES; i++) {
            audio[i] /= maxSample;
        }
    }
    
    console.log(`Generated audio with max sample value: ${maxSample}`);
    console.log('Audio contains:');
    console.log('- Frequency sweeping sine wave (200-800Hz)');
    console.log('- Frequency modulated sine wave (440Hz carrier)');
    console.log('- Beeps at times:', beepTimes.join(', '), 'seconds');
    console.log('- Noise bursts at times:', noiseTimes.join(', '), 'seconds');
    console.log('- 60Hz low frequency rumble');
    
    return audio;
}

// Convert Float32Array to WAV format
function float32ToWav(data, sampleRate = SAMPLE_RATE) {
    const numChannels = 1;
    const bytesPerSample = 2; // 16-bit PCM
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataLength = data.length * bytesPerSample;
    const buffer = Buffer.alloc(44 + dataLength);
    
    // RIFF header
    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + dataLength, 4);
    buffer.write('WAVE', 8);
    
    // fmt chunk
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16); // PCM chunk size
    buffer.writeUInt16LE(1, 20); // PCM format
    buffer.writeUInt16LE(numChannels, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(byteRate, 28);
    buffer.writeUInt16LE(blockAlign, 32);
    buffer.writeUInt16LE(bytesPerSample * 8, 34);
    
    // data chunk
    buffer.write('data', 36);
    buffer.writeUInt32LE(dataLength, 40);
    
    // PCM data (float32 -> int16)
    let offset = 44;
    for (let i = 0; i < data.length; i++) {
        let s = Math.max(-1, Math.min(1, data[i]));
        const sample = s < 0 ? s * 0x8000 : s * 0x7fff;
        buffer.writeInt16LE(Math.round(sample), offset);
        offset += 2;
    }
    
    return buffer;
}

// Process audio through chunking and reassembly
function processAudioChunking(originalAudio) {
    console.log('\\nProcessing audio through chunking system...');
    
    // Chunking parameters - 50% overlap as requested
    const chunkSize = Math.floor(SAMPLE_RATE * 5); // 5-second chunks
    const hopSize = Math.floor(chunkSize * 0.5); // 50% overlap
    
    console.log(`Chunk size: ${chunkSize} samples (${chunkSize/SAMPLE_RATE}s)`);
    console.log(`Hop size: ${hopSize} samples (${hopSize/SAMPLE_RATE}s)`);
    console.log(`Overlap: ${((chunkSize - hopSize) / chunkSize * 100).toFixed(1)}%`);
    
    // Plan chunks
    const chunks = planChunks(TOTAL_FRAMES, chunkSize, hopSize);
    console.log(`Planned ${chunks.length} chunks`);
    
    // Process chunks (without ML model - just pass through)
    const processedChunks = [];
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const chunkData = originalAudio.slice(chunk.start, chunk.end);
        
        // For this test, we're not using the ML model, just passing through
        // In a real scenario, this would go through the Spleeter model
        processedChunks.push(chunkData);
        
        if (i < 5 || i === chunks.length - 1) {
            console.log(`Chunk ${i + 1}: samples ${chunk.start}-${chunk.end} (${chunkData.length} samples)`);
        }
    }
    
    // Reassemble using overlap-add
    console.log('\\nReassembling audio using overlap-add...');
    const reassembledAudio = overlapAddStitchMono(processedChunks, TOTAL_FRAMES, chunkSize, hopSize);
    
    return reassembledAudio;
}

// Compare original and processed audio
function compareAudio(original, processed) {
    console.log('\\nComparing original and processed audio...');
    
    if (original.length !== processed.length) {
        console.error(`Length mismatch: original=${original.length}, processed=${processed.length}`);
        return;
    }
    
    let maxDiff = 0;
    let sumSqDiff = 0;
    let sumSqOriginal = 0;
    let zeroCount = 0;
    let significantDiffCount = 0;
    const threshold = 1e-6; // Threshold for significant difference
    
    for (let i = 0; i < original.length; i++) {
        const diff = Math.abs(original[i] - processed[i]);
        maxDiff = Math.max(maxDiff, diff);
        sumSqDiff += diff * diff;
        sumSqOriginal += original[i] * original[i];
        
        if (diff < threshold) zeroCount++;
        if (diff > threshold) significantDiffCount++;
    }
    
    const rmsDiff = Math.sqrt(sumSqDiff / original.length);
    const rmsOriginal = Math.sqrt(sumSqOriginal / original.length);
    const relativeError = rmsOriginal > 0 ? (rmsDiff / rmsOriginal) * 100 : 0;
    const similarity = 100 - relativeError;
    
    console.log(`\\nComparison Results:`);
    console.log(`Total samples: ${original.length}`);
    console.log(`Maximum difference: ${maxDiff.toExponential(6)}`);
    console.log(`RMS difference: ${rmsDiff.toExponential(6)}`);
    console.log(`RMS original: ${rmsOriginal.toExponential(6)}`);
    console.log(`Relative error: ${relativeError.toFixed(6)}%`);
    console.log(`Similarity: ${similarity.toFixed(6)}%`);
    console.log(`Samples with negligible difference (< ${threshold}): ${zeroCount} (${(zeroCount/original.length*100).toFixed(2)}%)`);
    console.log(`Samples with significant difference (> ${threshold}): ${significantDiffCount} (${(significantDiffCount/original.length*100).toFixed(2)}%)`);
    
    // Check specific time points where we placed beeps
    const checkTimes = [2, 5, 8, 12, 15, 18, 22, 25, 28]; // beep times
    console.log(`\\nChecking specific beep locations:`);
    
    checkTimes.forEach(time => {
        const sampleIndex = Math.floor(time * SAMPLE_RATE);
        if (sampleIndex < original.length) {
            const origVal = original[sampleIndex];
            const procVal = processed[sampleIndex];
            const diff = Math.abs(origVal - procVal);
            console.log(`Time ${time}s (sample ${sampleIndex}): original=${origVal.toFixed(6)}, processed=${procVal.toFixed(6)}, diff=${diff.toExponential(6)}`);
        }
    });
    
    return {
        maxDiff,
        rmsDiff,
        relativeError,
        similarity,
        significantDiffPercentage: (significantDiffCount / original.length) * 100
    };
}

// Main function
async function main() {
    try {
        console.log('Audio Chunking Test');
        console.log('====================');
        console.log(`Duration: ${DURATION} seconds`);
        console.log(`Sample rate: ${SAMPLE_RATE} Hz`);
        console.log(`Total frames: ${TOTAL_FRAMES}`);
        
        // Generate test audio
        const originalAudio = generateTestAudio();
        
        // Save original audio
        const originalWav = float32ToWav(originalAudio);
        fs.writeFileSync('test_original.wav', originalWav);
        console.log('\\nSaved original audio to: test_original.wav');
        
        // Process through chunking
        const processedAudio = processAudioChunking(originalAudio);
        
        // Save processed audio
        const processedWav = float32ToWav(processedAudio);
        fs.writeFileSync('test_processed.wav', processedWav);
        console.log('Saved processed audio to: test_processed.wav');
        
        // Compare
        const comparison = compareAudio(originalAudio, processedAudio);
        
        // Summary
        console.log('\\n' + '='.repeat(50));
        console.log('TEST SUMMARY');
        console.log('='.repeat(50));
        console.log(`Audio similarity: ${comparison.similarity.toFixed(4)}%`);
        console.log(`Samples with significant differences: ${comparison.significantDiffPercentage.toFixed(4)}%`);
        
        if (comparison.similarity > 99.9) {
            console.log('✅ EXCELLENT: Audio reconstruction is nearly perfect');
        } else if (comparison.similarity > 99) {
            console.log('✅ GOOD: Audio reconstruction is very accurate');
        } else if (comparison.similarity > 95) {
            console.log('⚠️  FAIR: Audio reconstruction has some noticeable differences');
        } else {
            console.log('❌ POOR: Audio reconstruction has significant issues');
        }
        
        console.log('\\nFiles created:');
        console.log('- test_original.wav: Original diverse test audio');
        console.log('- test_processed.wav: Audio after chunking and reassembly');
        console.log('\\nYou can listen to these files to hear any differences.');
        
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

// Run the test
main();
