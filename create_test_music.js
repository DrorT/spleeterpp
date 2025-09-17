// Create a test music file with multiple frequencies and dynamics
import { writeFileSync } from 'fs';

function generateTestMusic(sampleRate = 44100, durationSeconds = 30) {
  const samples = sampleRate * durationSeconds;
  const audio = new Float32Array(samples);
  
  // Create a more complex signal that mimics real music
  // Base frequencies (musical notes)
  const frequencies = [220, 330, 440, 550, 660]; // A3, E4, A4, C#5, E5
  
  // Create different sections with varying characteristics
  const sectionDuration = durationSeconds / 4;
  
  for (let section = 0; section < 4; section++) {
    const startSample = section * sectionDuration * sampleRate;
    const endSample = (section + 1) * sectionDuration * sampleRate;
    
    for (let i = startSample; i < endSample && i < samples; i++) {
      const t = (i - startSample) / sampleRate; // Time within section
      
      let sample = 0;
      
      // Add multiple harmonics for richer sound
      frequencies.forEach((freq, index) => {
        // Vary amplitude based on section and frequency
        const baseAmplitude = 0.1 / (index + 1); // Decrease amplitude for higher harmonics
        
        // Add some variation over time
        const amplitudeModulation = 0.8 + 0.2 * Math.sin(2 * Math.PI * 0.5 * t);
        const amplitude = baseAmplitude * amplitudeModulation;
        
        // Add frequency modulation for more "musical" feel
        const freqModulation = 1 + 0.02 * Math.sin(2 * Math.PI * 2 * t);
        const frequency = freq * freqModulation;
        
        // Add the sine wave
        sample += amplitude * Math.sin(2 * Math.PI * frequency * t);
      });
      
      // Add some percussive elements (transients)
      if (i % Math.floor(sampleRate * 0.5) < 100) { // Every 0.5 seconds, add a transient
        const transient = 0.3 * Math.exp(-((i % Math.floor(sampleRate * 0.5)) / (sampleRate * 0.1)));
        sample += transient * (Math.random() - 0.5); // Noise-like transient
      }
      
      // Add some noise for realism
      sample += 0.005 * (Math.random() - 0.5);
      
      // Apply envelope to avoid clicks at section boundaries
      const envelopeSamples = Math.floor(sampleRate * 0.1); // 100ms envelope
      if (i - startSample < envelopeSamples) {
        const envelope = (i - startSample) / envelopeSamples;
        sample *= envelope;
      } else if (endSample - i < envelopeSamples) {
        const envelope = (endSample - i) / envelopeSamples;
        sample *= envelope;
      }
      
      audio[i] = Math.max(-1, Math.min(1, sample)); // Clip to [-1, 1]
    }
  }
  
  return audio;
}

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
}

console.log('Creating test music file...');
const sampleRate = 44100;
const duration = 30; // 30 seconds
const testMusic = generateTestMusic(sampleRate, duration);
saveWavFile(testMusic, sampleRate, 'test_music.wav');
console.log('Created test_music.wav');
console.log(`Duration: ${duration} seconds`);
console.log(`Sample rate: ${sampleRate} Hz`);
console.log(`Samples: ${testMusic.length}`);
