#include <iostream>
#include <string>
#include <vector>
#include <chrono> // For timing
#include <sys/stat.h> // For mkdir
#include <sys/types.h> // For mode_t
#ifdef _WIN32
  #include <direct.h> // For _mkdir on Windows
  #define mkdir(path, mode) _mkdir(path)
#endif
#include "wave/file.h"
#include "spleeter/spleeter.h"

void WriteWaveform(const spleeter::Waveform& data, const std::string& output_path, uint32_t sample_rate, uint16_t channel_count) {
  std::vector<float> vec_data(data.size());
  std::copy(data.data(), data.data() + data.size(), vec_data.data());
  wave::File file;
  file.Open(output_path, wave::kOut);
  file.set_sample_rate(sample_rate);
  file.set_channel_number(channel_count);
  file.Write(vec_data);
}

void PrintUsage(const std::string& program_name) {
  std::cerr << "Usage: " << program_name << " <input_audio_path> <model_type> <output_folder_path>\n";
  std::cerr << "  <input_audio_path>: Path to the input WAV file.\n";
  std::cerr << "  <model_type>: Type of separation model to use (2, 4, or 5 stems).\n";
  std::cerr << "  <output_folder_path>: Path to the folder where separated stems will be saved.\n";
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string input_path = argv[1];
  std::string model_type_str = argv[2];
  std::string output_folder_path = argv[3];

  // Validate model type
  spleeter::SeparationType model_type;
  if (model_type_str == "2") {
    model_type = spleeter::TwoStems;
  } else if (model_type_str == "4") {
    model_type = spleeter::FourStems;
  } else if (model_type_str == "5") {
    model_type = spleeter::FiveStems;
  } else {
    std::cerr << "Error: Invalid model_type. Choose 2, 4, or 5.\n";
    PrintUsage(argv[0]);
    return 1;
  }

  // Create output directory if it doesn't exist
  #ifdef _WIN32
    int status = mkdir(output_folder_path.c_str());
  #else
    int status = mkdir(output_folder_path.c_str(), 0777);
  #endif

  if (status != 0 && errno != EEXIST) {
    std::cerr << "Error: Could not create output directory " << output_folder_path << "\n";
    return 1;
  }

  std::error_code err;

  // Initialize Spleeter
  // Assuming SPLEETER_MODELS is defined by CMake and points to the models directory
  spleeter::Initialize(std::string(SPLEETER_MODELS), {model_type}, err);
  if (err) {
    std::cerr << "Error initializing Spleeter: " << err.message() << "\n";
    return 1;
  }

  // Read input audio file
  wave::File input_file;
  try {
    input_file.Open(input_path, wave::kIn);
  } catch (const std::exception& e) {
    std::cerr << "Error opening input file " << input_path << ": " << e.what() << "\n";
    return 1;
  }

  std::vector<float> audio_data;
  try {
    input_file.Read(&audio_data);
  } catch (const std::exception& e) {
    std::cerr << "Error reading input file " << input_path << ": " << e.what() << "\n";
    return 1;
  }

  uint32_t sample_rate = input_file.sample_rate();
  uint16_t channel_count = input_file.channel_number();

  if (channel_count != 2) {
      // Spleeter typically expects stereo input. This is a simplification.
      // For mono, one might need to duplicate the channel.
      // For more channels, mixing down to stereo would be needed.
      std::cerr << "Warning: Spleeter++ example expects stereo (2 channel) input. File has " << channel_count << " channels. Proceeding may have unexpected results.\n";
  }
  
  auto source = Eigen::Map<spleeter::Waveform>(audio_data.data(), channel_count, audio_data.size() / channel_count);

  // Perform separation
  std::cout << "Starting stem separation...\n";
  auto start_time = std::chrono::high_resolution_clock::now();

  if (model_type == spleeter::TwoStems) {
    spleeter::Waveform vocals, accompaniment;
    spleeter::Split(source, &vocals, &accompaniment, err);
    if (err) {
      std::cerr << "Error during 2-stem separation: " << err.message() << "\n";
      return 1;
    }
    WriteWaveform(vocals, output_folder_path + "/vocals.wav", sample_rate, channel_count);
    WriteWaveform(accompaniment, output_folder_path + "/accompaniment.wav", sample_rate, channel_count);
  } else if (model_type == spleeter::FourStems) {
    spleeter::Waveform vocals, drums, bass, other;
    spleeter::Split(source, &vocals, &drums, &bass, &other, err);
    if (err) {
      std::cerr << "Error during 4-stem separation: " << err.message() << "\n";
      return 1;
    }
    WriteWaveform(vocals, output_folder_path + "/vocals.wav", sample_rate, channel_count);
    WriteWaveform(drums, output_folder_path + "/drums.wav", sample_rate, channel_count);
    WriteWaveform(bass, output_folder_path + "/bass.wav", sample_rate, channel_count);
    WriteWaveform(other, output_folder_path + "/other.wav", sample_rate, channel_count);
  } else if (model_type == spleeter::FiveStems) {
    spleeter::Waveform vocals, drums, bass, piano, other;
    spleeter::Split(source, &vocals, &drums, &bass, &piano, &other, err);
    if (err) {
      std::cerr << "Error during 5-stem separation: " << err.message() << "\n";
      return 1;
    }
    WriteWaveform(vocals, output_folder_path + "/vocals.wav", sample_rate, channel_count);
    WriteWaveform(drums, output_folder_path + "/drums.wav", sample_rate, channel_count);
    WriteWaveform(bass, output_folder_path + "/bass.wav", sample_rate, channel_count);
    WriteWaveform(piano, output_folder_path + "/piano.wav", sample_rate, channel_count);
    WriteWaveform(other, output_folder_path + "/other.wav", sample_rate, channel_count);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Stem separation completed in " << duration.count() << " ms.\n";

  if (model_type == spleeter::TwoStems) {
    std::cout << "Successfully separated into 2 stems (vocals, accompaniment) in " << output_folder_path << "\n";
  } else if (model_type == spleeter::FourStems) {
    std::cout << "Successfully separated into 4 stems (vocals, drums, bass, other) in " << output_folder_path << "\n";
  } else if (model_type == spleeter::FiveStems) {
    std::cout << "Successfully separated into 5 stems (vocals, drums, bass, piano, other) in " << output_folder_path << "\n";
  }

  return 0;
}
