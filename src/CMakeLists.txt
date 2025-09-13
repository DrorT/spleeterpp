add_subdirectory(spleeter_common)
if (${spleeter_enable_filter})
  add_subdirectory(artff)
  add_subdirectory(spleeter_filter)
endif()
add_subdirectory(spleeter)

# Command-line interface executable
add_executable(spleeter_cli
  spleeter_cli.cc
)

target_link_libraries(spleeter_cli
  PRIVATE
    spleeter
    spleeter_common
    wave
)

target_compile_definitions(spleeter_cli
  PRIVATE
    SPLEETER_MODELS="${spleeter_models_dir}"
)
