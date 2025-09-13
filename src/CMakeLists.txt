if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  add_subdirectory(spleeter_common)
endif()
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (${spleeter_enable_filter})
    add_subdirectory(artff)
    add_subdirectory(spleeter_filter)
  endif()
  add_subdirectory(spleeter)
endif()

# Command-line interface executable
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
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
else()
  add_subdirectory(wasm)
endif()
