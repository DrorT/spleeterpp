# Emscripten toolchain for spleeter++
set(CMAKE_SYSTEM_NAME Emscripten)
set(CMAKE_SYSTEM_VERSION 1)

# Compilers provided by emsdk/emcmake in PATH
set(CMAKE_C_COMPILER emcc)
set(CMAKE_CXX_COMPILER em++)
set(CMAKE_AR emar)
set(CMAKE_RANLIB emranlib)

# Include upstream Emscripten CMake definitions if available
if(DEFINED ENV{EMSCRIPTEN})
  include($ENV{EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake)
endif()

# Do NOT set global link flags here to avoid polluting compiler checks.
