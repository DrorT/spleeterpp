add_library(spleeter_wasm STATIC
  bridge.cc
)

# Emscripten-specific link flags for a JS glue output will be set at the top level when creating an executable if needed.

target_compile_definitions(spleeter_wasm PRIVATE SPLEETER_WASM_BUILD)

target_link_libraries(spleeter_wasm PRIVATE)

# Create a no-entry JS/WASM output exposing our KEEPALIVE functions
# Build JS/WASM via custom em++ invocation to ensure flags and inputs are correct
set(SPLEETER_WASM_JS ${CMAKE_CURRENT_BINARY_DIR}/spleeter.js)
set(SPLEETER_WASM_BIN ${CMAKE_CURRENT_BINARY_DIR}/spleeter.wasm)

set(EM_LINK_FLAGS
  -s MODULARIZE=1
  -s EXPORT_ES6=1
  -s ENVIRONMENT=web
  -s WASM_BIGINT=1
  -s ALLOW_MEMORY_GROWTH=1
  -s INITIAL_MEMORY=536870912
  -s EXPORTED_RUNTIME_METHODS=[cwrap,ccall,getValue,setValue,HEAPF32,HEAPU8,lengthBytesUTF8,stringToUTF8]
  -s EXPORTED_FUNCTIONS=[_sp_init,_sp_alloc,_sp_process_chunk,_sp_finalize,_sp_version]
)

add_custom_command(
  OUTPUT ${SPLEETER_WASM_JS} ${SPLEETER_WASM_BIN}
  COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/bridge.cc -I${PROJECT_SOURCE_DIR}/src -O3 ${EM_LINK_FLAGS} -o ${SPLEETER_WASM_JS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bridge.cc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Building spleeter.js/.wasm with em++"
  VERBATIM
)

add_custom_target(spleeter_wasm_module ALL
  DEPENDS ${SPLEETER_WASM_JS} ${SPLEETER_WASM_BIN}
)

# Copy artifacts to web/wasm for local serving
set(WEB_WASM_DIR ${PROJECT_SOURCE_DIR}/web/wasm)
add_custom_command(TARGET spleeter_wasm_module POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${WEB_WASM_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPLEETER_WASM_JS} ${WEB_WASM_DIR}/spleeter.js
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPLEETER_WASM_BIN} ${WEB_WASM_DIR}/spleeter.wasm
  COMMENT "Copying spleeter.js/.wasm to web/wasm"
)
