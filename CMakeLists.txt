cmake_minimum_required(VERSION 3.16)
project(distillation VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── OpenCASCADE ──────────────────────────────────
set(OpenCASCADE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/opencascade-8.0.0-vc14-64-combined/opencascade-8.0.0-vc14-64/cmake"
    CACHE PATH "Path to OpenCASCADE cmake directory")

set(OCCT_MODULES
    TKernel TKMath TKGeomBase TKGeomAlgo
    TKG2d TKG3d TKMesh TKBRep TKTopAlgo
    TKBO TKService TKV3d TKOpenGl TKPrim
)
foreach(mod ${OCCT_MODULES})
    find_library(OCCT_${mod}_LIB ${mod}
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/opencascade-8.0.0-vc14-64-combined/opencascade-8.0.0-vc14-64/win64/vc14/lib"
        NO_DEFAULT_PATH)
    if(NOT OCCT_${mod}_LIB)
        message(FATAL_ERROR "OCCT library ${mod} not found")
    endif()
endforeach()

set(OCCT_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/opencascade-8.0.0-vc14-64-combined/opencascade-8.0.0-vc14-64/inc"
)

set(OCCT_LIBRARIES)
foreach(mod ${OCCT_MODULES})
    list(APPEND OCCT_LIBRARIES ${OCCT_${mod}_LIB})
endforeach()

add_library(OCCT INTERFACE)
target_include_directories(OCCT INTERFACE "${OCCT_INCLUDE_DIR}")
target_link_libraries(OCCT INTERFACE ${OCCT_LIBRARIES})

# ── Eigen ────────────────────────────────────────
set(EIGEN3_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/opencascade-8.0.0-vc14-64-combined/3rdparty-vc14-64/vtk-9.4.1-x64/include/vtk-9.4/vtkeigen"
    CACHE PATH "Path to Eigen header-only library")
if(NOT EXISTS "${EIGEN3_ROOT}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found at ${EIGEN3_ROOT}.")
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_ROOT}"
)

# ── Sources ─────────────────────────────────────
file(GLOB_RECURSE DISTILLATION_SOURCES "src/*.cpp")

add_executable(distillation ${DISTILLATION_SOURCES})
target_include_directories(distillation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(distillation PRIVATE
    OCCT
    Eigen3::Eigen
)

# ── MSVC specific ───────────────────────────────
if(MSVC)
    target_compile_options(distillation PRIVATE /W3 /bigobj /utf-8)
    set_target_properties(distillation PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:distillation>"
    )
endif()

# ── Post-build: copy runtime DLLs ───────────────
set(OCCT_BIN_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/opencascade-8.0.0-vc14-64-combined/opencascade-8.0.0-vc14-64/win64/vc14/bin"
)
set(OCCT_3RDPARTY_BIN_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/opencascade-8.0.0-vc14-64-combined/3rdparty-vc14-64"
)

add_custom_command(TARGET distillation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying OCCT runtime DLLs..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OCCT_BIN_DIR}"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E echo "Copying 3rdparty DLLs..."
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/freetype-2.13.3-x64/bin/freetype.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/freeimage-3.18.0-x64/bin/FreeImage.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/tbb-2021.13.0-x64/bin/tbb12.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/tbb-2021.13.0-x64/bin/tbbmalloc.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/jemalloc-vc14-64/bin/jemalloc.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/openvr-1.14.15-64/bin/win64/openvr_api.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/ffmpeg-3.3.4-64/bin/avformat-57.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/ffmpeg-3.3.4-64/bin/avcodec-57.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/ffmpeg-3.3.4-64/bin/avutil-55.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/ffmpeg-3.3.4-64/bin/avdevice-57.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/ffmpeg-3.3.4-64/bin/avfilter-6.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY_BIN_DIR}/ffmpeg-3.3.4-64/bin/swscale-4.dll"
        "$<TARGET_FILE_DIR:distillation>"
    COMMAND ${CMAKE_COMMAND} -E echo "Done copying DLLs."
    COMMENT "Deploying runtime dependencies"
)
