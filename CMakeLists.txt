cmake_minimum_required(VERSION 3.16)
project(rulefit VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── OpenCASCADE (复用 OCCT 目录下的库) ────────────
set(OCCT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../OCCT/opencascade-8.0.0-vc14-64-combined/opencascade-8.0.0-vc14-64")
set(OCCT_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/../OCCT/opencascade-8.0.0-vc14-64-combined/3rdparty-vc14-64")

set(OCCT_MODULES
    TKernel TKMath TKGeomBase TKGeomAlgo
    TKG2d TKG3d TKMesh TKBRep TKTopAlgo
    TKBO TKService TKV3d TKOpenGl TKPrim
)
foreach(mod ${OCCT_MODULES})
    find_library(OCCT_${mod}_LIB ${mod}
        PATHS "${OCCT_ROOT}/win64/vc14/lib" NO_DEFAULT_PATH)
    if(NOT OCCT_${mod}_LIB)
        message(FATAL_ERROR "OCCT library ${mod} not found")
    endif()
endforeach()

set(OCCT_INCLUDE_DIR "${OCCT_ROOT}/inc")
set(OCCT_LIBRARIES)
foreach(mod ${OCCT_MODULES})
    list(APPEND OCCT_LIBRARIES ${OCCT_${mod}_LIB})
endforeach()

add_library(OCCT INTERFACE)
target_include_directories(OCCT INTERFACE "${OCCT_INCLUDE_DIR}")
target_link_libraries(OCCT INTERFACE ${OCCT_LIBRARIES})

# ── Eigen ────────────────────────────────────────
set(EIGEN3_ROOT "${OCCT_3RDPARTY}/vtk-9.4.1-x64/include/vtk-9.4/vtkeigen" CACHE PATH "Eigen")
if(NOT EXISTS "${EIGEN3_ROOT}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found at ${EIGEN3_ROOT}.")
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_ROOT}")

# ── Sources ─────────────────────────────────────
file(GLOB_RECURSE RULEFIT_SOURCES "src/*.cpp")

add_executable(rulefit ${RULEFIT_SOURCES})
target_include_directories(rulefit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rulefit PRIVATE OCCT Eigen3::Eigen)

if(MSVC)
    target_compile_options(rulefit PRIVATE /W3 /bigobj /utf-8)
    set_target_properties(rulefit PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:rulefit>")
endif()

# ── Post-build: copy runtime DLLs ───────────────
set(OCCT_BIN_DIR "${OCCT_ROOT}/win64/vc14/bin")
add_custom_command(TARGET rulefit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying OCCT runtime DLLs..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${OCCT_BIN_DIR}" "$<TARGET_FILE_DIR:rulefit>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY}/freetype-2.13.3-x64/bin/freetype.dll" "$<TARGET_FILE_DIR:rulefit>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY}/freeimage-3.18.0-x64/bin/FreeImage.dll" "$<TARGET_FILE_DIR:rulefit>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY}/tbb-2021.13.0-x64/bin/tbb12.dll" "$<TARGET_FILE_DIR:rulefit>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY}/tbb-2021.13.0-x64/bin/tbbmalloc.dll" "$<TARGET_FILE_DIR:rulefit>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OCCT_3RDPARTY}/jemalloc-vc14-64/bin/jemalloc.dll" "$<TARGET_FILE_DIR:rulefit>"
    COMMAND ${CMAKE_COMMAND} -E echo "Done copying DLLs."
    COMMENT "Deploying runtime dependencies"
)
