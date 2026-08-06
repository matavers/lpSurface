set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# path to parent directory (OCCT/)
get_filename_component(OCCT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

set(OCCT_ROOT "${OCCT_DIR}/opencascade-8.0.0-vc14-64-combined/opencascade-8.0.0-vc14-64")
set(OCCT_INC_DIR "${OCCT_ROOT}/inc")
set(OCCT_LIB_DIR "${OCCT_ROOT}/win64/vc14/lib")
set(OCCT_DLL_DIR "${OCCT_ROOT}/win64/vc14/bin")
set(OCCT_3RD_DIR "${OCCT_DIR}/opencascade-8.0.0-vc14-64-combined/3rdparty-vc14-64")

file(GLOB TEST_SOURCES *.cpp)
foreach(f ${TEST_SOURCES})
    get_filename_component(exe ${f} NAME_WE)
    add_executable(${exe} ${f})
    target_include_directories(${exe} PRIVATE "${OCCT_INC_DIR}")
    target_link_directories(${exe} PRIVATE "${OCCT_LIB_DIR}")
    target_link_libraries(${exe} PRIVATE
        TKernel TKMath TKGeomBase TKGeomAlgo
        TKG2d TKG3d TKBRep TKTopAlgo TKPrim
        TKService TKV3d TKOpenGl)

    # copy OCCT DLLs + 3rdparty DLLs
    add_custom_command(TARGET ${exe} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${OCCT_DLL_DIR}" "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/freetype-2.13.3-x64/bin/freetype.dll"           "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/freeimage-3.18.0-x64/bin/FreeImage.dll"         "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/tbb-2021.13.0-x64/bin/tbb12.dll"               "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/tbb-2021.13.0-x64/bin/tbbmalloc.dll"           "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/jemalloc-vc14-64/bin/jemalloc.dll"              "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/openvr-1.14.15-64/bin/win64/openvr_api.dll"     "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/ffmpeg-3.3.4-64/bin/avformat-57.dll"           "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/ffmpeg-3.3.4-64/bin/avcodec-57.dll"            "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/ffmpeg-3.3.4-64/bin/avutil-55.dll"             "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/ffmpeg-3.3.4-64/bin/avdevice-57.dll"           "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/ffmpeg-3.3.4-64/bin/avfilter-6.dll"            "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OCCT_3RD_DIR}/ffmpeg-3.3.4-64/bin/swscale-4.dll"             "$<TARGET_FILE_DIR:${exe}>"
        COMMAND ${CMAKE_COMMAND} -E echo "DLLs deployed for ${exe}"
    )
endforeach()
