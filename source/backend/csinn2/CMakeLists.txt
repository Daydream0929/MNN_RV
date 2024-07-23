# CSI-NN2 API
file(GLOB MNN_CSINN2_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.hpp
)

add_library(
    MNN_CSINN2 
    OBJECT
    ${MNN_CSINN2_SRCS}
)
# ========== riscv64 and riscv32
# IF (MNN_ARM82)
#     target_compile_options(MNN_CSINN2_SRCS PRIVATE -DMNN_NNAPI_ENABLED=1 -DENABLE_ARMV82)
# ELSE()
#     target_compile_options(MNN_CSINN2_SRCS PRIVATE -DMNN_NNAPI_ENABLED=1)
# ENDIF()

target_include_directories(MNN_CSINN2 PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)
