cmake_minimum_required(VERSION 3.30)
include(cmake/modules/find_library_create_target.cmake)
include(cmake/modules/FindOpenCvCustom.cmake)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER /usr/bin/gcc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "86") #set you architecture.

message(STATUS "test " ${CMAKE_C_COMPILER})

project(YoloGstForGPU VERSION 1.0  DESCRIPTION "YoloGstForGPU" LANGUAGES CXX CUDA)

include_directories("${PROJECT_SOURCE_DIR}/spdlog/include")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/include/common")
include_directories("${PROJECT_SOURCE_DIR}/videogst/include")
#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)
# opencv
include_directories(/usr/local/include/opencv4)
link_directories(/usr/local/lib)

# Gststreamer
include_directories(/usr/include/gstreamer-1.0/gst)
include_directories(/opt/nvidia/deepstream/deepstream-7.1/sources/includes)
include_directories(/usr/include/glib-2.0)
include_directories(/usr/lib/x86_64-linux-gnu/glib-2.0/include)
include_directories(/usr/include/gstreamer-1.0/)
include_directories(/usr/include/gstreamer-1.0/gst/app)

set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4" )
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" )
set(TRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
set(CUDA_LIB_DIR "/usr/local/cuda/lib64" )
set(OpenCV_LIB_DIR "/usr/local/lib" )
set(GST_LIB_DIR "/lib/x86_64-linux-gnu/")
set(GLIB_LIB_DIR "/usr/lib/x86_64-linux-gnu")
set(CUDA_TARGET_LIB "/usr/local/cuda-12.6/targets/x86_64-linux/lib")

# TensorRT
find_library_create_target(nvinfer_vc_plugin "nvinfer_vc_plugin" STATIC ${TRT_LIB_DIR})
find_library_create_target(nvinfer "nvinfer" STATIC ${TRT_LIB_DIR})
find_library_create_target(nvonnxparser "nvonnxparser" STATIC ${TRT_LIB_DIR})
find_library_create_target(nvinfer_plugin "nvinfer_plugin" STATIC ${TRT_LIB_DIR})
find_library_create_target(nvinfer_lean "nvinfer_lean" STATIC ${TRT_LIB_DIR})
set(TensorRT_LIBS nvinfer_vc_plugin nvinfer nvonnxparser nvinfer_plugin nvinfer_lean)

find_library(nvrtc_static "nvrtc_static" ${CUDA_LIB_DIR})
find_library(libnvrtc_builtins_static "nvrtc-builtins_static" ${CUDA_LIB_DIR})
find_library(libnvinfer_static "nvinfer_static" ${TRT_LIB_DIR})
find_library(libnvptxcompiler_static "nvptxcompiler_static" ${CUDA_TARGET_LIB})
find_library(nvonnxparser "nvonnxparser" ${CUDA_LIB_DIR})

set(LibTrtStatic
        ${nvrtc_static}
        ${libnvrtc_builtins_static}
        ${libnvinfer_static}
        ${libnvptxcompiler_static}
        ${nvonnxparser}
)

#cuda
find_library_create_target(cudart "cudart" SHARED ${CUDA_LIB_DIR})
find_library_create_target(libnvjpeg "nvjpeg" SHARED ${CUDA_LIB_DIR})
#opencv
FindOpenCvCustom(OpenCV_LIBS ${OpenCV_LIB_DIR})

# Gststreamer
find_library_create_target(gobject "libgobject-2.0.so" SHARED ${GST_LIB_DIR})
find_library_create_target(gstreamer "libgstreamer-1.0.so" SHARED ${GST_LIB_DIR})
find_library_create_target(gstapp "libgstapp-1.0.so" SHARED ${GST_LIB_DIR})
find_library_create_target(gstpbutils "libgstpbutils-1.0.so" SHARED ${GST_LIB_DIR} )
find_library_create_target(gstsdp "libgstsdp-1.0.so" SHARED ${GST_LIB_DIR})
find_library_create_target(glib "libglib-2.0.so" SHARED ${GLIB_LIB_DIR})
set(GST_LIBS gobject glib gstreamer gstapp gstpbutils gstsdp)

#Npp
find_library_create_target(nppc "nppc" SHARED ${CUDA_LIB_DIR})
find_library_create_target(nppig "nppig" SHARED ${CUDA_LIB_DIR})
find_library_create_target(nppicc "nppicc" SHARED ${CUDA_LIB_DIR})
find_library_create_target(nppial "nppial" SHARED ${CUDA_LIB_DIR})
find_library_create_target(nppisu "nppisu" SHARED ${CUDA_LIB_DIR})


set(ALL_LIB ${OpenCV_LIBS} ${TensorRT_LIBS} cudart libnvjpeg ${GST_LIBS})
set(ALL_INCLUDE_LIB ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} )

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/infrastructure/*.cpp
        ${PROJECT_SOURCE_DIR}/videogst/*.cpp
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/common/*.h
        ${PROJECT_SOURCE_DIR}/infrastructure/*.cu
        ${PROJECT_SOURCE_DIR}/videogst/*.cu)


# yololayer --------------------------------------------------------------------------------------------------------------
add_library(YoloToolsGPU STATIC
        ${PROJECT_SOURCE_DIR}/infrastructure/YoloToolsGPU.cu
        ${PROJECT_SOURCE_DIR}/include/CudaStream.h
        ${PROJECT_SOURCE_DIR}/videogst/include/CudaUtility.h
)
target_link_libraries(YoloToolsGPU PUBLIC ${ALL_LIB} stdc++fs)

#TRTLayer --------------------------------------------------------------------------------------------------------------
add_library(TRTLayer STATIC
        ${PROJECT_SOURCE_DIR}/infrastructure/BuilderOnnx.cpp
        ${PROJECT_SOURCE_DIR}/infrastructure/TRTEngine.cpp
)
target_include_directories(TRTLayer PUBLIC ${ALL_INCLUDE_LIB})
target_link_libraries(TRTLayer PUBLIC ${ALL_LIB} stdc++fs YoloToolsGPU)

#NvJpgEncoderlib --------------------------------------------------------------------------------------------------------------
add_library(NvJpgEncoderlib STATIC
        ${PROJECT_SOURCE_DIR}/include/NvJpgEncoder.h
        ${PROJECT_SOURCE_DIR}/infrastructure/NvJpgEncoder.cpp
)
target_include_directories(NvJpgEncoderlib PUBLIC ${ALL_INCLUDE_LIB})
target_link_libraries(NvJpgEncoderlib PUBLIC ${ALL_LIB} stdc++fs)

#GstreamerLib --------------------------------------------------------------------------------------------------------------
add_library(GstreamerLib STATIC
        ${PROJECT_SOURCE_DIR}/videogst/GstUtility.cpp
        ${PROJECT_SOURCE_DIR}/videogst/GstDecoder.cpp
        ${PROJECT_SOURCE_DIR}/videogst/GstBufferManager.cpp
        ${PROJECT_SOURCE_DIR}/include/common/FrameGpu.h
        ${PROJECT_SOURCE_DIR}/infrastructure/BufferFrameGpu.cpp
        ${PROJECT_SOURCE_DIR}/videogst/CudaYUV_NV12.cu
        ${PROJECT_SOURCE_DIR}/videogst/include/CudaYUV_NV12.h
        ${PROJECT_SOURCE_DIR}/videogst/include/cudaVector.h
        ${PROJECT_SOURCE_DIR}/include/common/magic_enum.hpp
        ${PROJECT_SOURCE_DIR}/include/EnginePipeline.h
        ${PROJECT_SOURCE_DIR}/include/SettingPipeline.h
        ${PROJECT_SOURCE_DIR}/infrastructure/MatConverter.cu
        ${PROJECT_SOURCE_DIR}/infrastructure/EnginePipeline.cpp
)
target_include_directories(GstreamerLib PUBLIC  ${GST_LIBS})
target_link_libraries(GstreamerLib PUBLIC stdc++fs ${GST_LIBS} ${OpenCV_LIBS} cudart)


# ExtensionCharp --------------------------------------------------------------------------------------------------------------
#add_library(ExtensionCharp STATIC ${SRCS} )
add_library(ExtensionCharp SHARED ${SRCS} )
target_include_directories(ExtensionCharp PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(ExtensionCharp PUBLIC ${ALL_LIB} stdc++fs ${LibTrtStatic})
target_link_libraries(ExtensionCharp PUBLIC ${ALL_LIB} stdc++fs  )

# TestRunProject --------------------------------------------------------------------------------------------------------------
#add_executable(TestRunProject ${PROJECT_SOURCE_DIR}/tests/TestRunProject.cpp
#        include/common/Common.h
#        infrastructure/BuilderOnnx.cpp)
#
#target_include_directories(TestRunProject PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(TestRunProject PUBLIC ${ALL_LIB} )


# TestExtensionCHarp --------------------------------------------------------------------------------------------------------------
add_executable(TestExtensionCharp ${PROJECT_SOURCE_DIR}/tests/TestExtensionCharp.cpp
        tests/Helper.hpp)
target_include_directories(TestExtensionCharp PUBLIC ${ALL_INCLUDE_LIB})
target_link_libraries(TestExtensionCharp PUBLIC ExtensionCharp  )

## TestTRTEngine --------------------------------------------------------------------------------------------------------------
add_executable(TestTRTEngine
        ${PROJECT_SOURCE_DIR}/tests/TestTRTEngine.cpp
        tests/Helper.hpp)
target_link_libraries(TestTRTEngine PUBLIC TRTLayer GstreamerLib NvJpgEncoderlib )


add_executable(TestNppFunction
        ${PROJECT_SOURCE_DIR}/tests/NppFunctionTest.cpp
        ${PROJECT_SOURCE_DIR}/infrastructure/NppFunction.cpp)
target_link_libraries(TestNppFunction PUBLIC ${ALL_LIB} ${OpenCV_LIBS} stdc++fs nppc nppig nppicc nppial nppisu)

# TestGstreamerLib --------------------------------------------------------------------------------------------------------------
#add_executable(TestGstDecoder  ${PROJECT_SOURCE_DIR}/tests/TestGstDecoder.cpp)
#target_link_libraries(TestGstDecoder PUBLIC  GstreamerLib)

# TestCuda --------------------------------------------------------------------------------------------------------------
#add_executable(TestCuda ${PROJECT_SOURCE_DIR}/tests/TestCuda.cu)
#target_include_directories(TestCuda PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(TestCuda PUBLIC ${ALL_LIB} stdc++fs)
#target_compile_features(TestCuda PUBLIC cxx_std_23)
#set_target_properties( TestCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# TestCuda --------------------------------------------------------------------------------------------------------------
#add_compile_options(-G)
#add_executable(TestYoloGPU ${PROJECT_SOURCE_DIR}/tests/TestYoloGPU.cu ${PROJECT_SOURCE_DIR}/infrastructure/YoloToolsGPU.cu)
#target_include_directories(TestYoloGPU PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(TestYoloGPU PUBLIC ${ALL_LIB} stdc++fs)
#target_compile_features(TestYoloGPU PUBLIC cxx_std_23)
#set_target_properties( TestYoloGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#TestMatConverter --------------------------------------------------------------------------------------------------------------
#add_compile_options(-G)
#add_executable(TestMatConverter
#        ${PROJECT_SOURCE_DIR}/tests/TestMatConverter.cu
#        ${PROJECT_SOURCE_DIR}/infrastructure/MatConverter.cu)
#target_include_directories(TestMatConverter PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(TestMatConverter PUBLIC ${ALL_LIB} stdc++fs)
#target_compile_features(TestMatConverter PUBLIC cxx_std_23)
#set_target_properties( TestMatConverter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# TestGPUMat --------------------------------------------------------------------------------------------------------------
#add_executable(TestGPUMat ${PROJECT_SOURCE_DIR}/tests/TestGPUMat.cpp)
#target_include_directories(TestGPUMat PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(TestGPUMat PUBLIC ${ALL_LIB} stdc++fs)

#TestNvJpgEncoder --------------------------------------------------------------------------------------------------------------
#add_executable(TestNvJpgEncoder ${PROJECT_SOURCE_DIR}/tests/TestNvJpgEncoder.cpp)
#target_include_directories(TestNvJpgEncoder PUBLIC ${ALL_INCLUDE_LIB})
#target_link_libraries(TestNvJpgEncoder PUBLIC ${ALL_LIB} stdc++fs NvJpgEncoderlib)