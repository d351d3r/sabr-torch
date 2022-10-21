cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sabr-torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2")

add_executable(sabr-torch src/main.cpp src/sigma_SABR.hpp src/LevenbergMarquad.hpp)
target_link_libraries(sabr-torch "${TORCH_LIBRARIES}")
set_property(TARGET sabr-torch PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sabr-torch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:sabr-torch>)
endif (MSVC)
