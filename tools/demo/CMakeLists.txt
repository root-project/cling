cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cling-demo)

# This project needs cling.
find_package(cling REQUIRED)

# It has one binary:
add_executable(cling-demo cling-demo.cpp)

# ...which links against clingInterpreter and its dependencies.
target_link_libraries(cling-demo clingInterpreter)

# Provide LLVMDIR to cling-demp.cpp:
target_compile_options(cling-demo PUBLIC -DLLVMDIR="${LLVM_INSTALL_PREFIX}" -I${LLVM_INSTALL_PREFIX}/include)
