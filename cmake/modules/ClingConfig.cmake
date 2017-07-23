# This file allows users to call find_package(Cling) and pick up our targets.

# Cling doesn't have any CMake configuration settings yet because it mostly
# uses LLVM's. When it does, we should move this file to ClingConfig.cmake.in
# and call configure_file() on it.

# Don't just use any llvm / clang: cling needs its own:
find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${CMAKE_CURRENT_LIST_DIR}/../llvm")
find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${CMAKE_CURRENT_LIST_DIR}/../clang")

# Provide all our library targets to users.
include("${CMAKE_CURRENT_LIST_DIR}/ClingTargets.cmake")
