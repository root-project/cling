# This file allows users to call find_package(Cling) and pick up our targets.

# Cling doesn't have any CMake configuration settings yet because it mostly
# uses LLVM's. When it does, we should move this file to ClingConfig.cmake.in
# and call configure_file() on it.

find_package(LLVM REQUIRED CONFIG)

# Provide all our library targets to users.
include("${CMAKE_CURRENT_LIST_DIR}/ClingTargets.cmake")
