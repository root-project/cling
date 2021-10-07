#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)

# Keep symbols for JIT resolution
set(LLVM_NO_DEAD_STRIP 1)

# Cling needs at least C++11; so does this demo.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_PROJECT_NAME)
  # Building as part of cling; all CMake variables are set.
else()
  # Building as separate project.
  project(cling-demo)

  # This project needs cling.
  find_package(cling REQUIRED)
endif()

# The project has one binary:
add_executable(cling-demo cling-demo.cpp)

# ...which links against clingInterpreter (and its dependencies).
target_link_libraries(cling-demo clingInterpreter)

# Provide LLVMDIR to cling-demp.cpp:
target_compile_options(cling-demo PUBLIC -DLLVMDIR="${LLVM_INSTALL_PREFIX}" -I${LLVM_INSTALL_PREFIX}/include)

set_target_properties(cling-demo
  PROPERTIES ENABLE_EXPORTS 1)

if(MSVC)
  set_target_properties(cling-demo PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
  set_property(TARGET cling-demo APPEND_STRING PROPERTY LINK_FLAGS " /EXPORT:?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00D_K@Z")
  set_property(TARGET cling-demo APPEND_STRING PROPERTY LINK_FLAGS " /EXPORT:?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00DM@Z")
  set_property(TARGET cling-demo APPEND_STRING PROPERTY LINK_FLAGS " /EXPORT:cling_runtime_internal_throwIfInvalidPointer")
endif()
