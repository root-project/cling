#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------

include(ExternalProject)

set(clad_install_dir ${CMAKE_BINARY_DIR}/etc/cling/)
# Specify include dirs for clad
set(CLAD_INCLUDE_DIRS ${clad_install_dir})
# Clad Libraries
set(_CLAD_LIBRARY_PATH ${clad_install_dir}/plugins/lib)

# build byproducts only needed by Ninja
if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
  set(CLAD_BYPRODUCTS
     ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladPlugin${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladDifferentiator${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
endif()

set(_clad_build_type ${CMAKE_CFG_INTDIR})
if(MSVC)
  if (winrtdebug)
    set(_clad_build_type Debug)
  else()
    set(_clad_build_type Release)
  endif()
endif(MSVC)

if(APPLE)
  set(_clad_extra_cmake_args -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()

if (CMAKE_CXX_STANDARD)
  list(APPEND _clad_extra_cmake_args -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD})
endif(CMAKE_CXX_STANDARD)

if (Clang_DIR)
  list(APPEND _clad_extra_cmake_args -DClang_DIR=${Clang_DIR})
endif(Clang_DIR)

if (LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN)
  list(APPEND _clad_extra_cmake_args -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=${LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN})
endif(LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN)

if (CLING_INCLUDE_TESTS)
  list(APPEND _clad_extra_cmake_args -DLLVM_EXTERNAL_LIT=${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
else()
  list(APPEND _clad_extra_cmake_args -DCLAD_BUILD_STATIC_ONLY=ON)
endif()

# Wrap download, configure and build steps in a script to log output
set(_clad_cmake_logging_settings
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
)

list(APPEND _clad_cmake_logging_settings LOG_OUTPUT_ON_FAILURE ON)

ExternalProject_Add(
  clad
  GIT_REPOSITORY https://github.com/vgvassilev/clad.git
  GIT_TAG v0.9
  UPDATE_COMMAND ""
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${clad_install_dir}/plugins
             -DLLVM_DIR=${LLVM_BINARY_DIR}
             -DCLANG_INCLUDE_DIRS=${CLANG_INCLUDE_DIRS}
             ${_clad_extra_cmake_args}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${_clad_build_type}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . --config ${_clad_build_type} --target install
  BUILD_BYPRODUCTS ${CLAD_BYPRODUCTS}
  ${_clad_cmake_logging_settings}
  # We need the target clangBasic to be built before building clad. However, we
  # support building prebuilt clang and adding clangBasic breaks this case.
  # Delegate the dependency resolution to the clingInterpreter target (which
  # will always depend on clangBasic).
  DEPENDS clingInterpreter
  )

# Register cladPlugin, cladDifferentiator
foreach (lib cladPlugin cladDifferentiator)
  add_library(${lib} IMPORTED STATIC GLOBAL)
  add_dependencies(${lib} clad)
endforeach()

set_property(TARGET cladPlugin PROPERTY IMPORTED_LOCATION ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladPlugin${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET cladDifferentiator PROPERTY IMPORTED_LOCATION ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladDifferentiator${CMAKE_STATIC_LIBRARY_SUFFIX})
