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
set(CLAD_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(_clad_build_type ${CMAKE_CFG_INTDIR})
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if (winrtdebug)
    set(_clad_build_type Debug)
  else()
    set(_clad_build_type Release)
  endif()
  if(asan)
    set(CLAD_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_EXTRA_CXX_FLAGS}")
  endif()
  set(EXTRA_BUILD_ARGS --config ${_clad_build_type})
endif()
if(NOT _clad_build_type STREQUAL "" AND NOT _clad_build_type STREQUAL ".")
  set(EXTRA_BUILD_ARGS --config ${_clad_build_type})
endif()

if(NOT MSVC AND CMAKE_VERSION VERSION_LESS 3.31.1)
  # Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/26398
  # The problem got eventually fixed upstream in CMake.
  list(APPEND EXTRA_BUILD_ARGS "-j 1")
endif()

set(_CLAD_LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/clad-prefix/src/clad-build/${_clad_build_type}/lib${LLVM_LIBDIR_SUFFIX})

# build byproducts only needed by Ninja
if(CMAKE_GENERATOR MATCHES Ninja)
  set(CLAD_BYPRODUCTS
    ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladPlugin${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladDifferentiator${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
endif()

if(APPLE)
  set(_clad_extra_cmake_args -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()

if (CMAKE_CXX_STANDARD)
  list(APPEND _clad_extra_cmake_args -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD})
endif(CMAKE_CXX_STANDARD)

if (LLVM_FORCE_USE_OLD_TOOLCHAIN)
  list(APPEND _clad_extra_cmake_args -DLLVM_FORCE_USE_OLD_TOOLCHAIN=${LLVM_FORCE_USE_OLD_TOOLCHAIN})
endif(LLVM_FORCE_USE_OLD_TOOLCHAIN)

list(APPEND _clad_extra_cmake_args -DCLAD_BUILD_STATIC_ONLY=ON)

# Wrap download, configure and build steps in a script to log output
set(_clad_extra_settings
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)

# If the CLAD_SOURCE_DIR variable is defined in the CMake configuration, we're
# skipping the download of the repository and use the passed directory.
if (DEFINED CLAD_SOURCE_DIR)
  list(APPEND _clad_extra_settings SOURCE_DIR ${CLAD_SOURCE_DIR})
else()
  list(APPEND _clad_extra_settings GIT_REPOSITORY https://github.com/vgvassilev/clad.git)
  list(APPEND _clad_extra_settings GIT_TAG v2.0)
endif()

# list(APPEND _clad_patches_list "patch1.patch" "patch2.patch")
list(APPEND _clad_patches_list "Add-missing-exception-include.patch")
set(_clad_patch_command
      ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/interpreter/cling/tools/plugins/clad/patches <SOURCE_DIR>
      && git checkout <SOURCE_DIR>
      && git apply --ignore-space-change --ignore-whitespace ${_clad_patches_list}
      )

ExternalProject_Add(
  clad
  UPDATE_COMMAND ""
  PATCH_COMMAND ${_clad_patch_command}
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_FLAGS=${CLAD_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_INSTALL_PREFIX=${clad_install_dir}/plugins
             -DLLVM_DIR=${LLVM_DIR}
             -DClang_DIR=${CLANG_CMAKE_DIR}
             ${_clad_extra_cmake_args}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${EXTRA_BUILD_ARGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${EXTRA_BUILD_ARGS} --target install
  BUILD_BYPRODUCTS ${CLAD_BYPRODUCTS}
  ${_clad_extra_settings}
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
