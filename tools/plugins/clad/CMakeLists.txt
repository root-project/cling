#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------

set(_clad_byproduct_binary_dir
  ${CMAKE_CURRENT_BINARY_DIR}/clad-prefix/src/clad-build/)
set(CLADDIFFERENTIATOR_LIB
  ${_clad_byproduct_binary_dir}/lib/Differentiator/${CMAKE_STATIC_LIBRARY_PREFIX}cladDifferentiator${CMAKE_STATIC_LIBRARY_SUFFIX}
  ${_clad_byproduct_binary_dir}/tools/${CMAKE_STATIC_LIBRARY_PREFIX}cladPlugin${CMAKE_STATIC_LIBRARY_SUFFIX}
  )

if(MSVC)
  if (winrtdebug)
    set(_clad_build_type Debug)
  else()
    set(_clad_build_type Release)
  endif()
  ExternalProject_Add(
    clad
    GIT_REPOSITORY https://github.com/vgvassilev/clad.git
    GIT_TAG v0.4
    UPDATE_COMMAND ""
    CMAKE_ARGS -G ${CMAKE_GENERATOR} -DCLAD_BUILD_STATIC_ONLY=ON
               -DCMAKE_INSTALL_PREFIX=${CLING_PLUGIN_INSTALL_PREFIX}
               -DCLAD_PATH_TO_LLVM_BUILD=${LLVM_BINARY_DIR}
               -DCLANG_INCLUDE_DIRS=${CLANG_INCLUDE_DIRS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${_clad_build_type}
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    # We need the target clangBasic to be built before building clad. However, we
    # support building prebuilt clang and adding clangBasic breaks this case.
    # Delegate the dependency resolution to the clingInterpreter target (which
    # will always depend on clangBasic).
    DEPENDS clingInterpreter
  )
else()
  set(_clad_build_type ${CMAKE_CFG_INTDIR})
  if(APPLE)
    set(_clad_extra_cmake_args
      -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
    )
  endif()

  ExternalProject_Add(
    clad
    GIT_REPOSITORY https://github.com/vgvassilev/clad.git
    GIT_TAG v0.4
    UPDATE_COMMAND ""
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${CLING_PLUGIN_INSTALL_PREFIX}
               -DCLAD_PATH_TO_LLVM_BUILD=${LLVM_BINARY_DIR}
               -DCLANG_INCLUDE_DIRS=${CLANG_INCLUDE_DIRS}
               -DCLAD_BUILD_STATIC_ONLY=ON
               ${_clad_extra_cmake_args}
    BUILD_BYPRODUCTS ${CLADDIFFERENTIATOR_LIB}
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    # We need the target clangBasic to be built before building clad. However, we
    # support building prebuilt clang and adding clangBasic breaks this case.
    # Delegate the dependency resolution to the clingInterpreter target (which
    # will always depend on clangBasic).
    DEPENDS clingInterpreter
  )
endif()

# Specify include dirs for clad
ExternalProject_Get_Property(clad source_dir)
set(CLAD_INCLUDE_DIRS ${source_dir}/clad/include/)

# Libraries
ExternalProject_Get_Property(clad binary_dir)
set(_CLAD_LIBRARY_PATH ${binary_dir})

# Register cladPlugin, cladDifferentiator
foreach (lib cladPlugin cladDifferentiator)
  add_library(${lib} IMPORTED STATIC GLOBAL)
  add_dependencies(${lib} clad)
endforeach()
set_property(TARGET cladPlugin PROPERTY IMPORTED_LOCATION ${_CLAD_LIBRARY_PATH}/tools/${_clad_build_type}/${CMAKE_STATIC_LIBRARY_PREFIX}cladPlugin${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET cladDifferentiator PROPERTY IMPORTED_LOCATION ${_CLAD_LIBRARY_PATH}/lib/Differentiator/${_clad_build_type}/${CMAKE_STATIC_LIBRARY_PREFIX}cladDifferentiator${CMAKE_STATIC_LIBRARY_SUFFIX})
