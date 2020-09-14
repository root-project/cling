#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------

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

if(MSVC)
  if (winrtdebug)
    set(_clad_build_type Debug)
  else()
    set(_clad_build_type Release)
  endif()
  ExternalProject_Add(
    clad
    GIT_REPOSITORY https://github.com/vgvassilev/clad.git
    GIT_TAG v0.7
    UPDATE_COMMAND ""
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
               -DCMAKE_INSTALL_PREFIX=${clad_install_dir}/plugins
               -DCLAD_PATH_TO_LLVM_BUILD=${LLVM_BINARY_DIR}

               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
               -DLLVM_DIR=${LLVM_DIR}
               -DClang_DIR=${Clang_DIR}
               -DCLANG_INCLUDE_DIRS=${CLANG_INCLUDE_DIRS}
               -DCLAD_BUILD_STATIC_ONLY=ON

    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${_clad_build_type}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --config ${_clad_build_type} --target install
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
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
    GIT_TAG v0.7
    UPDATE_COMMAND ""
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               -DLLVM_DIR=${LLVM_DIR}
               -DClang_DIR=${Clang_DIR}
               -DCLANG_INCLUDE_DIRS=${CLANG_INCLUDE_DIRS}
               -DCLAD_BUILD_STATIC_ONLY=ON
               ${_clad_extra_cmake_args}
    INSTALL_COMMAND env DESTDIR=${CMAKE_BINARY_DIR}/plugins ${CMAKE_COMMAND} --build . --target install
               COMMAND  ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/plugins/${CMAKE_INSTALL_PREFIX}/ ${clad_install_dir}/plugins
    BUILD_BYPRODUCTS ${CLAD_BYPRODUCTS}
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    # We need the target clangBasic to be built before building clad. However, we
    # support building prebuilt clang and adding clangBasic breaks this case.
    # Delegate the dependency resolution to the clingInterpreter target (which
    # will always depend on clangBasic).
    DEPENDS clingInterpreter
  )
endif()

# Register cladPlugin, cladDifferentiator
foreach (lib cladPlugin cladDifferentiator)
  add_library(${lib} IMPORTED STATIC GLOBAL)
  add_dependencies(${lib} clad)
endforeach()

set_property(TARGET cladPlugin PROPERTY IMPORTED_LOCATION ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladPlugin${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET cladDifferentiator PROPERTY IMPORTED_LOCATION ${_CLAD_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}cladDifferentiator${CMAKE_STATIC_LIBRARY_SUFFIX})
