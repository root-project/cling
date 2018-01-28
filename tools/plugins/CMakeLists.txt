#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

## If we drop a compatible cmake project in this folder we should automatically
## pick it up and build it.
#function(LISTSUBDIRS result curdir)
#  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
#  set(dirlist "")
#  foreach(child ${children})
#    if(IS_DIRECTORY ${curdir}/${child})
#      if(APPEND dirlist ${child})
#    endif()
#  endforeach()
#  set(${result} ${dirlist} PARENT_SCOPE)
#endfunction()
#
#LISTSUBDIRS(subdirs ${CMAKE_CURRENT_SOURCE_DIR}/plugins/)
#foreach(subdir ${SUBDIRS})
#  add_subdirectory(${subdir})
#endforeach()

ExternalProject_Add(
  clad
  GIT_REPOSITORY https://github.com/vgvassilev/clad.git
  GIT_TAG master
  UPDATE_COMMAND ""
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${CLING_PLUGIN_INSTALL_PREFIX}
             -DCLAD_PATH_TO_LLVM_BUILD=${LLVM_DIR}
  #BUILD_BYPRODUCTS ${_gtest_byproducts}
  # Wrap download, configure and build steps in a script to log output
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  )
