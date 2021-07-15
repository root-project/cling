#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------
if ((CLING_INCLUDE_TESTS OR CLING_BUILD_PLUGINS) AND NOT MSVC)
  add_subdirectory(example)
endif()

if (CLING_BUILD_PLUGINS)
#  # If we drop a compatible cmake project in this folder we should automatically
#  # pick it up and build it.
#  function(LISTSUBDIRS result curdir)
#    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
#    set(dirlist "")
#    foreach(child ${children})
#      if(IS_DIRECTORY ${curdir}/${child})
#        list(APPEND dirlist ${child})
#      endif()
#      endforeach()
#      set(${result} ${dirlist} PARENT_SCOPE)
#  endfunction()
#
#  LISTSUBDIRS(subdirs ${CMAKE_CURRENT_SOURCE_DIR}/plugins/)
#  foreach(subdir ${SUBDIRS})
#    add_subdirectory(${subdir})
#  endforeach()
  if(NOT DEFINED NO_CONNECTION OR NOT NO_CONNECTION)
    add_subdirectory(clad)
  endif()
endif(CLING_BUILD_PLUGINS)
