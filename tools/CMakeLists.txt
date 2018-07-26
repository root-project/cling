#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../lib/UserInterface/textinput
   OR CLING_INCLUDE_TESTS)
  add_subdirectory(driver)
  add_subdirectory(Jupyter)
  add_subdirectory(libcling)
  add_subdirectory(demo)
endif()

add_subdirectory(plugins)
