#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

add_subdirectory(Interpreter)
add_subdirectory(MetaProcessor)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/UserInterface/textinput OR
   CLING_INCLUDE_TESTS)
   add_subdirectory(UserInterface)
endif()
add_subdirectory(Utils)
