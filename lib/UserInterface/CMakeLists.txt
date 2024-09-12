#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set( LLVM_LINK_COMPONENTS
  LineEditor
  support
)

add_cling_library(clingUserInterface
  UserInterface.cpp

  LINK_LIBS
  clingMetaProcessor
  clingInterpreter
  clingUtils
)

if( MSVC )
  # Don't use Unicode in the User Interface (command prompt)
  remove_definitions(-DUNICODE -D_UNICODE)
endif()

if(UNIX)
  set_source_files_properties(UserInterface.cpp COMPILE_FLAGS "-fexceptions")
endif()
