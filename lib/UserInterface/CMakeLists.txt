#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set(LLVM_REQUIRES_EH true)

add_cling_library(clingUserInterface
  UserInterface.cpp
  textinput/Editor.cpp
  textinput/History.cpp
  textinput/KeyBinding.cpp
  textinput/Range.cpp
  textinput/SignalHandler.cpp
  textinput/StreamReader.cpp
  textinput/StreamReaderUnix.cpp
  textinput/StreamReaderWin.cpp
  textinput/TerminalConfigUnix.cpp
  textinput/TerminalDisplay.cpp
  textinput/TerminalDisplayUnix.cpp
  textinput/TerminalDisplayWin.cpp
  textinput/TextInput.cpp
  textinput/TextInputContext.cpp
)
#Look for textinput in current ROOT source tree

IF(MSVC)
  # windows.h doesn't compile with /Za
  get_target_property(NON_ANSI_COMPILE_FLAGS clingUserInterface COMPILE_FLAGS)
  if(NON_ANSI_COMPILE_FLAGS-NOFOUND)
    set(NON_ANSI_COMPILE_FLAGS "")
  endif()
  string(REPLACE "/Za" "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  # add /EHsc exception handling flag
  set(NON_ANSI_COMPILE_FLAGS "${NON_ANSI_COMPILE_FLAGS} /EHsc")
  set_target_properties(clingUserInterface PROPERTIES
  COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
ENDIF()

#add_subdirectory(textinput)
#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)
