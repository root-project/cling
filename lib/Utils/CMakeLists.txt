#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set(LLVM_USED_LIBS
  clangCodeGen
  clangBasic
  )

add_cling_library(clingUtils
  AST.cpp
  )

if ( MSVC )
  set_target_properties(clingUtils PROPERTIES
  COMPILE_FLAGS "/D_HAS_EXCEPTIONS=0")
endif ()
