#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------
set( LLVM_LINK_COMPONENTS
  support
)

add_cling_library(clingUtils OBJECT
  AST.cpp
  Paths.cpp
  SourceNormalization.cpp
  Validation.cpp

  LINK_LIBS
  clangSema
  clangAST
  clangLex
  clangBasic
)

if ( MSVC )
  set_target_properties(clingUtils PROPERTIES
  COMPILE_FLAGS "/D_HAS_EXCEPTIONS=0")
endif ()
