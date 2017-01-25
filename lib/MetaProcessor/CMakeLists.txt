#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set( LLVM_LINK_COMPONENTS
  core
  support
)

add_cling_library(clingMetaProcessor OBJECT
  Display.cpp
  InputValidator.cpp
  MetaLexer.cpp
  MetaParser.cpp
  MetaProcessor.cpp
  MetaSema.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangLex
  clangAST
  clangBasic

  clingInterpreter
  clingUtils
)
