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
  binaryformat
  )

if (LLVM_LIB_IS_SHARED)
    set(LIBS clang-cpp)
else()
    set(LIBS
        clangLex
        clangAST
        clangBasic)
endif()

add_cling_library(clingMetaProcessor OBJECT
  Display.cpp
  InputValidator.cpp
  MetaLexer.cpp
  MetaParser.cpp
  MetaProcessor.cpp
  MetaSema.cpp

  LINK_LIBS
  ${LIBS}
  clingInterpreter
  clingUtils
)
