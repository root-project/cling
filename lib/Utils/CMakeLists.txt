#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------
set(LLVM_LINK_COMPONENTS
  analysis
  core
  coroutines
  coverage
  executionengine
  ipo
  lto
  mc
  object
  option
  orcjit
  runtimedyld
  scalaropts
  support
  target
  transformutils
  binaryformat
  ${LLVM_TARGETS_TO_BUILD}
)

set(LIBS
  clangCodeGen
  clangDriver
  clangFrontend
  clangParse
  clangSema
  clangAnalysis
  clangEdit
  clangRewrite
  clangRewriteFrontend
  clangSerialization
  clangAST
  clangBasic
  clangLex
)

find_library(DL_LIBRARY_PATH dl)
if (DL_LIBRARY_PATH)
  list(APPEND LIBS dl)
endif()

add_cling_library(clingUtils OBJECT
  AST.cpp
  Diagnostics.cpp
  ParserStateRAII.cpp
  Output.cpp
  Paths.cpp
  PlatformPosix.cpp
  PlatformWin.cpp
  SourceNormalization.cpp
  UTF8.cpp
  Validation.cpp

  LINK_LIBS
  ${LIBS}
)

if ( MSVC )
  set_target_properties(clingUtils PROPERTIES
  COMPILE_FLAGS "/D_HAS_EXCEPTIONS=0")
endif ()
