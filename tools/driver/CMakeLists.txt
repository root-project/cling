#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  irreader
  codegen
  instrumentation
  ipo
#  linker
  selectiondag
  jit
  native
  )

# Keep symbols for JIT resolution
set(LLVM_NO_DEAD_STRIP 1)

add_cling_executable(cling
  cling.cpp)

set_target_properties(cling
  PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(cling
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangSema
  clangSerialization
  clangStaticAnalyzerCore

  clingUserInterface
  clingMetaProcessor
  clingInterpreter
  clingUtils
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
