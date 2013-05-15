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

add_cling_executable(cling
  cling.cpp)

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
