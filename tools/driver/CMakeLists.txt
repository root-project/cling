#set( LLVM_LINK_COMPONENTS
# jit
# linker
# native
#  )

add_cling_executable(cling
  RequiredSymbols.cpp
  cling.cpp)

target_link_libraries(cling
  clingUserInterface
  clingMetaProcessor
  clingInterpreter
  clingUtils
  clangFrontend
  clangSerialization
  clangDriver
  clangCodeGen
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangLex
  clangBasic
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
