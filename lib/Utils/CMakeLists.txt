set(LLVM_USED_LIBS
  clangCodeGen
  clangBasic
  )

add_cling_library(clingUtils
  AST.cpp
  )
