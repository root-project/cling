set(LLVM_USED_LIBS 
	clangBasic
	)

add_cling_library(clingMetaProcessor
  MetaProcessor.cpp
  InputValidator.cpp)
