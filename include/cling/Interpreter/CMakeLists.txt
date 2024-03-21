include_directories(${LLVM_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS ClingOptions.td)
tablegen(LLVM ClingOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(ClingDriverOptions)
