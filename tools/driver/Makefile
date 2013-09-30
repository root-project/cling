##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..
CLING_LEVEL := ../..

TOOLNAME = cling

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_SRC_DIR)/../../../clang/include -I$(PROJ_SRC_DIR)/../../../clang/lib \
-I$(PROJ_OBJ_DIR)/../../../clang/include
CXXFLAGS = -fno-rtti

# Keep exports to resolve symbols in JIT
#TOOL_NO_EXPORTS = 1

LINK_COMPONENTS := jit linker native bitwriter bitreader instrumentation option
USEDLIBS = clingUserInterface.a clingMetaProcessor.a clingInterpreter.a clingUITextInput.a clingUtils.a clangFrontend.a clangSerialization.a clangDriver.a clangCodeGen.a clangParse.a clangSema.a clangEdit.a clangAnalysis.a clangAST.a clangLex.a clangBasic.a

include $(LEVEL)/Makefile.common

ifeq ($(TARGET_OS),Cygwin)
# cygwin, export all symbols
LDFLAGS += -Wl,--export-all-symbols
endif
