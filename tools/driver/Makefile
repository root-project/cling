##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
# 
#                     The Cling Interpreter
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..
CLING_LEVEL := ../..

TOOLNAME = cling

CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_SRC_DIR)/../../../clang/include -I$(PROJ_SRC_DIR)/../../../clang/lib \
-I$(PROJ_OBJ_DIR)/../../../clang/include
CXXFLAGS = -fno-rtti

# Keep symbols for JIT resolution
NO_DEAD_STRIP := 1
KEEP_SYMBOLS := 1

LINK_COMPONENTS := bitwriter mcjit orcjit native option ipo profiledata instrumentation objcarcopts
USEDLIBS = clingUserInterface.a clingMetaProcessor.a clingInterpreter.a clingUITextInput.a clingUtils.a clangFrontend.a clangSerialization.a clangDriver.a clangCodeGen.a clangParse.a clangSema.a clangEdit.a clangAnalysis.a clangAST.a clangLex.a clangBasic.a

include $(LEVEL)/Makefile.common

ifeq ($(TARGET_OS),Cygwin)
# cygwin, export all symbols
LDFLAGS += -Wl,--export-all-symbols
endif
