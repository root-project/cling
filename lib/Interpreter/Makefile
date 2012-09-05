##===- cling/lib/Interpreter/Makefile-----------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the a C++ interpreter backend.
#
##===----------------------------------------------------------------------===##

CLING_LEVEL = ../..
LIBRARYNAME := clingInterpreter
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

# Inform the sources about their install location
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_SRC_DIR)/../../../clang/include \
-I$(PROJ_SRC_DIR)/../../../clang/lib -I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_OBJ_DIR)/../../../clang/include \
'-DCLANG_SYS_HEADERS="-I$(PROJ_SRC_DIR)../../../clang/lib/Headers"' \
'-DCLING_INSTDIR_INCL="$(DESTDIR)$(PROJ_includedir)"' \
'-DCLING_SRCDIR_INCL="$(LLVM_SRC_ROOT)/tools/cling/include"'

include $(CLING_LEVEL)/Makefile

